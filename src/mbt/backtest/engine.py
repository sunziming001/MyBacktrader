"""回测：把价格表交给引擎，取回净值曲线与成交明细。

引擎被隔离在本模块之后（ADR-0007）：数据层与信号层不依赖 backtrader。

两个入口，**同一条引擎路径**：

- :func:`run_portfolio_backtest` —— 组合入口，收一组 :class:`~mbt.data.market.MarketData`，
  多标的共享资金池，有最大持仓数与股票池约束。**新代码应当用它。**
- :func:`run_backtest` —— 单标的入口，签名字面与历史一致，内部构造单元素序列后委派给
  组合入口（最大持仓数为 1）。它**不改变既有行为**，故既有调用方与黄金值继续有效。

两者的差别只有一处，见 :func:`run_backtest` 的说明（sizer 的默认值）。
"""

from __future__ import annotations

from dataclasses import dataclass

import backtrader as bt
import pandas as pd

from mbt.data.market import MarketData
from mbt.data.panel import assemble_panel
from mbt.rules import SHIPPED_RULES_PATH, RuleTable
from mbt.screen import SCREEN_FIELDS
from mbt.universe import build_universe

from .costs import AStockBroker, AStockCommissionInfo
from .sizing import EqualWeightSizer

#: 出厂规则表路径。定义在规则层（``mbt.rules.SHIPPED_RULES_PATH``），此处转出以保持
#: 既有引用可用；「出厂表在哪」只有那一处定义。
DEFAULT_RULES_PATH = SHIPPED_RULES_PATH

#: 成交明细的列。
TRADE_COLUMNS = ("date", "size", "price", "value", "commission")

#: 未成交而终结的订单的列。**留痕**用：挂单失效、拒单、保证金不足都要能事后查到。
REJECT_COLUMNS = ("date", "symbol", "size", "status", "reason")


@dataclass(frozen=True)
class BacktestResult:
    """一次回测的产物。

    属性:
        equity_curve: 净值曲线，交易日为索引、组合总资产为值。
        trades: 成交明细，每笔成交一行（买入为正、卖出为负）。
        final_value: 期末总资产。
        rejected: **未成交而终结**的订单，每笔一行。挂单因停牌过久失效、因出池被拒、
            因资金不足被拒都在这里——只交出成交明细等于让这些事件悄无声息地发生。
    """

    equity_curve: pd.Series
    trades: pd.DataFrame
    final_value: float
    rejected: pd.DataFrame


class _EngineClock:
    """取**引擎主时钟**的混入：各标的当前日期里的最大者。

    不能用 ``self.datas[0].datetime.date(0)``：若第一个标的的历史比回测区间短（次新股、
    北交所早期、被 ``--limit`` 选中的任意一只），它的日期会**一直停在末根**，于是整条净值
    曲线的时间索引变成同一个日期的重复——而基准对齐、年化、以及所有按日期的下游计算都会
    因此算错，且**不报错**。

    实跑 CLI 时撞到过：40 个标的（北交所排在最前）跑出来的 `period.start == period.end`，
    而 `trading_days` 是 953。

    与 :class:`~mbt.backtest.costs.AStockBroker` 的主时钟是同一个道理：有 K 线的标的其日期
    正是时钟，停牌或未开始的落后，故最大值恰是时钟本身。
    """

    def _clock(self):
        latest = None
        for data in self.datas:
            if len(data) == 0:
                continue  # 该标的尚未开始（首根晚于当前 tick）
            current = data.datetime.date(0)
            if latest is None or current > latest:
                latest = current
        return latest


class _EquityRecorder(_EngineClock, bt.Analyzer):
    """逐根 K 线记录组合总资产。"""

    def start(self):
        self._dates = []
        self._values = []

    def next(self):
        self._dates.append(self._clock())
        self._values.append(self.strategy.broker.getvalue())

    def get_analysis(self):
        return pd.Series(self._values, index=pd.DatetimeIndex(self._dates), name="value")


class _FillRecorder(bt.Analyzer):
    """记录每一笔实际成交。"""

    def start(self):
        self._fills = []

    def notify_order(self, order):
        if order.status != order.Completed:
            return
        executed = order.executed
        # 由 TRADE_COLUMNS 派生键，列改名时不会静默产生 NaN 列；
        # strict=True 让字段数与列数不符时立即报错，而非静默截断
        self._fills.append(
            dict(
                zip(
                    TRADE_COLUMNS,
                    (
                        bt.num2date(executed.dt),
                        executed.size,
                        executed.price,
                        executed.value,
                        executed.comm,
                    ),
                    strict=True,
                )
            )
        )

    def get_analysis(self):
        return pd.DataFrame(self._fills, columns=list(TRADE_COLUMNS))


class _RejectionRecorder(_EngineClock, bt.Analyzer):
    """记录**未成交而终结**的订单。

    挂单失效（停牌过久）在 AC 里要求「留痕」。只把成交明细交出去，失效就查不到了——
    而「一笔单为什么没成交」恰恰是排查回测结果时最先要问的问题。
    """

    def start(self):
        self._rows = []

    def notify_order(self, order):
        if order.status == order.Completed or order.alive():
            return  # 成交的归成交明细；还活着的不算终结
        self._rows.append(
            dict(
                zip(
                    REJECT_COLUMNS,
                    (
                        self._clock(),
                        getattr(order.data, "_mbt_symbol", None),
                        order.created.size,
                        order.getstatusname(),
                        getattr(order, "_mbt_reject_reason", ""),
                    ),
                    strict=True,
                )
            )
        )

    def get_analysis(self):
        return pd.DataFrame(self._rows, columns=list(REJECT_COLUMNS))


def build_tradability(markets) -> pd.DataFrame:
    """当日具备成交条件的布尔表（有 K 线且成交量 > 0）。

    它答的是 ``data`` 自己答不了的问题：**今天这根 K 线是新的还是陈旧的**。停牌日
    backtrader 会返回上一根的陈旧 K 线，其 ``volume`` 看着完全正常，故只看 ``data``
    必然会把停牌日误判为可交易（实测确认，见 :class:`~mbt.backtest.costs.AStockBroker`）。

    涨跌停**不**在这里：它按方向区分（涨停买不进但卖得出），故归撮合层判。
    """
    volumes = pd.DataFrame({market.symbol: market.prices["volume"] for market in markets})
    closes = pd.DataFrame({market.symbol: market.prices["close"] for market in markets})
    return (closes.notna() & (volumes > 0)).astype(bool)


def run_portfolio_backtest(
    markets,
    strategy,
    *,
    cash=100_000.0,
    max_positions=None,
    rules=None,
    universe_rules=None,
    screen=None,
    sizer=None,
    sizer_options=None,
    commission=0.0,
    commission_min=0.0,
    commission_mode=None,
    slippage=0.0,
    order_expiry_ticks=5,
    **strategy_params,
):
    """对一组标的跑一次**组合**回测。

    参数:
        markets: 一组 :class:`~mbt.data.market.MarketData`。**必须已经过质检**
            （走 :func:`mbt.data.load_market_data`），本函数不重复检查——它拿到的只是一组
            价格表，无从判断跳空是公司行为还是坏数据（ADR-0005）。
        strategy: ``backtrader.Strategy`` 的子类。
        cash: 期初资金，全部标的**共享**这一个资金池。
        max_positions: 最大持仓**标的数**。``None`` 表示不限。给出时由撮合层强制：
            已持满时对**新标的**的买入被拒（对已有持仓加仓不受限）。
        rules: 规则表，``RuleTable`` 或 TOML 路径。默认取出厂表。
        universe_rules: 股票池的准入规则（:class:`~mbt.universe.UniverseRules`）。
            ``None`` 时用出厂设定（排除次新股、纳入四个板块）。

            **股票池没有开关**：本函数一律建池、一律由撮合层强制「池外不可买」。要放宽
            只能改这里的准入规则（如 ``min_bars=0``），而不是绕过它——能绕开的开关迟早
            会被打开然后忘记关。
        screen: 选股规则（:class:`~mbt.screen.Screen`）。给了它就与股票池**一同**构成
            买入闸门，且选股结果单独挂在 broker 上供策略查看（``broker.selection_mask``）。
            两种闸门分开检查，故 ``BacktestResult.rejected`` 里的 ``reason`` 能说清是
            「出池了」还是「今天没选它」。
        sizer: 持仓分配，传一个 ``backtrader.Sizer`` **子类**（不是实例）。``None`` 时用
            :class:`~mbt.backtest.sizing.EqualWeightSizer`（等权：把可用资金摊给剩余的
            持仓名额，见该类的说明）。要沿用 backtrader 的默认（每次固定股数）请传
            ``bt.sizers.FixedSize``。        sizer_options: 传给 ``sizer`` 的关键字参数。
        commission: 手续费率。``commission_mode`` 的约定同 :func:`run_backtest`。
        order_expiry_ticks: 标的连续多少个交易日无 K 线后挂单失效。

    返回:
        :class:`BacktestResult`。

    **策略如何读到三张掩码**

    引擎把三张 boolean 标的宽表挂到 broker 上，策略可以随时查：

    - ``self.broker.tradability_mask``：当日**是否具备成交条件**。停牌日为 ``False``，
      而此时 ``self.dataX.close[0]`` 返回的是**陈旧价**——读价格做决策前必须先查它，
      否则会基于几个月前的价格下单（见 ADR-0006）。
    - ``self.broker.universe_mask``：当日**是否在股票池内**。
    - ``self.broker.selection_mask``：当日**是否被选股规则选中**（未给 ``screen`` 时为
      ``None``）。它与股票池**分开**给，因为两者的处置不同：出池要清仓，只是没被选中
      则未必——合成一道闸门虽然等价，却把这份信息抹掉了。

    三者中前两张加上选股结果都会由**撮合层**强制（买入侧），策略读它们是为了自己做决定，
    不是替代约束。

    示例::

        import pandas as pd

        def next(self):
            # 掩码的索引是 DatetimeIndex，故要用 Timestamp 去取（用 datetime.date 会 KeyError）。
            today = pd.Timestamp(self.data0.datetime.date(0))
            for data in self.datas:
                if not self.broker.tradability_mask.at[today, data._name]:
                    continue          # 今天这根是陈旧的，不是新 K 线
                ...
    """
    markets = list(markets)
    if not markets:
        raise ValueError("组合回测至少要有一个标的")

    # 与单标的入口同一道守卫。原先它只在 `run_backtest` 里，于是**组合入口**（CLI 走的
    # 就是它）在 `commission_mode=None` 时会静默退回 "all_in"——替你选了一个成本口径，
    # 而两种口径算出的成本方向相反地错。
    if commission and commission_mode is None:
        raise ValueError(
            "commission > 0 时必须指定 commission_mode："
            "'all_in'（券商「全佣」，费率已含经手费与证管费）或 "
            "'net'（券商「净佣」，需另行叠加）。"
            "两种口径的成本不同，替你猜会静默算错。"
        )
    if commission_mode not in (None, "all_in", "net"):
        raise ValueError(f"commission_mode 只能是 'all_in' 或 'net'，收到 {commission_mode!r}")

    table = rules if isinstance(rules, RuleTable) else RuleTable.load(rules)

    # 复权在**这一处**统一做，且只做一次。于是选股用的面板、股票池、可交易掩码与撮合
    # 全都落在**同一条**价格序列上。
    #
    # 这不是洁癖：若面板取原始价而撮合取后复权价，同一个选股规则就在两个不同的序列上被
    # 评估——除权日的假跳空会进到动量、均线一类信号里（ADR-0003 明确否决「不复权直接用」
    # 正是为此），而成交却发生在复权后的序列上。那种错误不会报错，只会让选出来的标的与
    # 实际能成交的价格不是一回事。
    adjusted = [
        MarketData(symbol=market.symbol, prices=market.backward_adjusted(), events=())
        for market in markets
    ]

    universe_mask = build_universe(adjusted, rules=universe_rules, rule_table=table)

    selection_mask = None
    if screen is not None:
        selection_mask = screen.apply(
            assemble_panel(adjusted, SCREEN_FIELDS), universe_mask=universe_mask
        ).selected

    return _drive_engine(
        adjusted,
        strategy,
        table=table,
        universe_mask=universe_mask,
        selection_mask=selection_mask,
        cash=cash,
        max_positions=max_positions,
        sizer=sizer,
        sizer_options=sizer_options,
        commission=commission,
        commission_min=commission_min,
        commission_mode=commission_mode,
        slippage=slippage,
        order_expiry_ticks=order_expiry_ticks,
        strategy_params=strategy_params,
    )


def _drive_engine(
    markets,
    strategy,
    *,
    table,
    universe_mask,
    selection_mask,
    cash,
    max_positions,
    sizer,
    sizer_options,
    commission,
    commission_min,
    commission_mode,
    slippage,
    order_expiry_ticks,
    strategy_params,
):
    """把行情、掩码与费用装进引擎跑一次，取回产物。

    组合入口与单标的入口都走这里——AC 要求「单标的回测是组合回测的退化情形，不存在第二套
    代码路径」，故装配引擎的活只有这一处。

    ``markets`` 必须是**已复权**的行情（调用方负责）。引擎在这里不再自己复权，正是为了让
    选股面板与撮合共用同一条序列——这一点由 :func:`run_portfolio_backtest` 保证。

    ``universe_mask=None`` 表示**不设股票池闸门**，``selection_mask=None`` 表示**不设选股
    闸门**。后者只有组合入口在给了 ``screen`` 时才有，前者只有 :func:`run_backtest` 会给
    ``None``（理由见那里的说明）。两处都是内部分支，不对使用者暴露开关。
    """
    tradability = build_tradability(markets)

    cerebro = bt.Cerebro()
    for market in markets:
        data = bt.feeds.PandasData(dataname=market.prices)
        data._mbt_symbol = market.symbol
        data._name = market.symbol
        cerebro.adddata(data)

    cerebro.addstrategy(strategy, **strategy_params)
    if sizer is None:
        cerebro.addsizer(EqualWeightSizer, max_positions=max_positions)
    else:
        cerebro.addsizer(sizer, **(sizer_options or {}))

    broker = AStockBroker(
        rules=table,
        tradability=tradability,
        universe=universe_mask,
        selection=selection_mask,
        max_positions=max_positions,
        order_expiry_ticks=order_expiry_ticks,
    )
    broker.setcash(cash)
    for market in markets:
        broker.addcommissioninfo(
            AStockCommissionInfo(
                rules=table,
                commission=commission,
                commission_min=commission_min,
                commission_mode=commission_mode or "all_in",
            ),
            name=market.symbol,
        )
    if slippage:
        broker.set_slippage_perc(slippage)
    cerebro.setbroker(broker)

    cerebro.addanalyzer(_EquityRecorder, _name="equity")
    cerebro.addanalyzer(_FillRecorder, _name="fills")
    cerebro.addanalyzer(_RejectionRecorder, _name="rejections")

    runs = cerebro.run()
    run = runs[0]

    return BacktestResult(
        equity_curve=run.analyzers.equity.get_analysis(),
        trades=run.analyzers.fills.get_analysis(),
        final_value=cerebro.broker.getvalue(),
        rejected=run.analyzers.rejections.get_analysis(),
    )


def run_backtest(
    prices,
    symbol,
    strategy,
    cash=100_000.0,
    rules=None,
    commission=0.0,
    commission_min=0.0,
    commission_mode=None,
    slippage=0.0,
    **strategy_params,
):
    """对**单一标的**的价格表跑一次回测。

    它是 :func:`run_portfolio_backtest` 的**退化情形**（最大持仓数为 1），走**同一个**
    :func:`_drive_engine`，不存在第二套引擎路径。为了不改变既有行为，它有两处刻意的设定：

    - ``sizer`` 保持 backtrader 的默认（每次 1 股）而非组合入口的等权分配——改变它会让
      既有回测的期末资金全部改变，而那是既有黄金值的根基；
    - **不设股票池闸门**。调用方已经显式点名了标的，没有「池」可言——对着一个被点名的
      标的再判一次「它够不够格进池」，会把「回测这一只」变成「回测这一只，前提是它自己
      不反对」。这不是可绕过的开关：它只是本入口内部的选择，组合入口一律建池且没有参数
      能关掉（连 ``universe_rules`` 都只用于放宽条件）。

    要组合级行为请用 :func:`run_portfolio_backtest`。

    参数:
        prices: 交易日为索引、含 ``open`` / ``high`` / ``low`` / ``close`` / ``volume``
            的**字段宽表**。**本函数不做数据质检**——它拿到的是一张表，无法判断跳空是公司
            行为还是坏数据。要一份已质检的行情，请走
            :func:`mbt.data.load_market_data`（ADR-0005）。传原始价则结果未复权；
            传 :meth:`mbt.data.MarketData.backward_adjusted` 的后复权价则已复权。
        symbol: 标的符号，形如 ``sh600000``。交易制度按它取板块与 ST 状态，
            因此**必填**——缺了它就无法确定过户费与涨跌幅限制，费用会算错。它也须与
            价格表来自同一标的：本函数不校验这一点。
        strategy: ``backtrader.Strategy`` 的子类。
        cash: 期初资金。
        rules: 规则表，可以是 ``RuleTable`` 或 TOML 文件路径。默认取出厂规则表
            ``src/mbt/rules/a_share.toml``，其数值逐条带出处（见
            ``docs/research/a-share-trading-rules.md``）。
        commission: 手续费率（券商约定，如 0.0003）。
        commission_min: 单笔最低手续费。
        commission_mode: 该费率的口径，``commission > 0`` 时**必填**：

            - ``"all_in"``：券商「全佣」，费率**已含**经手费与证管费，不再叠加。
            - ``"net"``：券商「净佣」，费率**未含**，按成交日另行叠加经手费与证管费。

            不设默认值是刻意的：两种口径算出的成本不同（2026 年后单边差约 0.0054%），
            替你猜就会静默高估或低估成本。``commission == 0`` 时不起作用，可省略。
        slippage: 滑点比例，成交价按此劣化。

    .. warning::

        本函数**尚不可用于策略判断**：它只撮合一张给定的价格表，不负责取数、复权与
        数据质检。A 股制度约束已完成费用、T+1、涨跌停不可成交、无成交量不可成交，
        出厂规则表亦已落地（默认 ``rules`` 即可用）。仍缺：

        - **ST 限幅实际选不中**：出厂表不登记任何 ST 期间，故一律按非 ST 取限幅。
          本地数据不含股票名称，无法回溯历史 ST 状态，故主板 ST 的 5%（2026-07-06
          起为 10%）等规则在 ST 期间数据源到位前不会生效。
        - 少数成本项未建模（证券结算风险基金、大宗交易费率下浮），方向同为偏乐观，
          但量级远小于已建模的各项。见 README 的风险清单。
        - 股改复牌首日不设涨跌幅限制这类**历史制度空窗**不在规则表内，故 2005–2007
          年的数据做质检时可能误报（见 :mod:`mbt.data.anomaly` 的已知误报源）。
    """
    if commission and commission_mode is None:
        raise ValueError(
            "commission > 0 时必须指定 commission_mode："
            "'all_in'（券商「全佣」，费率已含经手费与证管费）或 "
            "'net'（券商「净佣」，需另行叠加）。"
            "两种口径的成本不同，替你猜会静默算错。"
        )
    if commission_mode not in (None, "all_in", "net"):
        raise ValueError(f"commission_mode 只能是 'all_in' 或 'net'，收到 {commission_mode!r}")

    market = MarketData(symbol=symbol, prices=prices, events=())

    table = rules if isinstance(rules, RuleTable) else RuleTable.load(rules)

    return _drive_engine(
        # 本入口无复权事件，故 ``prices`` 原样就是「已复权」的序列（调用方若传的是
        # `backward_adjusted()` 的结果，那正是所期望的；见参数说明）。
        [market],
        strategy,
        table=table,
        # 单标的入口不设股票池闸门，理由见本函数的说明。
        universe_mask=None,
        selection_mask=None,
        cash=cash,
        max_positions=1,
        # 沿用 backtrader 的默认 sizer（每次 1 股），以保住既有黄金值。组合入口的默认是
        # 等权分配，那是新行为，不去改既有回测的期末资金。
        sizer=bt.sizers.FixedSize,
        sizer_options={"stake": 1},
        commission=commission,
        commission_min=commission_min,
        commission_mode=commission_mode,
        slippage=slippage,
        order_expiry_ticks=5,
        strategy_params=strategy_params,
    )
