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

import array
from dataclasses import dataclass

import backtrader as bt
import numpy as np
import pandas as pd

from mbt.data.market import MarketData, backward_adjusted_markets
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
#:
#: ``value`` 是**成交金额** ``size × price``，**带符号**（买入为正、卖出为负）。
#:
#: .. warning::
#:
#:     **不要**改成 backtrader 的 ``order.executed.value``。那是「本次成交所对应的持仓
#:     **成本基础**」——买入手上是成本，而**平仓时它仍是当初的买入成本**，不是卖出所得。
#:     实测：买 100@10、卖 100@12，两行的 ``executed.value`` **都是 1000**（而卖出应当记
#:     1200）。用它会静默毁掉两处：
#:
#:     - 平仓配对（:func:`mbt.metrics.closed_trade_pnls`）用 ``value`` 反推卖出所得，
#:       于是买入价被当成卖出价，盈亏退化成「只等于手续费」——**胜率恒为 0**；
#:     - 换手率（:func:`mbt.metrics.compute_metrics`）累加 ``|value|``，数的是成本基础。
#:
#:     这个坑一度真实存在，且被测试夹具掩盖着：夹具按 ``size × price`` 造数（与本文口径
#:     一致），而引擎产出的却是 ``executed.value``——**测试与实现各说各话**，所以没被测出来。
TRADE_COLUMNS = ("date", "symbol", "size", "price", "value", "commission")

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
    """取**引擎主时钟**的日期，即「今天」。

    不能用 ``self.datas[0].datetime.date(0)``：若第一个标的的历史比回测区间短（次新股、
    北交所早期、被 ``--limit`` 选中的任意一只），它的日期会**一直停在末根**，于是整条净值
    曲线的时间索引变成同一个日期的重复——而基准对齐、年化、以及所有按日期的下游计算都会
    因此算错，且**不报错**。

    实跑 CLI 时撞到过：40 个标的（北交所排在最前）跑出来的 `period.start == period.end`，
    而 `trading_days` 是 953。

    **这个值 backtrader 每 tick 已经算好了，就在策略自己的 ``datetime`` 线上**：

    - runonce（本引擎走的那条）``Strategy._oncepost(dt)`` 里 ``dt`` 是 cerebro 取的「各标的
      下一根日期的最小者」，而它已经把 ``advance_peek() <= dt`` 的标的全部推进过——推完之后
      那个最小者恰好就是各标的当前日期的**最大者**；
    - runnext ``Strategy._clk_update`` 直接写 ``max(d.datetime[0] for d in self.datas if len(d))``。

    两条路径都在调用 ``next()`` 与各分析器的 ``next()`` **之前**把值写好，故读它与扫一遍
    ``self.datas`` 是同一件事，只是不再需要遍历全部标的（全市场 4752 只 × 2701 tick）。

    本混入只给**分析器**用（从 ``self.strategy`` 上读）。策略侧同一个值见
    ``examples.strategies.EngineClock``，撮合侧见 :meth:`AStockBroker._compute_clock`——
    那两处的等价论证与实测见 :func:`_drive_engine`。
    """

    def _clock(self):
        return self.strategy.datetime.date(0)


class _NextOnEveryBar:
    """把 ``prenext`` 接到 ``next``：**晚上市的标的不再阻塞其余标的**（票据 #31）。

    默认的 backtrader 只在**所有** data feed 都就绪之后才调 ``next()``，在那之前调
    ``prenext()``（空实现）。后果是：只消一只标的上得晚，**其余标的在整个前半段一次都不被
    策略看一眼**——而它们当时都在正常交易、也有信号。

    实测（34 个标的、区间自 2020 起）：一只 2026-08 才上市的标的，把 ``next()`` 的首次调用
    推到了第 926 根 K 线，即净值曲线的**前 3 年多全被跳过**。而净值曲线、年化、夏普都按整段
    算——**策略只动了 24 天，指标却按 4 年算**。它不报错，图上也只是「一条平线」。

    .. warning::

        **放开调用时点会同时放开一个静默危害，必须知道。** 一个**尚未上市**的标的，
        ``close[0]`` 给的不是「没有」，而是它**最后一根的价格**（即未来价）——实测乙的价格
        序列是 10,13,…,37，未上市时 ``close[0]`` 返回 37.0、``sma[0]`` 返回 31.0。

        故策略读价格前**必须**查 :attr:`AStockBroker.tradability_mask`（停牌与未上市都为
        ``False``）。这项工作本来就要做——停牌日的 ``close[0]`` 也是陈旧价——只是「忘了查」
        的后果从「拿到陈旧价」升级成了「拿到未来价」。

        这个危害在修掉本票之前**被默认的等待机制掩盖着**：``next()`` 从不早于全体就绪，
        策略便永远见不到未上市的标的。所以本改动不是引入新规矩，而是把一个既有的规矩变成了
        必须遵守。

    .. note::

        ``len(self)`` 现在是「引擎跑到第几根」，包括那些还没有任何标的就绪的日子。故
        ``if len(self) < N: return`` 这类「先观望 N 根」的写法**语义正确**——在修掉本票之前
        它会被推后到「最后一只标的上市之后 N 根」。
    """

    def prenext(self):
        self.next()


def _always_next(strategy):
    """返回一个「不等最后一只标的」的等价策略类；用户自己实现了 ``prenext`` 时原样返回。

    ``prenext`` 的常见写法就是 ``def prenext(self): self.next()``。若用户已经这么写，再包一层
    会让 ``next()`` 每根被调**两次**——那是静默的双倍交易，比不修更糟。故**显式实现优先**。
    """
    if getattr(strategy, "prenext", None) is not bt.Strategy.prenext:
        return strategy
    return type(
        strategy.__name__,
        (_NextOnEveryBar, strategy),
        {"__module__": strategy.__module__, "__qualname__": strategy.__qualname__},
    )


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
        #
        # `value` 用 `size × price`（成交金额、带符号），**不是** `executed.value`——后者是
        # 持仓的成本基础，平仓时仍按买入价计，见 TRADE_COLUMNS 的说明。
        self._fills.append(
            dict(
                zip(
                    TRADE_COLUMNS,
                    (
                        bt.num2date(executed.dt),
                        getattr(order.data, "_mbt_symbol", None),
                        executed.size,
                        executed.price,
                        executed.size * executed.price,
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


class _ProgressRecorder(_EngineClock, bt.Analyzer):
    """逐根把进度转交给上报方（:mod:`mbt.progress`）。

    **只读**：它不改订单、不动持仓、不进 ``get_analysis``，故加上它对回测结果没有任何影响。

    三个钩子 ``prenext`` / ``nextstart`` / ``next`` 都接到同一样处理上，理由见
    :meth:`_tick`。这也是为什么不用「策略里顺手报一下」：策略的 ``prenext`` 有可能被使用者
    自己定义，那时进度就会**静默漏掉**开头的若干根——而全市场回测里，全体标的最小的
    上市日把这一段拉得极长，漏掉的正是最需要看着的那一段。
    """

    params = (
        #: 上报方。引擎按 :class:`~mbt.progress.ProgressReporter` 的三个方法名调用它。
        ("reporter", None),
        #: 分母：引擎的 tick 数（各标的交易日的并集长度）。
        ("total", None),
    )

    def start(self):
        self._done = 0

    def _tick(self):
        # backtrader 按**最小周期状态**在 prenext / nextstart / next 三者中**只调一个**，
        # 故这里不会重复计数；三个都接是为了不丢开头那些根。
        self._done += 1
        # 时点以**回调**形式给出去，由上报方按节流决定要不要真的取——取一次要遍历全部标的。
        self.p.reporter.tick(self._done, self.p.total, self._clock)

    def prenext(self):
        self._tick()

    def nextstart(self):
        self._tick()

    def next(self):
        self._tick()


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
    listing_dates=None,
    screen=None,
    signals=None,
    sizer=None,
    sizer_options=None,
    commission=0.0,
    commission_min=0.0,
    commission_mode=None,
    slippage=0.0,
    order_expiry_ticks=5,
    progress=None,
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
            只能改这里的准入规则（如 ``min_trading_days=0``），而不是绕过它——能绕开的开关迟早
            会被打开然后忘记关。
        listing_dates: **真实上市日**（``{符号: 日期}``），来自
            :class:`~mbt.data.master.SecurityMasterDataSource`（票据 #25）。给了它，次新股门槛
            按「自上市日起的交易日」算；不给则回退到「本地行情根数」的近似口径。
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
        progress: 进度上报的接收端（:class:`~mbt.progress.ProgressReporter`）。``None``
            （默认）时**没有任何输出、也没有额外开销**。

            给出时，本函数会在四类时点上报：**复权** / **建股票池** / **算选股掩码**
            （逐个过滤器，由 :meth:`mbt.screen.Screen.apply` 报）/ **回测引擎**（逐根 K 线，
            按墙上时间节流）。

            它与回测结果**无关**：只读、不改变任何计算，故有进度与没进度的结果逐位相同
            （``tests/test_progress.py`` 里有一条测试盯着这一点）。它存在的唯一理由是让
            「分钟到小时量级的静默」变成可观察的（见 :mod:`mbt.progress`）。

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
    不是替代约束。**池与选股这两张按「下单那一根」判**，不按成交那一根——订单在 T 收盘下定、
    T+1 开盘成交，按下单那根判才是「策略在 T 做的决定」，按成交那根判会额外要求「T+1 仍成立」。
    见 :class:`~mbt.backtest.costs.AStockBroker` 的专节。

    示例::

        import pandas as pd

        def next(self):
            # 掩码的索引是 DatetimeIndex，故要用 Timestamp 去取（用 datetime.date 会 KeyError）。
            #
            # 「今天」取各标的当前日期的**最大者**，不要用 `self.data0.datetime.date(0)`：
            # 若第一个标的的历史比区间短（次新股、北交所早期、或 `--limit` 随手选中的某一只），
            # 它的日期会一直停在末根，于是掩码查到的是**另一天**——不报错，只是结果错。
            # 引擎内部记账用的是同一个办法（见 `_EngineClock`）。
            today = pd.Timestamp(max(data.datetime.date(0) for data in self.datas if len(data)))
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

    # 复权在**这一处**统一做，且只做一次（`backward_adjusted_markets` 是它的唯一定义）。
    # 于是选股用的面板、股票池、可交易掩码与撮合全都落在**同一条**价格序列上。
    #
    # 这不是洁癖：若面板取原始价而撮合取后复权价，同一个选股规则就在两个不同的序列上被
    # 评估——除权日的假跳空会进到动量、均线一类信号里（ADR-0003 明确否决「不复权直接用」
    # 正是为此），而成交却发生在复权后的序列上。那种错误不会报错，只会让选出来的标的与
    # 实际能成交的价格不是一回事。`mbt screen` 曾漏了这一处，同一个规则在两条路上给出不同
    # 答案（票据 #73）；两条路现在共用这个入口。
    #
    # 下面三处 ``progress.stage`` 与引擎内的那一处合起来回答「现在在哪一步」：这几段都是
    # **按标的数 × 根数**的顺序扫描，全市场尺度下每段都是分钟量级，而它们之间没有任何输出
    # （见 `mbt.progress`）。
    if progress is not None:
        progress.stage("复权", note=f"{len(markets)} 个标的")
    adjusted = backward_adjusted_markets(markets)

    if progress is not None:
        progress.stage("建股票池", note=f"{len(markets)} 个标的")
    universe_mask = build_universe(
        adjusted, rules=universe_rules, rule_table=table, listing_dates=listing_dates
    )

    selection_mask = None
    if screen is not None:
        # 不在这里再包一层 stage：`Screen.apply` 自己会按**逐个过滤器**上报，那比「正在算
        # 选股掩码」精确得多——而它内部恰恰是最慢的一段（每个过滤器各扫一遍全部标的）。
        selection_mask = screen.apply(
            _with_signals(assemble_panel(adjusted, SCREEN_FIELDS), signals),
            universe_mask=universe_mask,
            progress=progress,
        ).selected

    return _drive_engine(
        adjusted,
        strategy,
        table=table,
        universe_mask=universe_mask,
        selection_mask=selection_mask,
        signals=signals,
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
        progress=progress,
    )


def _with_signals(panel, signals):
    """把额外字段形信号并进行情面板——实现在 :func:`mbt.data.panel.with_signals`。

    留这个薄封装只为让引擎内部的调用点读起来短一点；**逻辑不在这里**，否则 CLI 的选股路径
    会需要第二份实现，而两份必然漂移。
    """
    from mbt.data.panel import with_signals

    return with_signals(panel, signals)


#: 行情表的前五列**必须是**这个顺序。``PandasDirectData`` 按**列号**取数（``itertuples``
#: 的位置），列序错了不会报错，只会把开高低收错位喂进引擎——那种错在净值曲线上看不出来，
#: 只会让每笔成交价都差一点。故这一列序是**契约**，由 ``_feed`` 强制检查。
_FEED_COLUMNS = ("open", "high", "low", "close", "volume")


class _BarFeed(bt.feeds.PandasDirectData):
    """日线（bar）模式的 feed：省掉 backtrader 的 **tick 记账**，并把**装载**改成整列拷贝。

    ## tick 记账

    ``feed.advance()`` 每标的每根都做两件与 bar 模式无关的事：``_tick_nullify()`` 把
    ``tick_open`` / ``tick_high`` / ``tick_low`` / ``tick_close`` 等清成 ``None``，紧接着
    ``_tick_fill()`` 再把刚推进到的这一根**抄**进这几个属性。整套记账只有**一个**消费者：
    ``BackBroker._try_exec`` 读 ``tick_*``，读不到就退回 ``data.open[0]`` / ``high[0]`` /
    ``low[0]`` / ``close[0]``——**正是 ``_tick_fill`` 抄进去的那几个值**。

    两者的时点也一样：cerebro 先推进标的（记账就发生在这一刻），同一 tick 内才轮到撮合，
    中间不会再有第二次推进。故对成交价**没有任何影响**，纯粹是每标的每根 ~10µs 的白开销。

    实测（232 标的）：逐根阶段 12.8s / 20.4s → 8.2s / 20.4s（省 4.6s，约 23%），净值曲线与成交
    明细与不省时**逐位相同**。装载阶段两边一样（7.6s / 7.5s）：``tick_*`` 不在 ``load()``
    里碰，只在 ``advance()`` / ``next()`` 里。

    .. warning::

        若将来升级 backtrader，要重新确认一件事：``BackBroker._try_exec`` 里那条
        「``tick_*`` 是 ``None`` 就退回 ``lines[0]``」的退路还在。它没了，这里就会静默地
        拿 ``None`` 当价格。

    ## 装载：整列拷贝（见 :meth:`preload`）

    原版 ``_load()`` 是**逐根**的 Python 循环：``itertuples()`` 取一行、对每列
    ``getattr(params, 列名)`` + ``getattr(lines, 列名)`` + ``line[0] = 值``，日期还要
    ``Timestamp.to_pydatetime()`` + ``date2num()``。全市场 980 万根 × 6 列，实测占引擎阶段
    的四成（81s / 212.9s）。这里改成把每列**一次性**拷进行缓冲，日期整列向量化。

    单独 A/B（232 标的、464,521 根真实 K 线，``.scratch/bench_preload_columns.py``）：装载
    **约 3.7s → 0.04s（约 90×）**，且两条路装出的全部缓冲**逐位相同**（含 NaN 与末尾那格
    lookahead）。按根数线性外推到全市场 980 万根是**省约 78s**，与实跑的 79s 对得上：首行
    进度从第 81s 提前到第 2s，引擎阶段 212.9s → **139.2s**，两份产物与逐根装载**逐位相同**。
    这条账按**总根数**（980 万根 × 6 列）放大，与标的数无关——与上面那组「每 tick × 每标的」
    是两种规模。

    另有一个**只读**的体检属性 ``_mbt_preload_mode``：``"columns"`` / ``"bars"`` 记下这一轮
    实际走了哪条路。它存在是为了让「整列那条路没生效」这件事**可观察**——那不会报错，只是慢
    （退回原路时引擎阶段会多出那 79s）。类属性先给 ``"bars"``（最保守的一种说法），
    :meth:`preload` 真正跑过才改成 ``"columns"``。
    """

    #: ``"bars"`` = 逐根（基类原路）；``"columns"`` = 整列拷贝。见类文档末尾。
    _mbt_preload_mode = "bars"

    def _tick_nullify(self):
        """bar 模式下没人消费 ``tick_*``，整段记账省掉（见类文档）。"""

    def _tick_fill(self, force=False):
        """同上——包括 ``force=True``（只有 resampler 会那么调，本引擎不用 resample）。"""

    def preload(self):
        """把整张行情表按**列**灌进 line 缓冲，而不是逐根 ``line[0] = 值``。

        等价性靠三件事，任何一条不成立就退回原路（:meth:`_column_preload_applies`）：

        - **行的顺序与取值**：``line.array`` 直接放一列的值，与逐根写进同一位置是同一批
          double。行的来源同样是 ``dataname`` 本身，不是别的视图。
        - **日期**：``date2num(午夜)`` = ``float(toordinal)``（``math.fsum`` 里只有一项非零），
          而 ``toordinal = 719163 + floor(ns / 86400e9)``。对真实行情（119 万个时间戳）
          逐根比过，**逐位相同**。这条公式**只对午夜成立**：它压根不看时刻，带时刻的表会被
          整天丢掉；而把时刻补回去的写法要对上 ``fsum`` 的求和顺序，逐位对不齐。故带时刻
          就退回原路（:meth:`_column_preload_applies`）。
        - **缓冲的形态**：每条线都得是 ``nrows + extension`` 格，其中末尾 ``extension`` 格是
          ``NAN``。少一格都不行：``buflen()`` 是 ``len(array) - extension``，短了引擎会少喂
          或**多喂**一根空 K 线。逐根那条路给**每条**线（哪怕表里没有这一列）每根都追加
          一格 ``NAN``——``Lines.forward()`` 是按线走的，而 ``_load()`` 只往映射到的列里写值
          （``openinterest`` 就是这种：参数 ``-1``，一路留 ``NAN``）。故这里对**没有对应列**的
          线补 ``nrows`` 格 ``NAN``，而不是留空。末尾仍走基类的 ``_last()`` + ``home()``，
          与逐根装载收尾一致。
        """
        if not self._column_preload_applies():
            self._mbt_preload_mode = "bars"
            return super().preload()

        prices = self.p.dataname
        nrows = len(prices)
        arrays = {
            "datetime": _date_column_as_number(prices.index),
            **{name: _double_array(prices[name]) for name in _FEED_COLUMNS},
        }

        for name in self.getlinealiases():
            line = getattr(self.lines, name)
            values = arrays.get(name)
            if values is None:
                # 表里没有这一列（参数为负）：逐根那条路留的就是 ``nrows`` 格 ``NAN``。
                values = array.array("d", [float("nan")] * nrows)
            line.array = values
            line.array.extend([float("nan")] * line.extension)

        self._rows = iter(())  # 缓冲已满，别再走 `_load()`（它靠 `_rows`）
        self._mbt_preload_mode = "columns"
        self._last()
        self.home()

    def _column_preload_applies(self) -> bool:
        """整列装载是否**能证明**与原路逐位相同；任何一条不确定就退回原路。"""
        # 过滤器 / 起止日期 / 时区都在 `load()` 的**逐根**路径里生效，整列拷贝绕不过去。
        if self._filters or self._ffilters or self._barstack or self._barstash:
            return False
        if self.p.fromdate is not None or self.p.todate is not None:
            return False
        # 时区：`load()` 里 `_tzinput` 那条分支会把每根时间戳 localize 一次。`_tzinput` /
        # `_tz` 由 `_start_finish()` 依 `p.tz` / `p.tzinput` 与 cerebro 的 tz 定出来；它们
        # 还没建就说明 `_start()` 没跑，那种情况下不冒这个险。
        if not hasattr(self, "_tzinput"):
            return False
        if self._tzinput or getattr(self, "_tz", None):
            return False

        # 列序契约：`open`..`volume` 必须按位落在行情表的第 1..5 列（见 `_FEED_COLUMNS`）。
        if (
            self.p.datetime,
            self.p.open,
            self.p.high,
            self.p.low,
            self.p.close,
            self.p.volume,
            self.p.openinterest,
        ) != (0, 1, 2, 3, 4, 5, -1):
            return False

        # 缓冲形态：`array.array`（UnBounded）才支持整体替换；QBuffer 是 deque。
        for line in self.lines:
            if getattr(line, "useislice", False) or line.bindings:
                return False

        index = self.p.dataname.index
        # 纳秒是这套算法的前提（`index.view('int64')` 配 `_NS_PER_DAY`）。
        if getattr(index, "dtype", None) != np.dtype("datetime64[ns]"):
            return False
        # 日期公式只产出午夜（见 `preload`）：带时刻的表退回原路。
        return bool((_nanoseconds(index) % _NS_PER_DAY == 0).all())


#: 1970-01-01 的 ``datetime.toordinal()``。``date2num`` 的基准就是 ordinal。
_EPOCH_ORDINAL = 719_163

#: 一天有多少纳秒——pandas 的时间戳内部是「自 1970 起的纳秒数」。
_NS_PER_DAY = 86_400_000_000_000


def _double_array(values) -> array.array:
    """把一列 numpy 数变成 ``array('d')``。走 ``tobytes``/``frombytes``，是 C 层的连续拷贝。"""
    contiguous = np.ascontiguousarray(values, dtype="<f8")
    out = array.array("d")
    out.frombytes(contiguous.tobytes())
    return out


def _nanoseconds(index) -> np.ndarray:
    """``DatetimeIndex`` 的自 1970 起纳秒数（``int64``）。"""
    return np.asarray(index.view("int64"))


def _date_column_as_number(index) -> array.array:
    """日期列 → backtrader 的浮点日期序号。

    **与 ``bt.date2num(那一天)`` 逐位相同**——但只对午夜时间戳成立，见 :meth:`_BarFeed.preload`。
    """
    days = _nanoseconds(index) // _NS_PER_DAY
    return _double_array(_EPOCH_ORDINAL + days)


def _feed(prices):
    """把一张行情宽表包成 backtrader 的 data feed。

    用 ``PandasDirectData`` 而不是 ``PandasData``，因为它快得多：前者走
    ``itertuples()``，后者每行每列各来一次 ``.iloc``。实测（232 只标的、全窗口）装载一段
    从 **26.5s 降到 4.0s（6.6×）**；全市场 4752 只、980 万根时装载仍是引擎阶段的**四成**
    （81s / 212.9s），是这一阶段眼下最大的一笔。两版在真实行情上产出的净值曲线与成交明细
    **逐位相同**。

    ``openinterest=-1`` 是显式的「本表没有这一列」。不写它，第 6 列（``amount``，
    成交额）会被当成持仓量读进去——今天没人看这一列，但那是**碰巧**没事，不是设计。
    """
    if tuple(prices.columns[: len(_FEED_COLUMNS)]) != _FEED_COLUMNS:
        raise ValueError(
            f"行情表的列序必须是 {_FEED_COLUMNS}（PandasDirectData 按列号取数），"
            f"收到 {tuple(prices.columns)}"
        )
    return _BarFeed(dataname=prices, openinterest=-1)


def _drive_engine(
    markets,
    strategy,
    *,
    table,
    universe_mask,
    selection_mask,
    signals,
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
    progress=None,
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

    # 引擎的每一处「每 tick 固定开销」都在这里交代清楚。它们与结果无关（下面逐条给出
    # 「为什么等价」），但会按 **标的数 × tick 数**放大：全市场是 4752 × 2701 = 1280 万次。
    # 实测（232 标的、同一条窗口）逐根阶段 14.8s → 4.7s；全市场引擎阶段 1481.6s →
    # 640.2s → **212.9s → 139.2s**（总用时 2149.8s → 1313.0s → 892.5s → **815.3s**），
    # 两份产物与**最初基线逐位相同**。各处的账分别是（各自单独 A/B 量得；相加会略多于总降幅，
    # 因为不同轮次的机器负载不同，故请只看**同一条 A/B 内**的差与「逐位相同」）：
    #
    # - ``stdstats=False``：**不挂绘图用的观察者**（Broker / BuySell / DataTrades）。本引擎的
    #   产物全部来自 analyzers（净值曲线、成交明细、拒单），观察者是给 ``cerebro.plot()``
    #   用的，而 ``_drive_engine`` 不把 cerebro 交出去，故它们在这里纯粹是每标的每根跑一遍
    #   的白开销——实测省 3.5s / 14.8s，是本组里最大的一笔，且这 3.5s **全在逐根**：把装配
    #   （入口 → 第一次 ``preload``）单独掐出来量，开关两边是 0.17s / 0.16s——观察者**创建**
    #   本身不花时间，是它们逐根跑才贵。
    # - 持仓表 ``AStockBroker.positions``：``_get_value`` 与 ``next()`` 每 tick 各走一遍整表，
    #   见 :class:`costs._LivePositions`（省 2.4s，另把浮点求和顺序钉死）。
    # - ``quicknotify=False``：显式写出来，因为它决定了「订单通知什么时候送达」这件与主时钟
    #   有关的事——False（默认，也是这里必须的）时通知在 ``_oncepost`` 里送达，而那一刻策略
    #   的 ``datetime`` 线**已经**被写成当根的主时钟，于是 :class:`_EngineClock` 与
    #   ``examples.strategies.EngineClock`` 可以直接读它；改成 True 会在 ``_brokernotify``
    #   里立刻送达（早于主时钟更新），拒单记录上的日期会整体提早一根。
    # - 撮合侧的主时钟 ``AStockBroker._compute_clock``：见那里的说明（省 0.3s）。
    # - feed 的 tick 记账：见 :class:`_BarFeed`（省 4.6s；把 feed 换回原版
    #   ``PandasDirectData`` 单独 A/B 量得。装载阶段两边一样——``load()`` 不碰 ``tick_*``，
    #   那是 ``advance()`` / ``next()`` 的事）。
    # - feed 的**装载**（整列拷贝）：见 :class:`_BarFeed` 的 :meth:`~_BarFeed.preload`。
    #   全市场实测那 81s 的「首行进度迟迟不来」变成 **2s**，引擎阶段随之少 73.7s
    #   （212.9s → 139.2s）。这一处是**装载**的账，不在上面那组「每 tick × 每标的」里
    #   ——它按**总根数**（980 万根 × 6 列）放大，与标的数无关。
    cerebro = bt.Cerebro(stdstats=False, quicknotify=False)
    for market in markets:
        data = _feed(market.prices)
        data._mbt_symbol = market.symbol
        data._name = market.symbol
        cerebro.adddata(data)

    cerebro.addstrategy(_always_next(strategy), **strategy_params)
    if sizer is None:
        # `sizer_options` 在这里也要生效：默认 sizer 的 `headroom`（留余地比例）是 AC 要求
        # **可配置**的那一项，只在自定义 sizer 那条路径生效会让它不可达。
        cerebro.addsizer(
            EqualWeightSizer, max_positions=max_positions, rules=table, **(sizer_options or {})
        )
    else:
        cerebro.addsizer(sizer, **(sizer_options or {}))

    broker = AStockBroker(
        rules=table,
        tradability=tradability,
        universe=universe_mask,
        selection=selection_mask,
        signals=signals,
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
    if progress is not None:
        # 分母是**引擎的 tick 数**，不是任何单个标的的根数：净值曲线每 tick 一行，而 tick
        # 走的是各标的交易日的并集，`build_tradability` 正是按并集建的表。用错分母会让
        # 进度条永远到不了 100%（全市场里次新股让并集比任何单只标的都长）。
        #
        # 「先装载」那句是**必须写的**：backtrader 在第一次 tick 之前要把全部标的的全部
        # K 线读进自己那套 line buffer，而这一段**没有任何逐根进度可言**（策略还没被调用）。
        # 实测它占引擎阶段的一大半（45 个标的时是 5s／8.7s），全市场就是**分钟量级**，
        # 且它与 `_dopreload` 绑在一起——关掉它会顺带把引擎从 runonce 切到 runnext，
        # 那是另一条执行路径，不是「日志」该动的东西。故这里只能把窗口**说出来**，
        # 好让「首行进度迟迟不来」不被读成「卡住」。首行进度行自带的 `已用 Ns` 会当场
        # 报出这一段到底有多长。
        progress.stage(
            "回测引擎",
            note=(
                f"{len(markets)} 个标的 × {len(tradability)} 根 K 线；"
                "先装载全部行情，首行进度会晚到"
            ),
        )
        cerebro.addanalyzer(
            _ProgressRecorder, _name="progress", reporter=progress, total=len(tradability)
        )

    runs = cerebro.run()
    run = runs[0]

    if progress is not None:
        progress.finish()

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
        # 单标的入口没有额外信号：它的调用方是库的直接使用者，要信号可以走组合入口。
        signals=None,
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
