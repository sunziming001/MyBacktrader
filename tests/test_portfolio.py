"""组合回测：共享资金池、股票池、最大持仓数、以及**组合场景独有的那类错误**（票据 #6）。

本文件的核心是 `test_an_order_must_not_fill_on_a_stale_bar_when_the_symbol_is_suspended`。
单标的路径不需要这个约束——停牌日**根本没有那一行**，引擎见不到它。但多标的场景下
backtrader 在停牌日返回的是**上一根的陈旧 K 线**：``volume`` 看着完全正常，于是订单会按
陈旧价成交。实测（8 根 K 线、B 缺 2 天）确认过：订单在 B 停牌当天按 20.0 成交了。那是
「凭空造出一笔没有成交的成交」，正是 ADR-0002 要防的虚假信心。
"""

from __future__ import annotations

import backtrader as bt
import pandas as pd
import pytest

from mbt.backtest import run_portfolio_backtest
from mbt.data import MarketData
from mbt.universe import MAIN_BOARD, UniverseRules


def flat_bars(n, value, start="2024-01-02", volume=1000):
    """n 根等值 K 线，全为整数值以便手算。"""
    return pd.DataFrame(
        {
            "open": [value] * n,
            "high": [value] * n,
            "low": [value] * n,
            "close": [value] * n,
            "volume": [volume] * n,
        },
        index=pd.bdate_range(start, periods=n),
    )


class BuyEverything(bt.Strategy):
    """第一根 K 线后，对每个标的各下一张买单。"""

    def next(self):
        if len(self) == 1:
            for data in self.datas:
                self.buy(data=data)


# --- 组合场景独有的缺口：陈旧 K 线不可成交 -------------------------------------


def test_an_order_must_not_fill_on_a_stale_bar_when_the_symbol_is_suspended(
    make_market, zero_cost_rules
):
    """停牌日不得成交——否则会按**陈旧价**造出一笔没有发生的成交。

    标的 A 有 6 个交易日；B 缺第 3、4 个（停牌）。策略在第 2 根 K 线后买 B：B 的下一个
    **真实**交易日是第 5 根，故成交日必须是那里，而不是停牌的第 3 根。

    这条在修掉缺口之前会失败：陈旧的 B 那根 ``volume`` 仍是 1000，于是订单在停牌当天
    按 B 的上一根收盘价成交。
    """
    a = make_market("sh600000", flat_bars(6, 10.0))
    b_full = flat_bars(6, 20.0)
    b = make_market("sz000001", b_full.drop([b_full.index[2], b_full.index[3]]))

    class BuyB(bt.Strategy):
        def next(self):
            if len(self) == 2 and not self.position:
                self.buy(data=self.datas[1])

    result = run_portfolio_backtest(
        [a, b],
        BuyB,
        cash=100_000.0,
        max_positions=1,
        rules=zero_cost_rules,
        universe_rules=UniverseRules(min_bars=0),
        sizer=bt.sizers.FixedSize,
        sizer_options={"stake": 100},
    )

    assert len(result.trades) == 1
    assert result.trades.iloc[0]["date"] == pd.Timestamp(
        "2024-01-08"
    ), "成交落在了停牌日——那是按陈旧价成交，凭空造出了一笔没有发生的成交"


def test_a_suspended_symbol_has_no_tradable_bar_but_its_neighbours_do(make_market, zero_cost_rules):
    """停牌只影响它自己：同一 tick 里其它标的照常成交。"""
    a = make_market("sh600000", flat_bars(6, 10.0))
    b_full = flat_bars(6, 20.0)
    b = make_market("sz000001", b_full.drop([b_full.index[2], b_full.index[3]]))

    result = run_portfolio_backtest(
        [a, b],
        BuyEverything,
        cash=100_000.0,
        max_positions=2,
        rules=zero_cost_rules,
        universe_rules=UniverseRules(min_bars=0),
    )

    dates = set(result.trades["date"])
    assert pd.Timestamp("2024-01-03") in dates, "A 在 B 停牌前就该成交"
    assert pd.Timestamp("2024-01-04") not in dates, "B 停牌当天不该有任何 B 的成交"


def test_the_strategy_can_see_the_tradability_mask(make_market, zero_cost_rules):
    """掩码挂在 broker 上，策略据此分辨「今天这根是新的还是陈旧的」。

    停牌日 ``data.close[0]`` 返回**陈旧价**，策略若不查掩码就会基于几个月前的价格做决策。
    这条同时验证掩码里 B 的停牌日为假、其邻居 A 为真。
    """
    a = make_market("sh600000", flat_bars(4, 10.0))
    b_full = flat_bars(4, 20.0)
    b = make_market("sz000001", b_full.drop([b_full.index[1]]))
    seen = {}

    class Peek(bt.Strategy):
        def next(self):
            mask = self.broker.tradability_mask
            today = self.data0.datetime.date(0)
            seen[today] = {
                "sh600000": bool(mask.at[pd.Timestamp(today), "sh600000"]),
                "sz000001": bool(mask.at[pd.Timestamp(today), "sz000001"]),
            }

    run_portfolio_backtest(
        [a, b],
        Peek,
        rules=zero_cost_rules,
        universe_rules=UniverseRules(min_bars=0),
    )

    suspended = pd.Timestamp("2024-01-03").date()
    assert seen[suspended]["sz000001"] is False
    assert seen[suspended]["sh600000"] is True


def test_an_order_expires_when_the_symbol_stays_stale_too_long(make_market, zero_cost_rules):
    """停牌超过上限即拒单——否则一笔单会在复牌日突然成交，那已不是原来那笔交易。

    用上限 2 个交易日、停牌 3 天：第 2 个无 K 线的交易日即作废。
    """
    a = make_market("sh600000", flat_bars(6, 10.0))
    b_full = flat_bars(6, 20.0)
    b = make_market("sz000001", b_full.drop([b_full.index[2], b_full.index[3], b_full.index[4]]))

    class BuyB(bt.Strategy):
        def next(self):
            if len(self) == 2 and not self.position:
                self.buy(data=self.datas[1])

    result = run_portfolio_backtest(
        [a, b],
        BuyB,
        cash=100_000.0,
        max_positions=1,
        rules=zero_cost_rules,
        universe_rules=UniverseRules(min_bars=0),
        order_expiry_ticks=2,
        sizer=bt.sizers.FixedSize,
        sizer_options={"stake": 100},
    )

    assert len(result.trades) == 0, "停牌过久的挂单应当失效，而不是等到复牌再成交"
    expired = result.rejected[result.rejected["symbol"] == "sz000001"]
    assert len(expired) == 1, "挂单失效必须留痕，否则事后查不到「这笔单为什么没成交」"
    assert expired.iloc[0]["status"] == "Rejected"


# --- 股票池：禁买不禁卖 -------------------------------------------------------


def test_a_buy_outside_the_universe_is_rejected(make_market, zero_cost_rules):
    """池外买入被撮合层拒绝，且**留痕**。

    用一只**创业板**股票与「只收主板」的准入规则造出池外状态：它的板块可判定，故费用
    与限幅都能算；被排除纯粹是因为准入规则。

    （刻意不用指数来造这个状态：非股票标的没有板块，规则层的 ``board_of`` 会**报错**而不是
    兜底——那是对「把股票制度硬套到指数上」的正确反应，但它会让这条测试测到别的东西。）
    """
    main_board = make_market("sh600000", flat_bars(4, 10.0))
    chinext = make_market("sz300750", flat_bars(4, 20.0))

    result = run_portfolio_backtest(
        [main_board, chinext],
        BuyEverything,
        cash=100_000.0,
        max_positions=2,
        rules=zero_cost_rules,
        universe_rules=UniverseRules(boards=frozenset({MAIN_BOARD}), min_bars=0),
        sizer=bt.sizers.FixedSize,
        sizer_options={"stake": 100},
    )

    assert result.trades["price"].tolist() == [10.0], "只有主板那只该成交"
    blocked = result.rejected[result.rejected["symbol"] == "sz300750"]
    assert len(blocked) == 1, "池外买入被拒必须留痕"


def test_a_sell_outside_the_universe_is_still_allowed(make_market, synthetic_rules):
    """池外只禁买、不禁卖——持仓可能因调仓或数据变化掉出池子，禁卖会把持仓卡死。

    做法是造一个「先在内、后在外」的标的：合成规则表登记 ``sh600243`` 自 **2025-04-01**
    起为 ST。于是 3 月底买入时它在池内，4 月起被排除——此时卖出仍须成交。

    这条同时验证了掩码确实在 4 月起翻了面（否则「卖出成功」可能只是因为它根本没出池）。
    """
    bars = flat_bars(8, 3.0, start="2025-03-25")
    market = make_market("sh600243", bars)
    seen = {}

    class BuyThenSellAfterSt(bt.Strategy):
        def next(self):
            today = self.data0.datetime.date(0)
            mask = self.broker.universe_mask
            seen[today] = bool(mask.at[pd.Timestamp(today), "sh600243"])
            if len(self) == 1:
                self.buy(data=self.data0, size=100)
            elif today >= pd.Timestamp("2025-04-01").date() and self.position:
                self.sell(data=self.data0, size=100)

    result = run_portfolio_backtest(
        [market],
        BuyThenSellAfterSt,
        cash=100_000.0,
        max_positions=1,
        rules=synthetic_rules,
        universe_rules=UniverseRules(min_bars=0),
        sizer=bt.sizers.FixedSize,
        sizer_options={"stake": 100},
    )

    assert seen[pd.Timestamp("2025-03-31").date()] is True
    assert seen[pd.Timestamp("2025-04-01").date()] is False, "标的并未出池，测试前提不成立"

    sells = result.trades[result.trades["size"] < 0]
    assert len(sells) == 1, "池外卖出被拦了——持仓会被卡死"


# --- 最大持仓数与持仓分配 -----------------------------------------------------


def test_max_positions_caps_the_number_of_held_symbols(make_market, zero_cost_rules):
    """持满之后对**新标的**的买入被拒。"""
    markets = [
        make_market("sh600000", flat_bars(4, 10.0)),
        make_market("sz000001", flat_bars(4, 20.0)),
        make_market("sz000002", flat_bars(4, 30.0)),
    ]

    result = run_portfolio_backtest(
        markets,
        BuyEverything,
        cash=100_000.0,
        max_positions=1,
        rules=zero_cost_rules,
        universe_rules=UniverseRules(min_bars=0),
        sizer=bt.sizers.FixedSize,
        sizer_options={"stake": 100},
    )

    assert len(result.trades) == 1, "最大持仓数为 1 时不该有第二笔建仓"


def test_adding_to_an_existing_position_is_not_counted_as_a_new_one(make_market, zero_cost_rules):
    """加仓不增加持仓的**标的数**，故不受最大持仓数限制。"""
    market = make_market("sh600000", flat_bars(6, 10.0))

    class BuyTwice(bt.Strategy):
        def next(self):
            if len(self) == 1:
                self.buy(data=self.data0, size=100)
            elif len(self) == 3:
                self.buy(data=self.data0, size=100)

    result = run_portfolio_backtest(
        [market],
        BuyTwice,
        cash=100_000.0,
        max_positions=1,
        rules=zero_cost_rules,
        universe_rules=UniverseRules(min_bars=0),
        sizer=bt.sizers.FixedSize,
        sizer_options={"stake": 100},
    )

    assert len(result.trades) == 2


# --- 黄金用例：能手算的极小组合 ------------------------------------------------


def test_the_backtest_uses_the_backward_adjusted_view_internally(
    make_prices, make_market, zero_cost_rules
):
    """组合入口**自动套用后复权**，调用方不必记得这件事（AC：复权视图在组合场景下同样生效）。

    构造一次 10 送 10：除权前 20.0、除权后 10.0 的原始价会留下一个 −50% 的假跳空。后复权
    视图把除权后的价格×2，故序列在除权日连续。测试断言「撮合用的价格是连续的」——
    若引擎忘了复权，成交价会落在 10.0 那一侧，与后复权序列的 20.0 不符。

    复权是引擎的职责，不是调用方的：组合场景下漏掉它的代价是**每个除权日一个假跳空**，
    比单标的更容易被忽略。
    """
    from mbt.data import AdjustmentEvent

    frame = flat_bars(4, 20.0)
    frame.iloc[2:, :] = 10.0  # 第 3 根起因除权而腰斩
    events = (AdjustmentEvent(symbol="sh600000", ex_date=frame.index[2].date(), bonus_per_10=10.0),)
    market = MarketData(symbol="sh600000", prices=frame, events=events)
    other = make_market("sz000001", flat_bars(4, 50.0))

    seen = {}

    class PeekAndBuy(bt.Strategy):
        def next(self):
            if len(self) == 1:
                seen["prices"] = [self.data0.open[i] for i in range(4)]
                self.buy(data=self.data0, size=100)

    result = run_portfolio_backtest(
        [market, other],
        PeekAndBuy,
        cash=100_000.0,
        max_positions=2,
        rules=zero_cost_rules,
        universe_rules=UniverseRules(min_bars=0),
        sizer=bt.sizers.FixedSize,
        sizer_options={"stake": 100},
    )

    assert seen["prices"] == pytest.approx(
        [20.0, 20.0, 20.0, 20.0]
    ), "引擎没有套用后复权——除权日留下了假跳空"
    assert len(result.trades) == 1


def test_golden_case_of_a_three_symbol_portfolio(make_market, zero_cost_rules):
    """黄金用例：3 个标的 × 6 个交易日，含一处停牌，期末资金**精确等于手算值**。

    构造（零费用规则表，故不含任何费用项，便于手算）：

    - ``sh600000`` 价格 10.0 → 10.5（第 3 根起），全程有 K 线
    - ``sz000001`` 价格恒 20.0
    - ``sz000002`` 价格恒 30.0，**缺第 2 根**（停牌）；它因此不能在第 2 个 tick 成交
    - 期初 100000，最大持仓数 3，等权分配（默认 sizer）

    手算依据是默认 sizer 的**成文口径**，本次改为：

    1. 先给每个持仓名额分配 ``可用资金 ÷ 剩余名额``；
    2. 再按**最坏成交价** ``定量价 × (1 + 当日板块限幅)`` 求该预算内买得起的最大股数——
       因为订单在本根收盘定量、在**下一根**成交，而下一根的价格受「前收盘 × (1 + 限幅)」约束。

    三个买单都在第 2 个 tick 提交，故各自看到「已持 0 只、剩 3 个名额、可用资金 100000」，
    预算 ``100000 ÷ 3 = 33333.33``。零费用表里沪深主板限幅都是 0.10，故最坏成交价是
    ``价格 × 1.1``：

    - ``sh600000``：``33333.33 ÷ 11.0 = 3030``
    - ``sz000001``：``33333.33 ÷ 22.0 = 1515``
    - ``sz000002``：``33333.33 ÷ 33.0 = 1010``（当日停牌，该单等到下一个 tick 才成交）

    手工核算的独立不变量：三笔共花 ``30300 × 3 = 90900``，故**剩余现金 9100**；
    期末持仓市值 ``3030×10.5 + 1515×20 + 1010×30 = 31815 + 30300 + 30300 = 92415``；
    合计 **101515**。

    .. note::

        与改动前相比，每笔**少买约 9%**（限幅的份额）——这是「按最坏成交价留余地」的**确定
        代价**，刻意承担它，换取「不会因下一根涨价而整笔被拒」。改动前是 3333/1666/1111，
        因为那时按当根收盘价定量。

    要说明的是：这几个数字与 sizer 的公式同源——被测对象是**「这个口径实现得对不对」**，
    而不是「这个口径本身好不好」。故另行断言两条与公式无关的不变量：总花费不超过期初
    资金、以及期末净值等于现金加持仓市值。
    """
    first = flat_bars(6, 10.0)
    first["open"] = [10.0, 10.0, 10.5, 10.5, 10.5, 10.5]
    first["high"] = [10.0, 10.0, 10.5, 10.5, 10.5, 10.5]
    first["low"] = [10.0, 10.0, 10.5, 10.5, 10.5, 10.5]
    first["close"] = [10.0, 10.0, 10.5, 10.5, 10.5, 10.5]
    third_full = flat_bars(6, 30.0)

    markets = [
        make_market("sh600000", first),
        make_market("sz000001", flat_bars(6, 20.0)),
        make_market("sz000002", third_full.drop([third_full.index[1]])),
    ]

    result = run_portfolio_backtest(
        markets,
        BuyEverything,
        cash=100_000.0,
        max_positions=3,
        rules=zero_cost_rules,
        universe_rules=UniverseRules(min_bars=0),
    )

    fills = result.trades.sort_values("date").reset_index(drop=True)
    assert len(fills) == 3
    assert fills["size"].tolist() == [3030, 1515, 1010]
    assert fills["price"].tolist() == pytest.approx([10.0, 20.0, 30.0])

    # 前两笔在第 2 个 tick，第三笔因停牌延到第 3 个 tick。
    assert fills["date"].tolist() == [
        pd.Timestamp("2024-01-03"),
        pd.Timestamp("2024-01-03"),
        pd.Timestamp("2024-01-04"),
    ]

    assert result.final_value == pytest.approx(101_515.0)

    # 与 sizer 公式无关的两条不变量：不超支，且净值 = 现金 + 持仓市值。
    spent = sum(size * price for size, price in zip(fills["size"], fills["price"], strict=True))
    assert spent == 90_900
    assert result.final_value == pytest.approx(
        100_000.0 - spent + 3030 * 10.5 + 1515 * 20.0 + 1010 * 30.0
    )


# --- 净值曲线的时间索引（实跑 CLI 时暴露出的真 bug） --------------------------


def test_the_equity_curve_index_is_the_engine_clock_not_the_first_symbol(
    make_market, zero_cost_rules
):
    """净值曲线的时间索引必须取自**引擎主时钟**，而不是 ``datas[0]``。

    这条钉的是一个**真 bug**：原先用 ``datas[0].datetime.date(0)``，而若第一个标的的历史比
    回测区间短（次新股、北交所早期、被 ``--limit`` 选中的任意一只），它的日期会一直停在
    末根，于是整条曲线的索引成了同一个日期的重复——**基准对齐、年化、以及所有按日期的下游
    计算都会因此算错，且不报错**。

    实跑 CLI 时撞到过：40 个标的（北交所排在最前）跑出 ``period.start == period.end``，
    而 ``trading_days`` 是 953。

    构造：短的那个（2 根）排在前面，长的（6 根）在后面。
    """
    short_first = make_market("sz000001", flat_bars(2, 20.0))
    long_second = make_market("sh600000", flat_bars(6, 10.0))

    result = run_portfolio_backtest(
        [short_first, long_second],
        BuyEverything,
        cash=100_000.0,
        max_positions=2,
        rules=zero_cost_rules,
        universe_rules=UniverseRules(min_bars=0),
        sizer=bt.sizers.FixedSize,
        sizer_options={"stake": 100},
    )

    index = result.equity_curve.index
    assert index.is_monotonic_increasing
    assert index.is_unique, "索引里有重复日期——说明日期取自了某个停滞的标的"
    assert len(index) == 6, "时钟应走完两个标的日期的并集"
    assert index[0].date() == pd.Timestamp("2024-01-02").date()
    assert index[-1].date() == pd.Timestamp("2024-01-09").date()


def test_the_fill_log_dates_also_come_from_the_engine_clock(make_market, zero_cost_rules):
    """成交明细的日期同理——它是撮合当日的日期，不该来自某个停滞的标的。"""
    short_first = make_market("sz000001", flat_bars(2, 20.0))
    long_second = make_market("sh600000", flat_bars(6, 10.0))

    result = run_portfolio_backtest(
        [short_first, long_second],
        BuyEverything,
        cash=100_000.0,
        max_positions=2,
        rules=zero_cost_rules,
        universe_rules=UniverseRules(min_bars=0),
        sizer=bt.sizers.FixedSize,
        sizer_options={"stake": 100},
    )

    dates = pd.to_datetime(result.trades["date"])
    assert dates.max() <= result.equity_curve.index.max()
    assert result.equity_curve.index.min() <= dates.min()


# --- 晚上市的标的不阻塞其余标的（票据 #31） -----------------------------------


class BuyAnyTradable(bt.Strategy):
    """任何**可交易**的标的一有机会就买 100 股。

    「今天」取各标的当前日期的最大者，而不是 ``self.data0``——理由见
    ``mbt.backtest.engine._EngineClock``。
    """

    def next(self):
        today = pd.Timestamp(max(d.datetime.date(0) for d in self.datas if len(d)))
        for data in self.datas:
            if self.getposition(data).size:
                continue
            if not self.broker.tradability_mask.at[today, data._name]:
                continue
            self.buy(data=data, size=100)


def test_a_late_listing_does_not_block_the_earlier_symbols(make_market, zero_cost_rules):
    """**本票的核心**：一只标的还没上市，不该让其余标的在前半段无事可做。

    构造：``sh600000`` 从第一天起就有 30 根；``sz000001`` **晚 15 个交易日**才上市。

    修掉之前：策略的 ``next()`` 要等到 ``sz000001`` 上市那一刻才被调用，于是 ``sh600000``
    在前 15 个交易日的信号**一次都不被看见**——它当时明明在正常交易。修掉之后，``sh600000``
    应当在它自己的第 2 根就成交，远早于 ``sz000001`` 上市。
    """
    a = make_market("sh600000", flat_bars(30, 10.0))
    c = make_market("sz000001", flat_bars(10, 20.0, start="2024-01-24"))
    c_first = c.prices.index[0]

    result = run_portfolio_backtest(
        [a, c],
        BuyAnyTradable,
        cash=100_000.0,
        max_positions=2,
        rules=zero_cost_rules,
        universe_rules=UniverseRules(min_trading_days=0),
    )

    fills = result.trades.sort_values("date").reset_index(drop=True)
    assert len(fills) == 2, "两只标的都应成交"

    # 用价格区分是谁：甲恒 10.0，丙从 20.0 起。
    assert fills.iloc[0]["price"] == 10.0, "第一笔是甲的"
    assert fills.iloc[0]["date"] < c_first, "甲在**丙上市之前**就该成交"
    assert fills.iloc[1]["price"] == 20.0
    assert fills.iloc[1]["date"] >= c_first


def test_the_strategy_is_called_from_the_first_bar(make_market, zero_cost_rules):
    """策略从**第一根**就被调用——``len(self)`` 首次是 1，不是「全体就绪」那一刻。

    这条钉住 ``len(self)`` 的语义：它是「引擎跑到第几根」，而**不是**「所有标的都开始交易之后
    过了几根」。二者在单标的下相同，故这个坑总在多标的时才冒出来（且不报错）。
    """
    observed = []

    class Observe(bt.Strategy):
        def next(self):
            if not observed:
                observed.append(len(self))

    a = make_market("sh600000", flat_bars(30, 10.0))
    c = make_market("sz000001", flat_bars(10, 20.0, start="2024-01-24"))

    run_portfolio_backtest(
        [a, c],
        Observe,
        cash=100_000.0,
        max_positions=2,
        rules=zero_cost_rules,
        universe_rules=UniverseRules(min_trading_days=0),
    )

    assert observed == [1], f"首次调用时 len(self) 应是 1，实际是 {observed}"


def test_a_symbol_that_has_not_listed_yet_reads_its_last_price_not_nan(
    make_market, zero_cost_rules
):
    """**放开调用时点后必须知道的危害**：未上市的标的，``close[0]`` 不是 NaN，而是它**最后一根
    的价格**——也就是**未来价**。

    乙的价格序列是 10,13,16,…,37（10 根）。它还没上市时 ``len(data)==0``，可 ``close[0]``
    读到的是 **37**（末根价）、``sma`` 读到的是末 5 根均值。

    故策略读价格前**必须**查 ``self.broker.tradability_mask``（停牌与未上市都是 ``False``）。
    这项工作本来就要做——停牌日的 ``close[0]`` 也是陈旧价——只是「忘了查」的后果从「拿到陈旧
    价」升级成了「拿到未来价」，而后者正是 ADR-0006 最防的东西。

    这条测试的作用是**把这个危害写成可核对的事实**，免得后人以为「数据没开始时指标会给 NaN」
    而把守卫删掉。
    """
    observed = {}

    class Observe(bt.Strategy):
        def next(self):
            if len(self) != 1:
                return
            for data in self.datas:
                observed[data._name] = (len(data), float(data.close[0]))

    rising = [10.0 + 3.0 * i for i in range(10)]
    prices = flat_bars(10, 10.0, start="2024-01-24")
    prices["open"] = rising
    prices["high"] = rising
    prices["low"] = rising
    prices["close"] = rising

    a = make_market("sh600000", flat_bars(30, 10.0))
    c = make_market("sz000001", prices)

    run_portfolio_backtest(
        [a, c],
        Observe,
        cash=100_000.0,
        max_positions=2,
        rules=zero_cost_rules,
        universe_rules=UniverseRules(min_trading_days=0),
    )

    assert observed["sh600000"] == (1, 10.0)
    length, close = observed["sz000001"]
    assert length == 0, "丙此时尚未上市"
    assert close == 37.0, "读到的是它的**末根价**（未来价），不是 NaN"


def test_a_user_defined_prenext_is_respected_and_next_is_not_called_twice(
    make_market, zero_cost_rules
):
    """用户自己写了 ``prenext`` 时，引擎**不再注入**——否则 ``next()`` 每根会被调两次。

    ``prenext`` 最常见的写法就是 ``def prenext(self): self.next()``。若在那之上再包一层，
    同一根 K 线里 ``next()`` 会跑两遍——那是**静默的双倍交易**，比不修更糟。

    这里同时钉住「每根 K 线恰好一次回调」，故 ``len(calls)`` 必须等于总根数。
    """
    calls = []

    class Custom(bt.Strategy):
        def prenext(self):
            calls.append("prenext")

        def next(self):
            calls.append("next")

    a = make_market("sh600000", flat_bars(30, 10.0))
    c = make_market("sz000001", flat_bars(10, 20.0, start="2024-01-24"))

    result = run_portfolio_backtest(
        [a, c],
        Custom,
        cash=100_000.0,
        max_positions=2,
        rules=zero_cost_rules,
        universe_rules=UniverseRules(min_trading_days=0),
    )

    total_bars = len(result.equity_curve)
    assert "prenext" in calls, "用户自己的 prenext 必须仍然被调用"
    assert len(calls) == total_bars, f"每根恰好一次回调，实际 {len(calls)} 次 / {total_bars} 根"
