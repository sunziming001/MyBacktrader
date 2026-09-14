"""组合回测：共享资金池、股票池、最大持仓数、以及**组合场景独有的那类错误**（票据 #6）。

本文件的核心是 `test_an_order_must_not_fill_on_a_stale_bar_when_the_symbol_is_suspended`。
单标的路径不需要这个约束——停牌日**根本没有那一行**，引擎见不到它。但多标的场景下
backtrader 在停牌日返回的是**上一根的陈旧 K 线**：``volume`` 看着完全正常，于是订单会按
陈旧价成交。实测（8 根 K 线、B 缺 2 天）确认过：订单在 B 停牌当天按 20.0 成交了。那是
「凭空造出一笔没有成交的成交」，正是 ADR-0002 要防的虚假信心。
"""

from __future__ import annotations

import math

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

    assert result.trades["symbol"].tolist() == ["sh600000"], "只有主板那只该成交"
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

    first, second = fills.iloc[0], fills.iloc[1]
    assert first["symbol"] == "sh600000"
    assert first["date"] < c_first, "甲在**丙上市之前**就该成交"
    assert second["symbol"] == "sz000001"
    assert second["date"] >= c_first


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


# --- 喂给引擎的行情表：列序是契约，feed 的类型是性能决定 ---------------------------------


def test_the_engine_feed_reads_columns_by_position_so_the_order_is_checked():
    """行情表列序不对时**当场报错**，而不是把开高低收错位喂进引擎。

    ``PandasDirectData`` 按 ``itertuples`` 的**位置**取数（列 1=open … 列 5=volume），
    故列序是契约。这类错不会抛异常、也不会在净值曲线上显形，只会让每笔成交价都差一点——
    正是最该用一道检查换掉的那种静默失真。
    """
    from mbt.backtest.engine import _feed

    good = flat_bars(4, 10.0)
    assert _feed(good) is not None  # 正常列序不报错

    shuffled = good[["open", "high", "low", "close", "volume"]].reindex(
        columns=["close", "high", "low", "open", "volume"]
    )
    with pytest.raises(ValueError, match="列序"):
        _feed(shuffled)


def test_the_engine_feed_is_the_direct_one_because_preload_dominates():
    """引擎用 ``PandasDirectData``——这条钉住的是**实测得到的一个数量级**，不是偏好。

    两者产出逐位相同（已在真实行情上比对过），但装载一段实测差 6.6 倍（26.5s → 4.0s／232 只），
    而全市场下装载是引擎阶段的一大半（545s / 1481.6s）。故这里断言类型本身：将来若有人
    「顺手统一成 PandasData」，这条测试会红，并把他指向这一段注释。
    """
    from mbt.backtest.engine import _feed

    feed = _feed(flat_bars(4, 10.0))
    assert isinstance(feed, bt.feeds.PandasDirectData)
    # `openinterest=-1` 是「本表没有这一列」。给了它，第 6 列 `amount` 会被当成持仓量。
    assert feed.p.openinterest == -1


# --- 装载：整列拷贝必须与逐根装载**逐位**相同 ---------------------------------
#
# `_BarFeed.preload` 把每列一次性拷进 line 缓冲（日期整列向量化），替掉了 `PandasDirectData._load`
# 的逐根循环——后者是全市场 980 万根 × 6 列的 Python 循环，实测占引擎阶段四成（81s / 212.9s）。
#
# 这一组的写法与别处不同：**直接比 line 缓冲的原始数组**，而不是比回测结果。理由是等价性的
# 论据在「同一批 double」上（见 `_BarFeed.preload` 的三条），比对结果只会在必要时才暴露它；
# 而这里能给出**逐元素**的比对，且任何一处差（哪怕只差一根、只差末位）都会红。


def _preload_both_ways(prices):
    """同一张行情表分别走**整列**与**逐根**两条装载路径，返回两条路径的 line 缓冲。

    逐根那条用一个显式退回基类的子类来拿——不靠改参数去「碰巧」绕开整列路径。
    """
    from mbt.backtest.engine import _BarFeed

    class _ByBar(_BarFeed):
        def preload(self):
            # `super(_BarFeed, self)` 是刻意的：跳过 `_BarFeed.preload` 这个覆盖，直落基类
            # `PandasDirectData` 的逐根原路。tick 那两处 no-op 仍在这里生效，但 `preload`
            # 不碰 `tick_*`（那是 `advance()` / `next()` 的事），故两边可比。
            return super(_BarFeed, self).preload()

    def load(feed):
        # 走 cerebro 的装配顺序：`_env` 由 `adddata` 给（`_start()` 要用它取交易日历），
        # 然后是 `reset` → `extend(size=lookahead)` → `_start` → `preload`
        # （见 `Cerebro.runstrategies`）。
        bt.Cerebro(stdstats=False).adddata(feed)
        feed.reset()
        feed.extend(size=1)  # 与 cerebro 一致：`lookahead` 的那一格
        feed._start()
        feed.preload()
        return feed

    columns = load(_BarFeed(dataname=prices, openinterest=-1))
    bars = load(_ByBar(dataname=prices, openinterest=-1))
    return columns, bars


def _line_snapshot(feed):
    """把 feed 的全部 line 缓冲照下来：原始数组 + 指针 + 那格 lookahead。"""
    return {
        alias: (
            # `array.array` 的相等是**逐元素**的精确比较，NaN 也按位（nan != nan，故下面用
            # 自己的比法：先比长度与 `idx`，再逐位比每格的值）。
            tuple(line.array),
            line.idx,
            line.lencount,
            line.extension,
            line.buflen(),
        )
        for alias, line in ((alias, getattr(feed.lines, alias)) for alias in _LINE_ALIASES)
    }


_LINE_ALIASES = ("datetime", "open", "high", "low", "close", "volume", "openinterest")


def _assert_same_lines(columns, bars):
    """两条装载路径的每一格都要**逐位**相同（含 NaN 的格子与那一格 lookahead）。"""
    left = _line_snapshot(columns)
    right = _line_snapshot(bars)
    assert set(left) == set(right)
    for alias in left:
        larr, lidx, llencount, lext, lbuflen = left[alias]
        rarr, ridx, rlencount, rext, rbuflen = right[alias]
        assert (lidx, llencount, lext, lbuflen) == (ridx, rlencount, rext, rbuflen), alias
        assert len(larr) == len(rarr), alias
        for i, (a, b) in enumerate(zip(larr, rarr, strict=True)):
            if math.isnan(a) or math.isnan(b):
                assert math.isnan(a) and math.isnan(b), f"{alias}[{i}]"
            else:
                assert a == b, f"{alias}[{i}]：{a!r} != {b!r}"


def test_the_engine_feed_loads_whole_columns_and_gets_the_same_lines_as_bar_by_bar():
    """整列装载与逐根装载产出**逐位相同**的 line 缓冲。

    等价性的三条论据见 ``_BarFeed.preload``：同一批 double、同一套日期（``date2num(午夜)``
    = ``float(toordinal)``）、以及末尾那 ``extension`` 格 lookahead。

    这里刻意放了两个坑：**周末之后的日子**（真实交易日历的间隔，能抓住「按索引推日期」的写法）
    与一根 **NaN 的 K 线**（停牌/缺失在面板里长这样，能抓住「NaN 被过滤掉」的写法）。
    """
    prices = flat_bars(6, 10.0, start="2024-01-05")  # 2024-01-05 是周五，跨周末
    prices.loc[prices.index[2], ["open", "high", "low", "close"]] = float("nan")

    columns, bars = _preload_both_ways(prices)

    # `_ByBar` 那一侧没有 mode 可言——它绕开的正是记这个属性的地方。「退回逐根」由
    # `test_the_engine_feed_falls_back_when_a_time_of_day_is_present` 拿真 feed 钉住。
    assert columns._mbt_preload_mode == "columns", "这张表本该走整列路径"
    _assert_same_lines(columns, bars)


def test_the_engine_feed_dates_are_the_same_numbers_bt_date2num_produces():
    """日期线逐格等于 ``bt.date2num(那一天的午夜)`` —— 向量化的那条公式在这里对账。

    ``date2num`` 对午夜时刻取 ``math.fsum((ordinal, 0.0, 0.0, 0.0, 0.0))``，只有一项非零，
    故结果**恰好**是 ``float(ordinal)``；而 ``ordinal = 719163 + floor(ns / 86400e9)``。
    两处都是整数进浮点，本世纪内远小于 2^53，故没有舍入。
    """
    prices = flat_bars(5, 10.0, start="2024-02-28")  # 跨 2/29（闰日）
    columns, _ = _preload_both_ways(prices)

    got = list(columns.lines.datetime.array[: len(prices)])  # 末尾那格是 lookahead 的 NAN
    expected = [bt.date2num(stamp.to_pydatetime()) for stamp in prices.index]
    assert got == expected
    assert math.isnan(columns.lines.datetime.array[len(prices)])


def test_the_engine_feed_falls_back_when_a_time_of_day_is_present():
    """带时刻的时间戳退回逐根装载——那条公式只对午夜成立。

    向量化那条路算的是 ``719163 + floor(ns / 86400e9)``，**压根不看时刻**；而 ``date2num``
    对非午夜时刻取 ``math.fsum((ordinal, h/24, m/1440, s/86400, us/86400e6))``。两者差的不
    只是末位——是整天。这里同时钉住「退回了」与「退回去之后仍然逐位相同」。
    """
    prices = flat_bars(5, 10.0)
    prices.index = prices.index + pd.Timedelta(hours=15)  # 收盘时刻，非午夜

    columns, bars = _preload_both_ways(prices)

    assert columns._mbt_preload_mode == "bars", "带时刻的表不该走整列路径"
    _assert_same_lines(columns, bars)
    assert list(columns.lines.datetime.array[: len(prices)]) == [
        bt.date2num(stamp.to_pydatetime()) for stamp in prices.index
    ]


# --- 逐根的固定开销：四处「与标的数成正比、与策略无关」的代价 -------------------
#
# 这一段钉的都是**每 tick × 每标的**的固定开销，它们不改变任何数值结果，却是全市场
# （4752 标的 × 2701 根 = 1280 万次）下的分钟量级。四处的账与实测（232 标的、逐根阶段
# 14.8s → 6.2s）见 `mbt.backtest.engine._drive_engine` 开头那段注释。
#
# 之所以每条都要有测试盯着，是因为这些改动全部**不报错的**：改坏了要么只是变慢（没人会
# 注意到），要么算出的日期偏一天（多标的 + 掩码的组合下不会崩，只会让决策落在另一天）。


def test_all_three_clocks_agree_with_the_naive_scan_on_every_tick(make_market, zero_cost_rules):
    """三处主时钟（策略侧 / 分析器侧 / 撮合侧）**逐根**与「扫一遍标的取最大值」对账。

    三处现在都不再扫标的了，各自的等价论证见 ``_EngineClock`` / ``EngineClock`` /
    ``AStockBroker._compute_clock``。这条测试把三者放在同一根 K 线上比对，且刻意放进一只
    **晚上市**的标的——那正是这个时钟存在的理由（第一个标的停在末根时，扫出来的最大值与它
    的日期会分叉）。

    - 策略侧：``EngineClock.today()``（读策略自己的 ``datetime`` 线）。
    - 撮合侧：``AStockBroker._compute_clock()``（比浮点日期序号、只转换赢家）。
    - 分析器侧：它写出的就是净值曲线的索引，故用整条索引逐根比——比只看首尾严得多，
      中间偏一天也会被抓到。
    """
    from examples.strategies import EngineClock

    naive_seen = []

    class Probe(EngineClock, bt.Strategy):
        def next(self):
            naive = max(d.datetime.date(0) for d in self.datas if len(d))
            naive_seen.append(naive)
            assert self.today() == pd.Timestamp(naive), "策略侧时钟与朴素扫描不一致"
            assert self.broker._compute_clock() == naive, "撮合侧时钟与朴素扫描不一致"

    early = make_market("sh600000", flat_bars(6, 10.0))
    late = make_market("sz000001", flat_bars(3, 20.0, start="2024-01-04"))

    result = run_portfolio_backtest(
        [early, late],
        Probe,
        cash=100_000.0,
        max_positions=2,
        rules=zero_cost_rules,
        universe_rules=UniverseRules(min_bars=0),
    )

    assert len(naive_seen) == 6, "时钟要走完两个标的日期的并集"
    assert [stamp.date() for stamp in result.equity_curve.index] == naive_seen


def test_the_engine_adds_no_plotting_observers(make_market, zero_cost_rules):
    """引擎不挂**绘图用**的观察者——它的产物全部来自 analyzers。

    ``bt.observers.Broker`` / ``BuySell`` / ``DataTrades`` 只在 ``cerebro.plot()`` 里有用，
    而 ``_drive_engine`` 不把 cerebro 交出去（返回的是 :class:`BacktestResult`），故它们在这里
    纯粹是每标的每根跑一遍的白开销：实测（232 标的）3.5s / 14.8s，是这一组里最大的一笔。

    将来若有人改回默认，这条会红——要加绘图观察者，请先想清楚这 24% 花得值不值。
    """
    counts = []

    class Probe(bt.Strategy):
        def next(self):
            counts.append(len(self.observers))

    run_portfolio_backtest(
        [make_market("sh600000", flat_bars(6, 10.0))],
        Probe,
        cash=100_000.0,
        rules=zero_cost_rules,
        universe_rules=UniverseRules(min_bars=0),
    )

    assert counts, "策略应当被调用过"
    assert set(counts) == {0}, f"不该有观察者，实际每根 {sorted(set(counts))} 个"


def test_the_engine_feed_leaves_the_tick_attributes_alone(make_market, zero_cost_rules):
    """bar 模式的 feed 不做 tick 记账：``tick_*`` 全程是 ``None``。

    唯一读 ``tick_*`` 的是 ``BackBroker._try_exec``，它读不到就退回 ``data.open[0]`` /
    ``high[0]`` / ``low[0]`` / ``close[0]``——**正是那段记账会抄进去的值**（记账发生在同一
    tick 的推进里，撮合紧随其后，中间没有第二次推进）。故「成交价仍然对」这件事由
    ``test_golden_case_of_a_three_symbol_portfolio`` 之类的用例盯着，这里只钉「记账确实省了」。
    """
    tick_values = []

    class Probe(bt.Strategy):
        def next(self):
            # `tick_*` 是 `_tick_nullify()` 建的（不是 `start()`），省掉记账后这些属性**根本
            # 不存在**——`BackBroker._try_exec` 用的就是 `getattr(..., None)`，故这里同一种读法。
            for alias in ("tick_open", "tick_high", "tick_low", "tick_close"):
                tick_values.append(getattr(self.data0, alias, None))

    run_portfolio_backtest(
        [make_market("sh600000", flat_bars(4, 10.0))],
        Probe,
        cash=100_000.0,
        rules=zero_cost_rules,
        universe_rules=UniverseRules(min_bars=0),
    )

    assert tick_values
    assert set(tick_values) == {None}, "tick_* 一次也不该被填"


def test_a_rejection_is_dated_on_the_day_it_happened(make_market, zero_cost_rules):
    """拒单记录上的日期是**当根**，不是上一根。

    ``_RejectionRecorder`` 在 ``notify_order`` 里取主时钟，而它现在读的是策略的 ``datetime``
    线——那就依赖 ``quicknotify=False``：订单通知必须在 ``_oncepost`` 里送达（那时主时钟**已经**
    写好），而不是在 ``_brokernotify`` 里立刻送达（那时还停在上一根）。这条测试就是那个依赖的
    哨兵：一旦有人把 ``quicknotify`` 打开，日期会整体早一根，这条会红。

    构造同 ``test_an_order_expires_when_the_symbol_stays_stale_too_long``：停牌 3 天、容忍 2 天，
    故第 3 个无 K 线的交易日作废——那一天是 ``2024-01-08``。
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

    expired = result.rejected[result.rejected["symbol"] == "sz000001"]
    assert len(expired) == 1
    assert expired.iloc[0]["date"] == pd.Timestamp("2024-01-08").date()


def test_the_position_table_is_walked_in_symbol_order_and_skips_empty_slots(
    make_market, zero_cost_rules
):
    """持仓表的遍历顺序 = **标的顺序**，且零持仓的标的**不出现在遍历里**。

    两件事，理由各不同（见 ``mbt.backtest.costs._LivePositions`` 的类文档）：

    - **顺序固定** ⇒ ``BackBroker._get_value`` 的浮点求和顺序固定。它按**插入顺序**求和，
      而浮点加法不满足结合律，于是「哪些标的先被碰过」会改变净值曲线的末位（实测：同一份
      行情、同一批成交，净值曲线 868 行差在末位，如 ``101524.17725994284`` / ``...285``）。
      预置后插入顺序恒为标的顺序，与「谁先被碰过」脱钩。
    - **跳过零持仓** ⇒ 省掉每 tick × 每标的的 ``getcommissioninfo`` + ``data.close[0]``。
      零持仓在 ``next()`` 的两处循环（都被 ``if pos:`` 挡着）与 ``_get_value`` 里都是**精确
      的 0**，故跳过逐位不变——只是不再为全市场 4752 标的 × 2701 根各付一遍。

    顺序用「**先碰第 3 只、再碰第 1 只**」来验：若遍历仍按插入顺序，得到的是 ``[d2, d0]``；
    标的顺序则是 ``[d0, d2]``。第 2 只从头到尾没人碰，它不出现，但仍应**在表里**（预置过了）。
    """
    seen = {}

    class Toucher(bt.Strategy):
        def next(self):
            if len(self) == 1:
                self.buy(data=self.datas[2])  # 先碰第 3 只
            elif len(self) == 2:
                self.buy(data=self.datas[0])  # 再碰第 1 只
            # 第 3 根时两笔都已成交，此时遍历顺序才看得出差别。
            seen[len(self)] = [data._name for data in self.broker.positions]

    result = run_portfolio_backtest(
        [
            make_market("sh600000", flat_bars(5, 10.0)),
            make_market("sh600001", flat_bars(5, 30.0)),
            make_market("sh600002", flat_bars(5, 20.0)),
        ],
        Toucher,
        cash=100_000.0,
        max_positions=3,
        rules=zero_cost_rules,
        universe_rules=UniverseRules(min_bars=0),
        sizer=bt.sizers.FixedSize,
        sizer_options={"stake": 100},
    )

    # 第 1 根：还没成交，遍历为空——零持仓被跳过。
    assert seen[1] == []
    # 第 2 根：只有第 3 只成交（它先被碰过）。
    assert seen[2] == ["sh600002"]
    # 第 3 根：两只都在，顺序必须是**标的顺序**（第 1 只在前），而不是插入顺序（第 3 只在先）。
    assert seen[3] == ["sh600000", "sh600002"]

    assert len(result.equity_curve) == 5, "顺序变了不该影响曲线长度，这里只是顺带钉住"


def test_the_position_table_holds_every_symbol_even_the_untouched_ones(
    make_market, zero_cost_rules
):
    """零持仓标的**跳过遍历**，但**必须留在表里**——否则读它的持仓会造出一个新键。

    这是上一条的另一半：跳过不能顺手改成 ``del``。留着的代价是一次字典插入（全市场 4752 次，
    毫秒量级），换来的是插入顺序稳定；删掉则会让下一个碰它的标的插到表尾，顺序重新变成
    「谁先被碰过」，那正是上一条要掐掉的东西。
    """
    sizes = []

    class Toucher(bt.Strategy):
        def next(self):
            sizes.append([self.broker.positions[data].size for data in self.datas])

    run_portfolio_backtest(
        [
            make_market("sh600000", flat_bars(4, 10.0)),
            make_market("sh600001", flat_bars(4, 30.0)),
        ],
        Toucher,
        cash=100_000.0,
        rules=zero_cost_rules,
        universe_rules=UniverseRules(min_bars=0),
    )

    assert sizes, "策略应当被调用过"
    assert set(tuple(row) for row in sizes) == {(0, 0)}, "没人下过单，两只都该在表里且为零持仓"
