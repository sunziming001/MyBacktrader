"""验证撮合侧的 A 股硬约束：T+1、涨跌停不可成交、无成交量不可成交。

测试缝 S2 + S3：价格来自仓库内的真实 fixture，限幅来自夹具规则表，费用为零。

回归样本 ``sh600243`` 一个文件里同时含三种形态，且跨了一次限幅变更：

===================  ==========================================
2025-03-20           +10.06% 涨停一字（3.61 = 3.28 × 1.1，此时非 ST）
2025-04-22           停牌（无记录）
2025-04-23           −4.86% 跌停一字（2.35 = 2.47 × 0.95，此时已带 ST）
===================  ==========================================

限幅的**方向性**是这里的关键：涨停一字上「买不进」但「卖得出」（卖方不用排队），
跌停一字上「卖不出」但「买得到」。若把限幅写成一个方向的开关，两个方向必错一个。
"""

from __future__ import annotations

import backtrader as bt
import pandas as pd

from mbt.backtest import run_backtest, run_portfolio_backtest
from mbt.data import TdxDataSource
from mbt.universe import UniverseRules

SYMBOL = "sh600243"

#: 涨停一字的日期与其成交价（前收 3.28 × 1.1，四舍五入到分）。
LIMIT_UP_DAY = pd.Timestamp("2025-03-20")

#: 停牌日：无记录。
HALT_DAY = pd.Timestamp("2025-04-22")

#: 跌停一字的日期与其成交价（前收 2.47 × 0.95，四舍五入到分）。
LIMIT_DOWN_DAY = pd.Timestamp("2025-04-23")


def _prices(fixture_root, symbol=SYMBOL):
    return TdxDataSource(fixture_root).daily(symbol)


def _run(fixture_root, limit_rules, strategy, symbol=SYMBOL, cash=1_000_000.0, **params):
    return run_backtest(
        _prices(fixture_root, symbol),
        symbol=symbol,
        strategy=strategy,
        cash=cash,
        rules=limit_rules,
        **params,
    )


# --- 涨跌停不可成交 ---


class BuyOnBar(bt.Strategy):
    """在第 ``bar`` 根 K 线（1 起算）下一张买单——成交本应发生在下一根。"""

    params = (("bar", 0),)

    def next(self):
        if len(self) == self.p.bar:
            self.buy(size=100)


def test_buy_cannot_fill_on_a_limit_up_one_word_board(fixture_root, limit_rules):
    """涨停一字上买单不成交，挂单保留到下一根可成交的 K 线。

    2025-03-19 下单，目标是次日的涨停一字 2025-03-20。若限幅判定失效，成交会落在
    3.61——一个现实中买不到的价格（当天买单排队到收盘都未必成交）。
    """
    result = _run(fixture_root, limit_rules, BuyOnBar, bar=7)

    trades = result.trades
    assert len(trades) == 1
    assert trades.iloc[0]["date"] == pd.Timestamp("2025-03-21"), "挂单必须跳过涨停一字"
    assert trades.iloc[0]["price"] == 3.96


class BuyThenSellAfter(bt.Strategy):
    """按 bar 号各下一张单，并把「是否真的发出过卖单」记进 ``log``。"""

    params = (("buy_bar", 0), ("sell_bar", 0), ("log", None))

    def next(self):
        if len(self) == self.p.buy_bar:
            self.buy(size=100)
        elif len(self) == self.p.sell_bar and self.position:
            self.sell(size=100)
            self.p.log.append(self.data.datetime.date(0))


def test_sell_cannot_fill_on_a_limit_down_one_word_board(fixture_root, limit_rules):
    """跌停一字上卖单不成交——本 fixture 里它是最后一根，故卖单到收盘仍未成交。

    2025-04-18 买入（次日 2025-04-21 成交），再在 2025-04-21 下卖单，目标是
    2025-04-23 的跌停一字。注意 2.35 按 5% 限幅才算跌停价：若 ST 期间没登记，
    限幅会退到 10%，这笔现实中卖不掉的单就会被判成成交。
    """
    log = []
    result = _run(fixture_root, limit_rules, BuyThenSellAfter, buy_bar=28, sell_bar=29, log=log)

    assert log == [pd.Timestamp("2025-04-21").date()], "夹具失效：卖单根本没发出来，本测试将空过"

    trades = result.trades
    assert len(trades) == 1, "只应有买入成交，卖单被跌停一字挡住"
    assert trades.iloc[0]["date"] == pd.Timestamp("2025-04-21")
    assert trades.iloc[0]["size"] == 100


# --- 停牌与无成交量 ---


def test_order_carries_across_a_halt_to_the_next_available_day(fixture_root, limit_rules):
    """停牌日（2025-04-22）无记录，成交只能落在次一个有记录的日子 2025-04-23。

    这是对「停牌表现为缺记录」这一数据层事实的回归护栏（见
    ``docs/research/tdx-halt-and-limit-representation.md``）：若有人用前收盘价补出
    04-22 的占位 K 线，成交日就会变成 04-22，本断言即失败。
    """
    result = _run(fixture_root, limit_rules, BuyOnBar, bar=29)

    trades = result.trades
    assert len(trades) == 1
    assert trades.iloc[0]["date"] == LIMIT_DOWN_DAY
    assert trades.iloc[0]["date"] != HALT_DAY
    assert trades.iloc[0]["price"] == 2.35, "跌停一字上买入是可以成交的（卖方不用排队）"


def test_buy_cannot_fill_on_a_zero_volume_bar(fixture_root, limit_rules, make_prices):
    """当日成交量为 0 即不可成交，挂单保留到下一根。

    股票停牌是缺记录，但基金/债券会留下零成交量的平价 K 线；同一个「本日不可成交」
    判定必须把这条也覆盖，否则会在这些品种上凭空造出成交。
    """
    prices = make_prices([10.0, 10.0, 10.0, 10.0], start="2024-01-02")
    prices.loc[prices.index[1], "volume"] = 0

    result = run_backtest(
        prices, symbol="sh600000", strategy=BuyOnBar, cash=100_000.0, rules=limit_rules, bar=1
    )

    trades = result.trades
    assert len(trades) == 1
    assert trades.iloc[0]["date"] == pd.Timestamp("2024-01-04"), "必须跳过零成交量的 01-03"


# --- T+1 ---


class BuyAndSellInTheSameBar(bt.Strategy):
    """同一次 ``next()`` 里既买又卖——两张单会在同一根 K 线上撮合，构成 T+1 违规。"""

    params = (("bar", 0),)

    def next(self):
        if len(self) == self.p.bar:
            self.buy(size=100)
            self.sell(size=100)


def test_sell_of_shares_bought_the_same_day_is_rejected(fixture_root, limit_rules):
    """当日买入的股份当日不可卖：同日撮合的卖单被拒，不得在 T+1 之前卖出。

    卖单是**被拒绝**而非留到次日：现实中交易所直接拒单，拖到明天会凭空变成一笔
    当时并不合法的成交。
    """
    result = _run(fixture_root, limit_rules, BuyAndSellInTheSameBar, bar=2)

    trades = result.trades
    assert len(trades) == 1, "只应有买入成交；同日卖单必须被拒"
    assert trades.iloc[0]["size"] == 100


def test_sell_on_a_later_day_is_allowed(fixture_root, limit_rules):
    """T+1 只挡当日回转，不得误伤隔日卖出。

    这是上一条的对照组：若 T+1 实现成「见卖单就拒」，本测试会失败。
    """
    result = _run(fixture_root, limit_rules, BuyThenSellAfter, buy_bar=2, sell_bar=3, log=[])

    trades = result.trades
    assert len(trades) == 2, "买入次日卖出必须成交"
    assert list(trades["size"]) == [100, -100]


# --- 涨跌停价的舍入约定 ---


def test_limit_price_rounds_in_decimal_half_up_not_binary():
    """涨跌停价 = 前收盘价 × (1 ± 限幅)，按**十进制**四舍五入到分。

    这三条是实测得到的分歧样本：市场实际触及的价位全部落在十进制一侧。内建
    ``round()`` 走二进制浮点，在恰好半分处会少一分。见
    ``docs/research/tdx-halt-and-limit-representation.md``。
    """
    from mbt.rules import limit_price

    assert limit_price(14.45, 0.10, -1) == 13.01, "round(14.45 * 0.9, 2) 会错给 13.00"
    assert limit_price(15.25, 0.10, -1) == 13.73, "round(15.25 * 0.9, 2) 会错给 13.72"
    assert limit_price(5.35, 0.10, +1) == 5.89, "round(5.35 * 1.1, 2) 会错给 5.88"


def test_sell_is_blocked_when_the_bar_closes_at_the_decimal_limit_price(limit_rules):
    """跌停一字的价格是十进制算出的 13.01，不是 ``round`` 的 13.00。

    差这一分就漏判一字板：若判不出限价，13.01 这根一字板上的卖单会当天成交——
    一笔现实中卖不掉的成交。
    """
    prices = pd.DataFrame(
        {
            "open": [14.45, 14.45, 13.01, 13.20],
            "high": [14.45, 14.45, 13.01, 13.20],
            "low": [14.45, 14.45, 13.01, 13.20],
            "close": [14.45, 14.45, 13.01, 13.20],
            "volume": [1000] * 4,
        },
        index=pd.bdate_range("2024-01-02", periods=4),
    )

    class BuyThenSell(bt.Strategy):
        def next(self):
            if len(self) == 1:
                self.buy(size=100)
            elif len(self) == 2 and self.position:
                self.sell(size=100)

    result = run_backtest(
        prices, symbol="sh600000", strategy=BuyThenSell, cash=100_000.0, rules=limit_rules
    )

    sells = result.trades[result.trades["size"] < 0]
    assert len(sells) == 1
    assert sells.iloc[0]["date"] == pd.Timestamp("2024-01-05"), "跌停一字必须挡住卖单"
    assert sells.iloc[0]["price"] == 13.20


# --- 买单只有一次成交机会（票据 #84） -----------------------------------------
#
# 这条开关是给**入场条件逐日重算**的规则用的：T 日算出的信号若拖到 T+3 才成交，那时它早已
# 不是当日那个信号了，而「它仍然成交了」不会报错、只让回测与真实时序对不上。
#
# 下面各例一律用**合成行情**，为的是把「下一个成交时点是什么形态」钉死到某一天：
# ``10.0 → 11.0`` 是主板 +10% 的一字板（见 `conftest.make_prices` 的告警），``10.0 → 9.0``
# 是 −10% 的一字板。真实 fixture 里那两种形态各只有一天，凑不出「停牌之后又是一字板」这类
# 组合，而那种组合正是「第一个机会用掉了吗」最容易写错的地方。

#: 六个交易日，用于合成行情。
DATES = pd.bdate_range("2025-01-02", periods=6)

#: 一个交易日都没落的标的（``sh600000``，沪主板）。
FULL_DATES = DATES

#: 缺 2025-01-06 那一天——即那天**停牌**。
HALTED_DATES = DATES.delete(2)


def _frame(dates, closes):
    """o = h = l = c 的等振幅 K 线，故相邻「恰好一个限幅」的一步就是一字板。

    不用 ``make_prices`` 那个工厂：它按 ``bdate_range`` 生成**连着的**日子，而下面「停牌」
    那两例要的正是**缺一天**的索引。
    """
    return pd.DataFrame(
        {
            "open": list(closes),
            "high": list(closes),
            "low": list(closes),
            "close": list(closes),
            "volume": [1000] * len(closes),
        },
        index=dates,
    )


def _portfolio(markets, strategy, limit_rules, **params):
    return run_portfolio_backtest(
        markets,
        strategy,
        cash=1_000_000.0,
        max_positions=1,
        rules=limit_rules,
        universe_rules=UniverseRules(min_trading_days=0),
        sizer=bt.sizers.FixedSize,
        sizer_options={"stake": 100},
        **params,
    )


class BuyOnDay(bt.Strategy):
    """在指定的**那一天**（引擎主时钟的日期）对指定标的下一张买单。

    按日期而不按 ``len(self)`` 定位：下面几例里有的标的**停牌**，于是根数在各标的之间对不齐，
    而「第几根」会因此漂移——那是测试自己制造的噪声。
    """

    params = (("day", None), ("symbol", None))

    def __init__(self):
        self.by_name = {data._name: data for data in self.datas}

    def next(self):
        if pd.Timestamp(self.datetime.date(0)) == pd.Timestamp(self.p.day):
            self.buy(data=self.by_name[self.p.symbol], size=100)


class BuyThenSellOnDay(bt.Strategy):
    """在 ``buy_day`` 买、在 ``sell_day`` 卖——两笔都落在指定的那一天。"""

    params = (("buy_day", None), ("sell_day", None), ("symbol", None))

    def __init__(self):
        self.by_name = {data._name: data for data in self.datas}

    def next(self):
        today = pd.Timestamp(self.datetime.date(0))
        data = self.by_name[self.p.symbol]
        if today == pd.Timestamp(self.p.buy_day):
            self.buy(data=data, size=100)
        elif today == pd.Timestamp(self.p.sell_day) and self.getposition(data).size:
            self.sell(data=data, size=100)


class LimitBuyOnDay(bt.Strategy):
    """在指定那天挂一张**远低于市价**的限价买单。

    它在本日是**可成交**的（不是一字板、也没停牌），但价格不达标，故撮合试过之后订单仍然
    活着——这正是「第一个成交时点没成交」的另一条来路。
    """

    params = (("day", None), ("symbol", None), ("limit_price", 5.0))

    def __init__(self):
        self.by_name = {data._name: data for data in self.datas}

    def next(self):
        if pd.Timestamp(self.datetime.date(0)) == pd.Timestamp(self.p.day):
            self.buy(
                data=self.by_name[self.p.symbol],
                size=100,
                exectype=bt.Order.Limit,
                price=self.p.limit_price,
            )


def _one_shot_buys_on_a_limit_up_board(make_market, limit_rules, **params):
    """买单在 2025-01-03 下定、目标是 01-06 那个 +10% 一字板。

    ``params`` 空着时不传 ``one_shot_buys``——那才是真的在量**默认值**。
    """
    prices = _frame(FULL_DATES, [10.0, 10.0, 11.0, 11.0, 11.0, 11.0])
    markets = [make_market("sh600000", prices)]
    return _portfolio(markets, BuyOnDay, limit_rules, day="2025-01-03", symbol="sh600000", **params)


def test_a_one_shot_buy_is_rejected_when_the_next_bar_is_a_limit_up_board(make_market, limit_rules):
    """开着开关时，涨停一字上买不进的买单**当场拒单**，不再挂着等下一根。

    拒单而不是留到次日，是为了让「T 日的信号只对 T+1 生效」这句话成立；留着它会在一根
    几天后的 K 线上成交，而那笔成交用的早已不是当日那个信号。
    """
    result = _one_shot_buys_on_a_limit_up_board(make_market, limit_rules, one_shot_buys=True)

    assert result.trades.empty, "买单该被拒，而不是留到下一根成交"
    assert len(result.rejected) == 1
    assert result.rejected.iloc[0]["date"] == pd.Timestamp("2025-01-06").date(), "落在第一个时点"
    assert "一次成交机会" in result.rejected.iloc[0]["reason"], "拒单理由要读得懂"


def test_the_one_shot_switch_is_off_by_default(make_market, limit_rules):
    """默认关着：**一个参数都不传**时，买单照旧跳过涨停一字、在下一根成交。

    这条刻意不显式传 ``one_shot_buys=False``——那测的是「传了假」，不是默认值。两台只差
    那个开关，成交却落在不同的日子，这就是「既有行为不变」的直接对照。
    """
    result = _one_shot_buys_on_a_limit_up_board(make_market, limit_rules)

    assert len(result.trades) == 1
    assert result.trades.iloc[0]["date"] == pd.Timestamp("2025-01-07"), "跳过一字板，落在下一根"


def _one_shot_buys_over_a_halt(make_market, limit_rules, **params):
    """买单在 2025-01-03 下定，而标的 01-06 那天**停牌**（没有 K 线）。

    另需要一个 01-06 有 K 线的同伴，否则主时钟根本走不到那一天，``_try_exec`` 不会被调用
    ——那样测的就不是「第一个成交时点用掉了」，而是「那一天不存在」。（全市场几千个标的时
    这不是问题；单标的回测里则**没有**这一天，故那条来路只在多标的场景下才出现。）
    """
    halted = make_market("sh600000", _frame(HALTED_DATES, [10.0, 10.0, 10.0, 10.0, 10.0]))
    companion = make_market("sz000001", _frame(FULL_DATES, [20.0] * 6))
    return _portfolio(
        [halted, companion],
        BuyOnDay,
        limit_rules,
        day="2025-01-03",
        symbol="sh600000",
        **params,
    )


def test_a_one_shot_buy_is_rejected_when_the_next_bar_is_a_halt(make_market, limit_rules):
    """停牌挡住的那一次也算「机会用掉了」——两条来路都得拒单，不能只拒一字板那条。"""
    result = _one_shot_buys_over_a_halt(make_market, limit_rules, one_shot_buys=True)

    assert result.trades.empty, "停牌当天买不进，这张单也该拒"
    assert len(result.rejected) == 1
    assert result.rejected.iloc[0]["date"] == pd.Timestamp("2025-01-06").date(), "拒在停牌那天"
    assert "一次成交机会" in result.rejected.iloc[0]["reason"]


def test_a_buy_still_carries_across_a_halt_when_the_switch_is_off(make_market, limit_rules):
    """对照组：开关关着时，同一笔买单跨过停牌日在复牌当天成交。"""
    result = _one_shot_buys_over_a_halt(make_market, limit_rules)

    assert len(result.trades) == 1
    assert result.trades.iloc[0]["date"] == pd.Timestamp("2025-01-07"), "复牌当天成交"


def test_a_one_shot_buy_is_rejected_when_the_first_opportunity_simply_does_not_fill(
    make_market, limit_rules
):
    """本日**可成交**、撮合也试过了，而价格没到——一样算机会用掉了。

    这条把拒单的判据钉在「第一个成交时点没成交」这句**原话**上，而不是钉在「本日不可成交」
    上。两者的差别在这里露出来：这份行情全程平坦，没有任何一字板与停牌，故下面那个拒单
    只可能出自「试过了、没成」这一路。
    """
    prices = _frame(FULL_DATES, [10.0] * 6)
    markets = [make_market("sh600000", prices)]

    result = _portfolio(
        markets,
        LimitBuyOnDay,
        limit_rules,
        day="2025-01-02",
        symbol="sh600000",
        one_shot_buys=True,
    )

    assert result.trades.empty, "限价 5 元买 10 元的票，本就不该成交"
    assert len(result.rejected) == 1, "机会用掉了就该拒，而不是一直挂着"
    assert result.rejected.iloc[0]["date"] == pd.Timestamp("2025-01-03").date(), "拒在第一个时点"
    assert "一次成交机会" in result.rejected.iloc[0]["reason"]


def test_a_limit_buy_keeps_waiting_when_the_switch_is_off(make_market, limit_rules):
    """对照组：开关关着时，同一张限价单一直挂着，且**不为没成交而被拒**。

    它最终也不会成交（价格永远差得远），故这里能证明的只是「没有被拒单」——而那正是要证的：
    关掉开关就等于回到「挂单逐根重试」。
    """
    prices = _frame(FULL_DATES, [10.0] * 6)
    markets = [make_market("sh600000", prices)]

    result = _portfolio(markets, LimitBuyOnDay, limit_rules, day="2025-01-02", symbol="sh600000")

    assert result.trades.empty
    assert result.rejected.empty, "关着开关时，没成交的限价单不该被拒"


def test_a_sell_is_not_affected_by_the_one_shot_switch(make_market, limit_rules):
    """**卖单不受这条开关影响**：跌停一字上卖不出去，挂单必须继续等。

    拒掉它等于把一笔已经决定要卖的丢掉。这也与 A 股的不对称一致：涨停一字买不进但卖得出，
    跌停一字卖不出但买得到——故这条开关**只**动买单那一侧。
    """
    prices = _frame(FULL_DATES, [10.0, 10.0, 9.0, 9.0, 9.0, 9.0])
    markets = [make_market("sh600000", prices)]

    result = _portfolio(
        markets,
        BuyThenSellOnDay,
        limit_rules,
        buy_day="2025-01-02",
        sell_day="2025-01-03",
        symbol="sh600000",
        one_shot_buys=True,
    )

    fills = result.trades.sort_values("date").reset_index(drop=True)
    assert len(fills) == 2, "买卖各一笔：卖单必须留着，而不是被这条开关拒掉"
    assert fills.iloc[0]["date"] == pd.Timestamp("2025-01-03")
    assert fills.iloc[1]["size"] == -100
    assert fills.iloc[1]["date"] == pd.Timestamp("2025-01-07"), "跌停一字挡住，次日才成交"
    assert fills.iloc[1]["price"] == 9.0
