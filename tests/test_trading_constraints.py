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

from mbt.backtest import run_backtest
from mbt.data import TdxDataSource

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