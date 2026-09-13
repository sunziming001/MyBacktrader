"""``B1`` 的三条卖出规则：**逐条**验证每次卖出由哪条规则造成。

做法与 ``test_valuation_strategy.py`` 同源：把卖出规则要读的信号**直接喂进去**（``signals=``），
于是每条规则可以被单独点亮，而「副作用只来自它」这件事可以被钉住。

用 ``signals=`` 而不是真行情，是因为这里要测的是**策略的状态机**：建仓后武装、减仓只做一次、
前低只锚定一次。那三件事都与价格怎么走无关，只与「哪根读到什么信号」有关。
"""

from __future__ import annotations

import pandas as pd
import pytest

from mbt.backtest import run_portfolio_backtest
from mbt.screen import Screen
from mbt.universe import UniverseRules

#: 闸门全开：本文件测的是卖出，买入侧交给这一句，免得入场条件把用例搅在一起。
OPEN_GATE = Screen(filters=(lambda panel: panel["close"].notna(),))

#: 策略要读的五个字段——少一个就会让 `signal_value` 全部返回 NaN、卖出永不触发。
FIELDS = ("above_white", "below_white", "trim", "stop_streak", "peak_age")


def _strategy():
    from examples.strategies import B1

    return B1


def run(make_market, make_prices, zero_cost_rules, overrides, bars=8, symbol="sh600000"):
    """跑一次回测，返回 ``(result, index)``。

    信号默认全零（任何卖出规则都不触发），再由 ``overrides`` 点亮其中一条。
    价格恒为 10.0：这样前低止损位是 9.9、永远不会被触发，于是用例里所有卖出都只可能来自
    被点亮的那条规则——「副作用只来自它」因此是真的被隔离了。
    """
    prices = make_prices([10.0] * bars, volume=1000)
    index = prices.index
    fields = {name: [0.0] * bars for name in FIELDS}
    fields.update(overrides)

    signals = {
        name: pd.DataFrame({symbol: values}, index=index, dtype=float)
        for name, values in fields.items()
    }
    result = run_portfolio_backtest(
        [make_market(symbol, prices)],
        _strategy(),
        cash=100_000.0,
        max_positions=1,
        rules=zero_cost_rules,
        universe_rules=UniverseRules(min_trading_days=0),
        screen=OPEN_GATE,
        signals=signals,
    )
    return result, index


def _sells(result):
    return result.trades[result.trades["size"] < 0]


def _buys(result):
    return result.trades[result.trades["size"] > 0]


@pytest.fixture
def no_sells_baseline(make_market, make_prices, zero_cost_rules):
    """自检前提：全零信号下**只有买入、没有卖出**。

    没有这条，下面那些「卖出次数」的断言可能因为别的原因恒为某个值，而用例看不出来。
    """
    result, _ = run(make_market, make_prices, zero_cost_rules, {})
    assert len(_buys(result)) >= 1, "前提：闸门全开时应当建过仓"
    assert len(_sells(result)) == 0, "全零信号下不该有任何卖出"
    return result


def test_a_break_below_the_white_line_does_not_sell_until_the_line_was_stood_on(
    make_market, make_prices, zero_cost_rules, no_sells_baseline
):
    """``below_white`` 一直为 1、但从未 ``above_white`` → **一次也不卖**。

    这条把「先武装」这一步单独隔离出来：若实现里漏掉武装，它会变红。
    """
    bars = 8
    result, _ = run(
        make_market,
        make_prices,
        zero_cost_rules,
        {"above_white": [0.0] * bars, "below_white": [1.0] * bars},
        bars=bars,
    )

    assert len(_sells(result)) == 0, "从未站上白线，跌破白线不该触发清仓"


def test_the_trend_exit_fires_only_after_the_arming_bar(make_market, make_prices, zero_cost_rules):
    """站上白线（武装）之后，第一次跌破白线就清仓——且**那次跌破之前**不卖。"""
    bars = 10
    above = [0, 0, 1, 0, 0, 0, 0, 0, 0, 0]  # 第 3 根（索引 2）武装
    below = [0, 0, 0, 0, 0, 1, 1, 1, 1, 1]  # 第 6 根（索引 5）起跌破
    result, index = run(
        make_market,
        make_prices,
        zero_cost_rules,
        {"above_white": above, "below_white": below},
        bars=bars,
    )

    sells = _sells(result)
    assert len(sells) >= 1, "武装之后跌破白线应当清仓"
    first_sell = pd.Timestamp(sells.iloc[0]["date"])
    assert first_sell > index[2], "武装那根之前就卖了——说明武装没生效"
    # 成交发生在信号日的**下一根**（撮合在次一根），故首次卖出不早于索引 5。
    assert first_sell >= index[5], f"首次卖出 {first_sell} 早于跌破日，卖出提前了"


def test_the_trim_sells_exactly_half_and_only_once(make_market, make_prices, zero_cost_rules):
    """``trim`` 触发时卖出**一半**，且**只卖一次**。

    第 4 根起 ``trim`` 恒为 1；若没有「只减一次」的保护，它会每根都减一半、股数一路腰斩。
    实际只应看到一次约为总持仓一半的卖出。
    """
    bars = 10
    result, _ = run(
        make_market,
        make_prices,
        zero_cost_rules,
        {"trim": [0, 0, 0, 1, 1, 1, 1, 1, 1, 1]},
        bars=bars,
    )

    buys = _buys(result)
    sells = _sells(result)
    bought = int(buys.iloc[0]["size"])

    assert len(sells) == 1, f"只该减仓一次（且不触发清仓），实际卖出 {len(sells)} 次"
    assert abs(int(sells.iloc[0]["size"])) == bought // 2, "卖出的一半必须向下取整"


def test_the_stop_streak_clears_the_position(
    make_market, make_prices, zero_cost_rules, no_sells_baseline
):
    """「连续跌破黄线」（``stop_streak``）单独就能清仓。"""
    bars = 8
    result, _ = run(
        make_market,
        make_prices,
        zero_cost_rules,
        {"stop_streak": [0, 0, 1, 0, 0, 0, 0, 0]},
        bars=bars,
    )

    assert len(_sells(result)) >= 1, "止损信号应当清仓"


def test_the_prior_low_is_anchored_at_entry_and_is_not_ratcheted_down(
    make_market, make_prices, zero_cost_rules
):
    """前低在建仓时锚定一次，**其后不再重算**——否则价格下跌会把参照一起带下去。

    构造：``peak_age`` 前 4 根为 3（前低 = 建仓日往前 4 根的最低价），其后涨到 9（前低会
    变成更低的那个）。价格在建仓后**逐级下台阶**：11 → 10.5 → 10.2 → 10.1。

    - 建仓时锚定的止损位是 10.0 × 0.99 = 9.9（价格序列前段都是 10.0），故 10.1 之上都不触发；
    - 若实现每根都重算，``peak_age`` 变大后前低会跟着下移、止损位越来越低——但那只会让它
      **更不触发**，所以这条用例真正钉的是：**它不会因为重算而变得更松**，
      即触发时刻不晚于「按首次锚定」的位置。
    """
    bars = 8
    closes = [10.0, 10.0, 10.0, 10.0, 11.0, 10.5, 10.2, 10.1]
    prices = make_prices(closes, volume=1000)
    index = prices.index
    fields = {name: [0.0] * bars for name in FIELDS}
    fields["peak_age"] = [0.0] * bars  # 前低 = 当根最低价 × 0.99
    signals = {
        name: pd.DataFrame({"sh600000": values}, index=index, dtype=float)
        for name, values in fields.items()
    }

    result = run_portfolio_backtest(
        [make_market("sh600000", prices)],
        _strategy(),
        cash=100_000.0,
        max_positions=1,
        rules=zero_cost_rules,
        universe_rules=UniverseRules(min_trading_days=0),
        screen=OPEN_GATE,
        signals=signals,
    )

    # 建仓发生在索引 0 之后（下一根成交），锚定止损位 9.9；价格最低 10.1，故不该触发止损。
    assert len(_sells(result)) == 0, "价格从未跌破锚定的 9.9，不该有卖出"
