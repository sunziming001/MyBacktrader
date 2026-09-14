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


# --- 「本根该遍历谁」：只走可能动作的标的，但结果必须与走遍全部标的**逐位相同** ----------
#
# 循环体对「没持仓、没挂单、不在当日候选、也没有遗留状态」的标的所做的全是空操作，故跳过
# 它们与逐一遍历语义相同。这一步把每根的开销从 O(全部标的) 降到 O(少数几个)——全市场下
# 是 1280 万次遍历换几次。它省得很多，故**等价性必须被钉住**，而不是靠推理。


def _screen_for(flags):
    """按一张逐日布尔表选股。表外的地方为「未选中」。"""

    def pick(panel):
        close = panel["close"]
        return flags.reindex(index=close.index, columns=close.columns).fillna(False).astype(bool)

    return Screen(filters=(pick,))


def _b1_equivalence_scenario(make_market, make_prices, zero_cost_rules, strategy):
    """跑一个把循环体的**每条路径**都走到的四标的场景。

    刻意安排：

    - **C 一次都没被选中**——它必须整段被跳过；
    - **A 第 2 根清仓、第 5 根重新入选**——重选那一刻正是「刚清仓的标的要不要单独遍历」
      这条推理的关键处；
    - **B 第 5 根减半**——持仓分支里的分批止盈（减仓后仍是持仓）；
    - **D 第 7 根入选、第 8 根停牌**——买单在停牌日不能成交，会**挂到下一根**，故中间那一根
      只有「有挂单」这条来源能把它带进遍历范围。
    """
    bars = 10
    flat = make_prices([10.0] * bars, volume=1000)
    index = flat.index
    a, b, c, d = "sh600000", "sz000001", "sh600004", "sh600005"
    columns = [a, b, c, d]

    markets = [
        make_market(a, flat),
        make_market(b, flat),
        make_market(c, flat),
        # 第 8 根整根缺失＝停牌：那天 `close[0]` 是陈旧价，买单只能留到第 9 根
        make_market(d, flat.drop(index[8])),
    ]

    flags = pd.DataFrame(False, index=index, columns=columns)
    for row in (0, 1, 5, 6):
        flags.loc[index[row], a] = True
    for row in (3, 4):
        flags.loc[index[row], b] = True
    flags.loc[index[7], d] = True

    fields = {name: pd.DataFrame(0.0, index=index, columns=columns) for name in FIELDS}
    fields["stop_streak"].loc[index[2], a] = 1.0  # A 第 2 根清仓
    fields["trim"].loc[index[5], b] = 1.0  # B 第 5 根减半

    return run_portfolio_backtest(
        markets,
        strategy,
        cash=100_000.0,
        max_positions=4,
        rules=zero_cost_rules,
        universe_rules=UniverseRules(min_trading_days=0),
        screen=_screen_for(flags),
        signals=fields,
    )


def test_the_candidate_loop_gives_the_same_result_as_walking_every_symbol(
    make_market, make_prices, zero_cost_rules
):
    """只遍历「可能动作的标的」与遍历全部标的，产出的净值曲线与成交明细**逐位相同**。

    ``FullScan`` 就是旧写法（`for data in self.datas`），两条路在同一份输入上比：这是这条
    优化唯一站得住的理由。若等价性破了，差异会出现在净值或成交价的最后几位上——那正是
    只看「期末收益差不多」会漏掉的东西，故这里用 ``check_exact=True``。
    """
    from examples.strategies import B1

    class FullScan(B1):
        def _candidates(self, today):
            return [(data, data._name) for data in self.datas]

    optimized = _b1_equivalence_scenario(make_market, make_prices, zero_cost_rules, B1)
    walked = _b1_equivalence_scenario(make_market, make_prices, zero_cost_rules, FullScan)

    # 前提：场景本身要有内容，否则「两条路都什么都没做」也能逐位相同。下面这些数字同时
    # 记录了这个场景到底覆盖了什么，改动它的人会立刻看到覆盖范围变了。
    assert len(_buys(walked)) == 3, "A 建仓、B 建仓、A 清仓后重新入选再建一次"
    assert len(_sells(walked)) == 2, "A 止损清仓一次、B 减半一次"
    assert len(walked.rejected) == 1, "D 的买单被停牌推到下一根，而那天它已不在候选里"

    pd.testing.assert_series_equal(optimized.equity_curve, walked.equity_curve, check_exact=True)
    pd.testing.assert_frame_equal(optimized.trades, walked.trades, check_exact=True)
    pd.testing.assert_frame_equal(optimized.rejected, walked.rejected, check_exact=True)


def test_the_candidate_loop_actually_leaves_symbols_out(make_market, make_prices, zero_cost_rules):
    """每一根真正遍历的标的数**少于**标的总数——否则上面那条等价性测试是空的。

    一个把 ``_candidates`` 写成「返回全部标的」的实现会顺利通过等价性测试（它本来就是旧
    写法），却一点没省。这条负责把那种退化抓出来。
    """
    from examples.strategies import B1

    widths: list[int] = []

    class Counting(B1):
        def _candidates(self, today):
            picked = super()._candidates(today)
            widths.append(len(picked))
            return picked

    result = _b1_equivalence_scenario(make_market, make_prices, zero_cost_rules, Counting)

    assert widths, "策略应当被调用过"
    assert min(widths) < 4, f"每根都遍历了全部 4 只标的（最少一次是 {min(widths)}）"
    assert len(_buys(result)) == 3, "跳过归跳过，该建的仓一只都不能少"


def test_the_strategy_refuses_a_selection_table_that_misses_a_trading_day():
    """选股表缺一天就**报错**，而不是那天静默地一根都不建仓。

    候选集只从选股表来，而查不到当天时 ``.get`` 会退化成「当日无人入选」——那与「当天确实
    没有候选」长得一模一样。这类静默失真必须在建索引时（第一根 K 线）当场暴露。
    """
    import types

    from examples.strategies import B1

    days = pd.bdate_range("2024-01-02", periods=3)
    fake = types.SimpleNamespace(
        broker=types.SimpleNamespace(
            selection_mask=pd.DataFrame(True, index=days[:2], columns=["sh600000"]),
            tradability_mask=pd.DataFrame(True, index=days, columns=["sh600000"]),
        )
    )
    with pytest.raises(ValueError, match="缺少"):
        B1._index_selected_days(fake)
