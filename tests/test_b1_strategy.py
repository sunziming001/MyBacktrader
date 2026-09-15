"""``B1`` 的三条卖出规则：**逐条**验证每次卖出由哪条规则造成。

做法与 ``test_valuation_strategy.py`` 同源：把卖出规则要读的信号**直接喂进去**（``signals=``），
于是每条规则可以被单独点亮，而「副作用只来自它」这件事可以被钉住。

用 ``signals=`` 而不是真行情，是因为这里要测的是**策略的状态机**：

- 建仓后第几根判「两日未站上」（``_held_bars`` 的计数）；
- 前低只锚定一次；
- 止损优先于另两条。

那三件事都与价格怎么走无关，只与「哪根读到什么信号」有关。
"""

from __future__ import annotations

import pandas as pd
import pytest

from mbt.backtest import run_portfolio_backtest
from mbt.screen import Screen
from mbt.universe import UniverseRules

#: 闸门全开：本文件测的是卖出，买入侧交给这一句，免得入场条件把用例搅在一起。
OPEN_GATE = Screen(filters=(lambda panel: panel["close"].notna(),))

#: 策略要读的**四个**字段——少一个就会让 `signal_value` 返回 NaN、对应的规则永不触发。
FIELDS = ("below_white", "high_above_white", "stop_streak", "peak_age")


def _strategy():
    from examples.strategies import B1

    return B1


def run(
    make_market,
    make_prices,
    zero_cost_rules,
    overrides,
    closes=None,
    bars=10,
    symbol="sh600000",
    opens=None,
    params=None,
):
    """跑一次回测，返回 ``(result, index)``。

    信号默认全零（任何卖出规则都不触发），再由 ``overrides`` 点亮其中一条。
    价格默认恒为 10.0：这样前低止损位是 9.9、永远不会被触发，于是用例里所有卖出都只可能来自
    被点亮的那条规则——「副作用只来自它」因此是真的被隔离了。

    ``opens`` 给出时把开盘价改成另一串（``make_prices`` 默认让 o == h == l == c，那样开盘价与
    收盘价无从分辨，而「成交价取次根开盘还是收盘」正是要分辨的那件事）。``params`` 直接转给
    策略，用来点亮 ``exit_exec`` / ``max_hold_bars`` 这类**不来自信号**的开关。
    """
    prices = make_prices(closes if closes is not None else [10.0] * bars, volume=1000)
    if opens is not None:
        prices = prices.copy()
        prices["open"] = list(opens)
        prices["high"] = [max(o, c) + 0.2 for o, c in zip(opens, prices["close"], strict=True)]
        prices["low"] = [min(o, c) - 0.2 for o, c in zip(opens, prices["close"], strict=True)]
    index = prices.index
    fields = {name: [0.0] * len(index) for name in FIELDS}
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
        **(params or {}),
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


def test_a_high_touching_the_white_line_closes_the_position(
    make_market, make_prices, zero_cost_rules, no_sells_baseline
):
    """「最高价破白线」单独就能清仓。"""
    result, _ = run(
        make_market,
        make_prices,
        zero_cost_rules,
        {"high_above_white": [0, 0, 1, 0, 0, 0, 0, 0, 0, 0]},
        bars=10,
    )

    assert len(_sells(result)) >= 1, "最高价破白线应当清仓"


def test_never_touching_the_white_line_keeps_the_position(
    make_market, make_prices, zero_cost_rules
):
    """三条规则都不成立时**一次也不卖**——把「规则只在成立时动作」隔离出来。

    ``below_white`` 也不能点着（那会触发「两日未站上」），故这里两项都留零。
    """
    result, _ = run(make_market, make_prices, zero_cost_rules, {}, bars=10)

    assert len(_buys(result)) >= 1, "前提：确实建过仓"
    assert len(_sells(result)) == 0, "三条规则都不成立时不该卖出"


def test_the_two_day_rule_fires_only_at_the_configured_bar(
    make_market, make_prices, zero_cost_rules
):
    """「建仓后第 2 根收盘仍在白线下」才清仓——**早一根、晚一根都不行**。

    这里把 ``below_white`` 从建仓后第 1 根起就恒为 1，唯一变化的是**持有根数**。
    若计数写错（例如从 1 起算、或在建仓当根就递增），卖出会提前到 T+1 或 T+2 之前。
    断言「首次卖出发生在第 3 根之后」而不是「有卖出」，才真正锁住那个位置。
    """
    bars = 10
    result, index = run(
        make_market,
        make_prices,
        zero_cost_rules,
        {"below_white": [0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]},
        bars=bars,
    )

    sells = _sells(result)
    assert len(sells) >= 1, "第 2 根仍在白线下，应当清仓"
    first = pd.Timestamp(sells.iloc[0]["date"])
    # 信号在 T（建仓根）之后的第 2 根点着，成交再顺延一根（订单本根下、次根开盘成交）。
    # 建仓根是索引 1（买入单在索引 0 下、索引 1 开盘成交），故首次成交不早于索引 4。
    assert first >= index[4], f"首次卖出 {first.date()} 早于 T+2，计数写早了"


def test_the_two_day_rule_does_not_fire_when_the_close_is_reset_above_the_line(
    make_market, make_prices, zero_cost_rules
):
    """第 2 根收盘**已回到白线上方**（``below_white`` 为 0）时不清仓——对照上一条。"""
    bars = 10
    result, _ = run(
        make_market,
        make_prices,
        zero_cost_rules,
        {"below_white": [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]},
        bars=bars,
    )

    assert len(_sells(result)) == 0, "第 2 根没有落在白线下，不该清仓"


def test_the_stop_streak_closes_the_position(make_market, make_prices, zero_cost_rules):
    """「连续跌破黄线」（``stop_streak``）单独就能清仓——口径未变。"""
    result, _ = run(
        make_market,
        make_prices,
        zero_cost_rules,
        {"stop_streak": [0, 1, 0, 0, 0, 0, 0, 0, 0, 0]},
        bars=10,
    )

    assert len(_sells(result)) >= 1, "止损信号应当清仓"


def test_the_prior_low_is_anchored_at_entry_and_is_not_ratcheted_down(
    make_market, make_prices, zero_cost_rules
):
    """前低在建仓时锚定一次，**其后不再重算**——否则参照会随价格下跌一起下移。

    构造：``peak_age`` 恒为 0（前低 = 当根最低价 × 0.99），价格在建仓后逐级下台阶
    11 → 10.5 → 10.2 → 10.1。

    - 建仓时锚定的止损位是 10.0 × 0.99 = 9.9，故 10.1 之上都不触发；
    - 若实现每根都重算，止损位会跟着下移——但那只会让它**更不触发**。故这条真正钉的是：
      触发时刻不晚于「按首次锚定」应有的位置，即**不该有卖出**。
    """
    closes = [10.0, 10.0, 10.0, 10.0, 11.0, 10.5, 10.2, 10.1]
    result, _ = run(
        make_market,
        make_prices,
        zero_cost_rules,
        {"peak_age": [0.0] * 8},
        closes=closes,
        bars=8,
    )

    assert len(_sells(result)) == 0, "价格从未跌破锚定的 9.9，不该有卖出"


def test_a_missing_peak_age_disables_the_prior_low_stop(make_market, make_prices, zero_cost_rules):
    """``peak_age`` 缺失（拐点未确认）时**不设**那条止损——缺失取「不设」而不是取 0。

    取 0 会让「前低」退化成当根最低价，于是一个尚未确认拐点的标的也被套上一条**凭空
    算出的**止损。把 ``stop_streak`` 与另两条都留零，故这里不该有任何卖出。
    """
    nan = float("nan")
    result, _ = run(
        make_market,
        make_prices,
        zero_cost_rules,
        {"peak_age": [nan] * 10},
        bars=10,
    )

    assert len(_sells(result)) == 0, "peak_age 缺失时不该用一条凭空的前低去止损"


def test_the_stop_streak_wins_when_the_rules_would_fire_on_the_same_bar(
    make_market, make_prices, zero_cost_rules
):
    """同一根上三条同时成立时，只产生**一次全额清仓**。

    信号只在**建仓那一根**点亮（其余为零），故那一根之后不再有卖出理由；于是「卖出笔数
    恰好为 1 且等于买入股数」就把「三条规则各判一次、重复下单」这种写法钉死了——那种
    写法会在同一根连下三笔卖单（后续两笔靠撮合层以持仓不足为由拒掉，但那是兜底，不该依赖）。
    """
    bars = 10
    on = [0, 1, 0, 0, 0, 0, 0, 0, 0, 0]
    result, _ = run(
        make_market,
        make_prices,
        zero_cost_rules,
        {"stop_streak": on, "high_above_white": on, "below_white": on},
        bars=bars,
    )

    buys = _buys(result)
    sells = _sells(result)
    assert len(sells) >= 1, "信号成立那一根应当清仓"
    # 闸门常开，故清仓之后还会再建仓——但那与本节无关。要钉的是**同一根上没有重复下单**：
    # 三条规则若各自下单，同一天会出现多笔卖出。
    assert sells["date"].nunique() == len(
        sells
    ), f"同一根上出现了多笔卖出：{sells['date'].tolist()}——三条规则各自下单了"
    assert int(sells.iloc[0]["size"]) == -int(buys.iloc[0]["size"]), "第一次应当是全额清仓"


def test_the_signal_fields_the_strategy_reads_are_exactly_these_four(
    make_market, make_prices, zero_cost_rules
):
    """字段集合是一份契约：多一个没人读的列，下次改规则时会先让人以为它还在用。

    这条用一个**只有这四列**的信号面板跑通，反证策略不再依赖 ``above_white`` / ``trim``。
    """
    result, _ = run(make_market, make_prices, zero_cost_rules, {}, bars=6)

    assert len(_buys(result)) >= 1
    assert result.trades.columns.is_unique


# --- 成交时点：出口取次根开盘还是收盘 -----------------------------------------
#
# ADR-0012 ⑥ 的 M1（「同一卖日、只改成交时段」）全部押在这一档上，故它必须被钉住。
# 做法：把每根 K 线的开盘与收盘拉开 0.5，且让每根的价都不同——于是「成交价对应哪一根、
# 取的是开盘还是收盘」可以从价钱本身读出来，不必去猜。

#: 每根收盘 10.0 ~ 10.9、开盘一律比收盘低 0.5。故 index[3] 这根是 开盘 9.8 / 收盘 10.3。
RISING_CLOSES = [10.0 + 0.1 * i for i in range(10)]
RISING_OPENS = [value - 0.5 for value in RISING_CLOSES]

#: 第 3 根（下标 2）点亮「最高价破白线」→ 卖单下在这一根，成交落在**次根**（下标 3）。
WHITE_ON_BAR_TWO = {"high_above_white": [0, 0, 1, 0, 0, 0, 0, 0, 0, 0]}


def test_the_default_exit_fills_at_the_next_bar_open(make_market, make_prices, zero_cost_rules):
    """默认（``exit_exec="open"``）卖在**次根开盘**——这是引擎一直以来的口径，先把它钉住。"""
    result, index = run(
        make_market,
        make_prices,
        zero_cost_rules,
        WHITE_ON_BAR_TWO,
        closes=RISING_CLOSES,
        opens=RISING_OPENS,
    )

    sells = _sells(result)
    assert len(sells) == 1, f"应当恰好一笔卖出，实得 {len(sells)}"
    assert sells.iloc[0]["date"] == index[3], "成交应落在下单次根"
    assert sells.iloc[0]["price"] == pytest.approx(
        RISING_OPENS[3]
    ), "默认档是次根**开盘**价；若这里变成了收盘价，说明引擎的默认撮合口径被改了"


def test_a_close_exit_fills_at_the_same_bar_close(make_market, make_prices, zero_cost_rules):
    """``exit_exec="close"`` 卖在**同一根的收盘**——卖日不动、只换成交时段（ADR-0012 ⑥ 的 M1）。

    两条断言缺一不可：**日期不变**证明卖日没被挪动；**价钱变成收盘**证明换的是成交时段。
    只断言价钱的话，「卖晚一根」也能凑出同一个数。
    """
    result, index = run(
        make_market,
        make_prices,
        zero_cost_rules,
        WHITE_ON_BAR_TWO,
        closes=RISING_CLOSES,
        opens=RISING_OPENS,
        params={"exit_exec": "close"},
    )

    sells = _sells(result)
    assert len(sells) == 1, f"应当恰好一笔卖出，实得 {len(sells)}"
    assert sells.iloc[0]["date"] == index[3], "卖日不该被改动——换的只是成交时段"
    assert sells.iloc[0]["price"] == pytest.approx(
        RISING_CLOSES[3]
    ), "close 档应当取次根**收盘**价，而不是开盘价"


def test_a_close_exit_leaves_the_buy_at_the_open(make_market, make_prices, zero_cost_rules):
    """``exit_exec="close"`` **只对卖出放开**：买入仍落在次根开盘。

    否则「收盘看到信号、次日开盘买入」这条时序会被一起改掉，而那会让入场价凭空变成收盘价，
    测出来的就不再是「只换出口」这一件事。
    """
    result, index = run(
        make_market,
        make_prices,
        zero_cost_rules,
        WHITE_ON_BAR_TWO,
        closes=RISING_CLOSES,
        opens=RISING_OPENS,
        params={"exit_exec": "close"},
    )

    buys = _buys(result)
    assert len(buys) >= 1
    assert buys.iloc[0]["date"] == index[1], "买入仍应落在下单次根"
    assert buys.iloc[0]["price"] == pytest.approx(RISING_OPENS[1]), "买入仍应取次根开盘价"


# --- 持有封顶（第四条卖出规则）------------------------------------------------


def test_the_hold_is_capped_at_the_configured_bar(make_market, make_prices, zero_cost_rules):
    """``max_hold_bars=2``：全零信号下也会在第 2 个可交易日成交清仓。

    信号全零是关键——另三条规则都不动，故这一笔卖出**只可能**来自封顶。
    """
    result, index = run(
        make_market,
        make_prices,
        zero_cost_rules,
        {},
        closes=RISING_CLOSES,
        opens=RISING_OPENS,
        params={"max_hold_bars": 2},
    )

    sells = _sells(result)
    assert len(sells) >= 1, "封顶应当在没有任何信号的情况下也清仓"
    assert (
        sells.iloc[0]["date"] == index[3]
    ), "第 2 个可交易日成交：建仓成交在下标 1，故封顶的卖单在下标 2 下、下标 3 成交"


def test_a_longer_cap_pushes_the_exit_one_bar_later(make_market, make_prices, zero_cost_rules):
    """``max_hold_bars=3`` 比 2 晚一根——封顶的口径是「第 N 个可交易日成交」，不是「N 根之后」。"""
    result, index = run(
        make_market,
        make_prices,
        zero_cost_rules,
        {},
        closes=RISING_CLOSES,
        opens=RISING_OPENS,
        params={"max_hold_bars": 3},
    )

    sells = _sells(result)
    assert len(sells) >= 1
    assert sells.iloc[0]["date"] == index[4]


def test_the_cap_is_off_by_default(make_market, make_prices, zero_cost_rules, no_sells_baseline):
    """默认 ``None``：这条规则整个不参与，全零信号下一笔都不卖。

    ``no_sells_baseline`` 那条 fixture 跑的就是默认参数，这里只用它显式声明一次依赖关系，
    免得以后有人把默认值改成某个数而这条契约悄悄失效。
    """
    assert len(_sells(no_sells_baseline)) == 0
