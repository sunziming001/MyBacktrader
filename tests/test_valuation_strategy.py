"""估值策略与接线：信号的传递、买入闸门、以及三条卖出条件（票据 #37）。

本文件的重心是**接线**——估值信号要同时到达选股规则与策略，而卖出条件全靠它。三条卖出
各自单独测一遍，另有一条专测「信号缺失时不得凭空卖出」。

「信号缺失不得动作」这条不是形式主义：若把缺失当成 0，``PE ≤ 0`` 会立刻为真，于是**每个
没有财报的日子都会触发一次卖出**——而那种错误不会报错，只会在明细里表现为异常密集的卖出。
"""

from __future__ import annotations

import pandas as pd
import pytest

from mbt.backtest import run_portfolio_backtest
from mbt.data.errors import MarketDataError
from mbt.data.valuation import clip_fields
from mbt.screen import Screen, valuation_screen
from mbt.universe import UniverseRules

STRATEGY = "examples.strategies:ValuationReversal"


def valuation_frames(index, columns, **overrides):
    """一组估值信号帧，默认全 1.0（合法值），按需覆盖。"""
    base = {
        "pe": 1.0,
        "pe_percentile": 0.5,
        "peg": 0.5,
    }
    base.update(overrides)
    return {
        name: pd.DataFrame(value, index=index, columns=columns, dtype=float)
        for name, value in base.items()
    }


def columns_of(markets):
    return [market.symbol for market in markets]


# --- 选股规则：买入条件 -------------------------------------------------------


def test_the_valuation_screen_keeps_the_cheapest_qualifying_symbol():
    """三个条件都过、且按**百分位最低**取前 N——因子是 ``1 − 百分位``。"""
    index = pd.bdate_range("2024-01-02", periods=3)
    columns = ["a", "b", "c"]
    frames = valuation_frames(
        index,
        columns,
        pe=[[1.0, 1.0, 1.0]] * 3,
        pe_percentile=[[0.01, 0.05, 0.20]] * 3,  # c 的 0.20 超过 8%，落选
        peg=[[0.3, 0.6, 0.3]] * 3,
    )
    from mbt.data.panel import Panel

    result = valuation_screen(top_n=1).apply(Panel(frames))

    assert result.candidates(index[0]) == ["a"], "百分位最低者优先"


def test_the_valuation_screen_excludes_non_positive_pe_and_out_of_range_peg():
    index = pd.bdate_range("2024-01-02", periods=2)
    columns = ["cheap", "loss", "expensive_growth", "negative_peg"]
    frames = valuation_frames(
        index,
        columns,
        pe=[[1.0, -1.0, 1.0, 1.0]] * 2,
        pe_percentile=[[0.01, 0.01, 0.01, 0.01]] * 2,
        peg=[[0.3, 0.3, 1.5, -0.2]] * 2,
    )
    from mbt.data.panel import Panel

    result = valuation_screen(top_n=5).apply(Panel(frames))

    assert result.candidates(index[0]) == ["cheap"], "亏损、PEG 过高、PEG 为负都该落选"


def test_the_valuation_screen_yields_nothing_when_everything_is_missing():
    """全缺失 → 空候选集。缺失取假（过滤信号的既定契约），不是「全都合格」。"""
    index = pd.bdate_range("2024-01-02", periods=2)
    frames = {
        name: pd.DataFrame(float("nan"), index=index, columns=["a"], dtype=float)
        for name in ("pe", "pe_percentile", "peg")
    }
    from mbt.data.panel import Panel

    assert valuation_screen(top_n=1).apply(Panel(frames)).candidates(index[0]) == []


# --- 先算后截 -----------------------------------------------------------------


def test_clip_fields_keeps_only_the_window_and_the_requested_symbols(make_market, make_prices):
    markets = [make_market("sh600000", make_prices([10.0] * 200))]
    index = markets[0].prices.index
    frames = valuation_frames(index, ["sh600000"], pe=1.0)

    clipped = clip_fields(frames, markets, start="2024-03-01", end="2024-04-30")

    got = clipped["pe"]
    assert got.index.min() >= pd.Timestamp("2024-03-01")
    assert got.index.max() <= pd.Timestamp("2024-04-30")
    assert got.index.equals(
        index[(index >= "2024-03-01") & (index <= "2024-04-30")]
    ), "截断口径必须与 slice_markets 一致（两端含）"
    assert list(got.columns) == ["sh600000"]


def test_clip_fields_rejects_a_frame_missing_a_market_symbol(make_market, make_prices):
    """信号覆盖不到某个行情标的 → 报错，而不是静默少一列。

    反向（信号**多出**列）则是允许的且必要的：信号是在**完整历史**上算的，而截面可能因区间
    无数据而少几个标的——那几列必须丢掉。故只有「少」才报错。
    """
    markets = [
        make_market("sh600000", make_prices([10.0] * 20)),
        make_market("sz000001", make_prices([10.0] * 20)),
    ]
    index = markets[0].prices.index
    frames = valuation_frames(index, ["sh600000"], pe=1.0)  # 少了 sz000001

    with pytest.raises(MarketDataError, match="缺少标的"):
        clip_fields(frames, markets)


def test_clip_fields_drops_columns_the_markets_no_longer_have(make_market, make_prices):
    """信号**多出**列时静默丢掉——那正是「先算后截」的必然结果。"""
    markets = [make_market("sh600000", make_prices([10.0] * 20))]
    index = markets[0].prices.index
    frames = valuation_frames(index, ["sh600000", "sz000001"], pe=1.0)

    clipped = clip_fields(frames, markets, start=None)

    assert list(clipped["pe"].columns) == ["sh600000"]


# --- 策略：三条卖出条件 -------------------------------------------------------


class _Holder:
    """把买入闸门设成全开：用一条**不过滤**的 Screen，于是策略只管卖出。"""

    @staticmethod
    def open_gate():
        return Screen()


def _run_with_signals(make_market, make_prices, zero_cost_rules, **overrides):
    """6 根恒定价，信号可覆盖；返回回测结果。"""
    from examples.strategies import ValuationReversal

    markets = [make_market("sh600000", make_prices([10.0] * 8))]
    index = markets[0].prices.index
    signals = valuation_frames(index, ["sh600000"], **overrides)

    return run_portfolio_backtest(
        markets,
        ValuationReversal,
        cash=100_000.0,
        max_positions=1,
        rules=zero_cost_rules,
        universe_rules=UniverseRules(min_trading_days=0),
        screen=_Holder.open_gate(),
        signals=signals,
    )


def _sells(result):
    return result.trades[result.trades["size"] < 0]


@pytest.mark.parametrize(
    ("field", "benign", "triggering"),
    [
        ("pe", 0.5, -0.5),  # 转亏：PE ≤ 0 触发，故触发值是**负**的
        ("pe_percentile", 0.5, 0.9),  # 百分位 > 0.70 触发
        ("peg", 0.5, 2.0),  # PEG > 1.1 触发
    ],
)
def test_each_exit_condition_sells(
    make_market, make_prices, zero_cost_rules, field, benign, triggering
):
    """三条卖出条件各自都能单独触发一次卖出，且**副作用只来自那条条件**。

    用「触发值 vs 正常值」对照，而不是只断言「有卖出」：闸门全开时买卖会反复发生，故卖出
    笔数本身未必是 1；真正要钉的是「只换掉那一个信号值，卖出就从有变无」。
    """
    series = [benign] * 2 + [triggering] * 6

    triggered = _run_with_signals(make_market, make_prices, zero_cost_rules, **{field: series})

    untouched = _run_with_signals(
        make_market, make_prices, zero_cost_rules, **{field: [benign] * 8}
    )

    assert len(_sells(triggered)) >= 1, "该条件应当触发卖出"
    assert len(_sells(untouched)) == 0, "把该信号换成正常值就不该卖出——副作用只来自它"


def test_missing_signals_never_trigger_an_exit(make_market, make_prices, zero_cost_rules):
    """**信号全缺失时不得卖出。**

    若把缺失当成 0，``PE ≤ 0`` 会立刻为真——于是每个没有财报的日子都卖一次，而那种错误
    只会表现为「卖出异常密集」，不会报错。``NaN`` 的比较为假，「不知道」因此天然是「不动作」。
    """
    missing = [float("nan")] * 8

    result = _run_with_signals(
        make_market, make_prices, zero_cost_rules, pe=missing, pe_percentile=missing, peg=missing
    )

    assert len(_sells(result)) == 0
    assert len(result.trades) == 1, "买入仍然发生（买入闸门全开），但一次都不卖"


def test_the_strategy_does_not_buy_when_the_gate_says_no(make_market, make_prices, zero_cost_rules):
    """买入闸门关着就不买——买入条件属于选股规则，策略不自己重算一遍（ADR-0001）。"""
    from examples.strategies import ValuationReversal

    markets = [make_market("sh600000", make_prices([10.0] * 8))]
    index = markets[0].prices.index
    signals = valuation_frames(index, ["sh600000"])

    result = run_portfolio_backtest(
        markets,
        ValuationReversal,
        cash=100_000.0,
        max_positions=1,
        rules=zero_cost_rules,
        universe_rules=UniverseRules(min_trading_days=0),
        # 一个什么都不选的闸门
        screen=Screen(filters=(lambda panel: panel["close"] < 0.0,)),
        signals=signals,
    )

    assert len(result.trades) == 0
