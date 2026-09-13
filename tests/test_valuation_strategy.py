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
from mbt.data.panel import clip_fields
from mbt.data.valuation import EQUITY, MARKET_CAP, ROE
from mbt.screen import (
    DRAWDOWN_FIELD,
    Screen,
    drawdown_fields,
    undervalued_growth_screen,
    valuation_screen,
)
from mbt.universe import UniverseRules

STRATEGY = "examples.strategies:UndervaluedGrowth"


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


# --- 低估成长的买入条件（票据 #51） -------------------------------------------


def growth_frames(index, columns, **overrides):
    """加跌幅、市值、ROE 与净资产字段的一组信号帧（默认都过门槛）。

    默认值：跌幅 0.5（> 42%）、市值 500 亿（> 100 亿）、ROE 25（> 10）、净资产 50 亿（> 0）。
    """
    frames = valuation_frames(index, columns, **overrides)
    frames[DRAWDOWN_FIELD] = pd.DataFrame(
        overrides.get("drawdown_1y", 0.5), index=index, columns=columns, dtype=float
    )
    frames[MARKET_CAP] = pd.DataFrame(
        overrides.get("market_cap", 5e10), index=index, columns=columns, dtype=float
    )
    frames[ROE] = pd.DataFrame(
        overrides.get("roe", 25.0), index=index, columns=columns, dtype=float
    )
    frames[EQUITY] = pd.DataFrame(
        overrides.get("equity", 5e9), index=index, columns=columns, dtype=float
    )
    return frames


def test_the_growth_screen_ranks_by_roe_not_by_drawdown():
    """**排序因子是 ROE**（票据 #58）——质量越高越靠前，而不是跌得越深越靠前。

    构造：三只**都过六个条件**，但「跌幅最深的是 c，ROE 最高的是 a」。按跌幅排序会选 c，
    按 ROE 必须选 **a**——这一条把「换排序因子」这件事钉住。

    三只的百分位都要 < 8%，否则它们会被过滤掉、测的就不是排序因子了（第一版写错过：把
    c 的百分位写成 0.20，它先被过滤，于是选出的与排序因子无关）。
    """
    index = pd.bdate_range("2024-01-02", periods=3)
    columns = ["a", "b", "c"]
    frames = growth_frames(
        index,
        columns,
        pe=[[1.0, 1.0, 1.0]] * 3,
        pe_percentile=[[0.001, 0.050, 0.070]] * 3,  # 三只都过 8%
        peg=[[0.3, 0.6, 0.3]] * 3,
        drawdown_1y=[[0.45, 0.60, 0.90]] * 3,  # c 跌得最深
        roe=[[30.0, 20.0, 11.0]] * 3,  # a 质量最高
    )
    from mbt.data.panel import Panel

    got = undervalued_growth_screen(top_n=1).apply(Panel(frames))

    assert got.candidates(index[0]) == ["a"], "ROE 最高者优先（而非跌幅最深者）"


def test_the_growth_screen_requires_roe_above_the_threshold():
    """ROE 必须**严格大于** 10：恰好 10 不算，10.1 才算。"""
    index = pd.bdate_range("2024-01-02", periods=2)
    columns = ["low", "edge", "high"]
    frames = growth_frames(
        index,
        columns,
        pe=[[1.0, 1.0, 1.0]] * 2,
        pe_percentile=[[0.01, 0.01, 0.01]] * 2,
        peg=[[0.3, 0.3, 0.3]] * 2,
        drawdown_1y=[[0.9, 0.9, 0.9]] * 2,
        roe=[[9.9, 10.0, 10.1]] * 2,
    )
    from mbt.data.panel import Panel

    got = undervalued_growth_screen(top_n=5).apply(Panel(frames))

    assert got.candidates(index[0]) == ["high"]


def test_a_missing_roe_excludes_the_symbol():
    """ROE 缺失（取不到权益）→ 排除，不凑数。"""
    index = pd.bdate_range("2024-01-02", periods=2)
    columns = ["known", "unknown"]
    frames = growth_frames(
        index,
        columns,
        pe=[[1.0, 1.0]] * 2,
        pe_percentile=[[0.01, 0.01]] * 2,
        peg=[[0.3, 0.3]] * 2,
        drawdown_1y=[[0.9, 0.9]] * 2,
        roe=[[25.0, float("nan")]] * 2,
    )
    from mbt.data.panel import Panel

    got = undervalued_growth_screen(top_n=5).apply(Panel(frames))

    assert got.candidates(index[0]) == ["known"]


# --- 票据 #62：净资产为正 -----------------------------------------------------


def test_the_growth_screen_excludes_a_negative_equity_company_despite_its_huge_roe():
    """净资产为负 → 排除，哪怕它的 ROE 高得离谱。

    ROE 的分子分母**同时为负**时商为正，故资不抵债的公司可以带着巨大的正 ROE 通过 ROE
    门槛。全市场回测里收益率最低的 100 只中实测有 ``sh600340``（ROE 99.1）、
    ``sz000826``（109.6）、``sz300266``（745.4）——都是这类。

    构造让 ``insolvent`` 的 ROE 远高于 ``healthy``：若过滤被删掉，它会抢走唯一的名额，
    测试随即变红。

    .. note::

        这个 fixture 里 ``insolvent`` 的 **PE 是正的**（1.0），而在真实数据里这不可能——
        净资产为负且 ROE > 10 蕴含净利为负，PE 因而必为负。故本用例只验证「过滤被正确接上」
        这一件机械事实；它**在真实数据上能拦住什么**由
        :func:`test_the_equity_filter_only_bites_when_negative_pe_is_allowed` 回答。
    """
    index = pd.bdate_range("2024-01-02", periods=2)
    columns = ["healthy", "insolvent"]
    frames = growth_frames(
        index,
        columns,
        pe=[[1.0, 1.0]] * 2,
        pe_percentile=[[0.01, 0.01]] * 2,
        peg=[[0.3, 0.3]] * 2,
        drawdown_1y=[[0.9, 0.9]] * 2,
        roe=[[25.0, 745.4]] * 2,  # 负净资产那家的 ROE 更高，必须仍被排除
        equity=[[5e9, -1e9]] * 2,
    )
    from mbt.data.panel import Panel

    got = undervalued_growth_screen(top_n=1).apply(Panel(frames))

    assert got.candidates(index[0]) == ["healthy"]


def test_the_equity_filter_only_bites_when_negative_pe_is_allowed():
    """**「净资产 > 0」被「动态PE > 0」蕴含，故在默认参数下不改变候选集。**

    证明：净资产 < 0 且 ROE > 10 ⟹ 净利TTM < 0（ROE 的分子分母同时为负时商为正）
    ⟹ PE = 市值 ÷ 净利TTM < 0 ⟹ 已被 ``profitable`` 排除。

    这条断言把「冗余」写进测试而不是留在注释里：

    - 默认参数下，加不加净资产过滤，结果**一样**（都是空集）——拦它的是 PE；
    - 把 ``pe_above`` 调到负数（显式放行小幅亏损）后，它才会通过其余条件，
      此时**只有净资产过滤**能拦住它。

    谁将来改 ``pe_above`` 的默认值、或删掉 ``profitable``，这条会告诉他净资产过滤从此
    **不再冗余**。
    """
    index = pd.bdate_range("2024-01-02", periods=2)
    columns = ["insolvent"]
    frames = growth_frames(
        index,
        columns,
        pe=[[-0.5]] * 2,  # 亏到 PE 为负——负净资产 + 正 ROE 的必然结果
        pe_percentile=[[0.01]] * 2,
        peg=[[0.3]] * 2,
        drawdown_1y=[[0.9]] * 2,
        roe=[[745.4]] * 2,
        equity=[[-1e9]] * 2,
    )
    from mbt.data.panel import Panel

    panel = Panel(frames)
    without = float("-inf")  # 关掉净资产过滤（任何有限净资产都通过）

    assert (
        undervalued_growth_screen(top_n=5).apply(panel).candidates(index[0]) == []
    ), "默认下 PE > 0 已经拦掉它"
    assert (
        undervalued_growth_screen(top_n=5, min_equity=without).apply(panel).candidates(index[0])
        == []
    ), "所以默认参数下净资产过滤是冗余的：加不加结果都一样"

    assert undervalued_growth_screen(top_n=5, pe_above=-1.0, min_equity=without).apply(
        panel
    ).candidates(index[0]) == ["insolvent"], "放行负 PE 后它会通过其余全部条件"
    assert (
        undervalued_growth_screen(top_n=5, pe_above=-1.0).apply(panel).candidates(index[0]) == []
    ), "此时只有净资产过滤能拦住它"


def test_the_growth_screen_requires_an_equity_strictly_above_zero():
    """净资产必须**严格大于** 0：恰好归零不算，1 元才算——与其余门槛同一口径。"""
    index = pd.bdate_range("2024-01-02", periods=2)
    columns = ["negative", "zero", "positive"]
    frames = growth_frames(
        index,
        columns,
        pe=[[1.0, 1.0, 1.0]] * 2,
        pe_percentile=[[0.01, 0.01, 0.01]] * 2,
        peg=[[0.3, 0.3, 0.3]] * 2,
        drawdown_1y=[[0.9, 0.9, 0.9]] * 2,
        equity=[[-1.0, 0.0, 1.0]] * 2,
    )
    from mbt.data.panel import Panel

    got = undervalued_growth_screen(top_n=5).apply(Panel(frames))

    assert got.candidates(index[0]) == ["positive"]


def test_a_missing_equity_excludes_the_symbol():
    """净资产缺失（无可用财报）→ 排除，不凑数。"""
    index = pd.bdate_range("2024-01-02", periods=2)
    columns = ["known", "unknown"]
    frames = growth_frames(
        index,
        columns,
        pe=[[1.0, 1.0]] * 2,
        pe_percentile=[[0.01, 0.01]] * 2,
        peg=[[0.3, 0.3]] * 2,
        drawdown_1y=[[0.9, 0.9]] * 2,
        equity=[[5e9, float("nan")]] * 2,
    )
    from mbt.data.panel import Panel

    got = undervalued_growth_screen(top_n=5).apply(Panel(frames))

    assert got.candidates(index[0]) == ["known"]


def test_the_growth_screen_requires_a_drawdown_deeper_than_the_threshold():
    """跌幅必须**严格大于**门槛：42% 本身不算，42.1% 才算。"""
    index = pd.bdate_range("2024-01-02", periods=2)
    columns = ["shallow", "edge", "deep"]
    frames = growth_frames(
        index,
        columns,
        pe=[[1.0, 1.0, 1.0]] * 2,
        pe_percentile=[[0.01, 0.01, 0.01]] * 2,
        peg=[[0.3, 0.3, 0.3]] * 2,
        drawdown_1y=[[0.41, 0.42, 0.421]] * 2,
    )
    from mbt.data.panel import Panel

    got = undervalued_growth_screen(top_n=5).apply(Panel(frames))

    assert got.candidates(index[0]) == ["deep"], "恰好 42% 不算（严格大于）"


def test_a_missing_drawdown_excludes_the_symbol():
    """跌幅缺失（回看窗口不足）→ 排除，而不是当成 0 或合格。"""
    index = pd.bdate_range("2024-01-02", periods=2)
    columns = ["known", "unknown"]
    frames = growth_frames(
        index,
        columns,
        pe=[[1.0, 1.0]] * 2,
        pe_percentile=[[0.01, 0.01]] * 2,
        peg=[[0.3, 0.3]] * 2,
        drawdown_1y=[[0.9, float("nan")]] * 2,
    )
    from mbt.data.panel import Panel

    got = undervalued_growth_screen(top_n=5).apply(Panel(frames))

    assert got.candidates(index[0]) == ["known"]


def test_the_growth_screen_keeps_the_three_original_conditions():
    """原来那三条一个都不能少——亏损、百分位不够低、PEG 越界的都要落选。"""
    index = pd.bdate_range("2024-01-02", periods=2)
    columns = ["ok", "loss", "expensive", "growing_too_fast", "negative_peg"]
    frames = growth_frames(
        index,
        columns,
        pe=[[1.0, -1.0, 1.0, 1.0, 1.0]] * 2,
        pe_percentile=[[0.01, 0.01, 0.20, 0.01, 0.01]] * 2,
        peg=[[0.3, 0.3, 0.3, 1.5, -0.2]] * 2,
        drawdown_1y=[[0.9] * 5] * 2,
    )
    from mbt.data.panel import Panel

    got = undervalued_growth_screen(top_n=5).apply(Panel(frames))

    assert got.candidates(index[0]) == ["ok"]


def test_a_missing_market_cap_excludes_the_symbol():
    """市值缺失 → 排除（缺失取假是过滤信号的既定契约）。"""
    index = pd.bdate_range("2024-01-02", periods=2)
    columns = ["known", "unknown"]
    frames = growth_frames(
        index,
        columns,
        pe=[[1.0, 1.0]] * 2,
        pe_percentile=[[0.01, 0.01]] * 2,
        peg=[[0.3, 0.3]] * 2,
        drawdown_1y=[[0.9, 0.9]] * 2,
        market_cap=[[5e10, float("nan")]] * 2,
    )
    from mbt.data.panel import Panel

    got = undervalued_growth_screen(top_n=5).apply(Panel(frames))

    assert got.candidates(index[0]) == ["known"]


def test_the_growth_screen_requires_a_market_cap_above_the_threshold():
    """市值必须**严格大于** 100 亿：恰好 100 亿不算，100 亿零 1 元才算。"""
    index = pd.bdate_range("2024-01-02", periods=2)
    columns = ["small", "edge", "big"]
    frames = growth_frames(
        index,
        columns,
        pe=[[1.0, 1.0, 1.0]] * 2,
        pe_percentile=[[0.01, 0.01, 0.01]] * 2,
        peg=[[0.3, 0.3, 0.3]] * 2,
        drawdown_1y=[[0.9, 0.9, 0.9]] * 2,
        market_cap=[[9e9, 1e10, 1e10 + 1]] * 2,
    )
    from mbt.data.panel import Panel

    got = undervalued_growth_screen(top_n=5).apply(Panel(frames))

    assert got.candidates(index[0]) == ["big"]


def test_drawdown_fields_computes_from_the_full_history(make_market, make_prices):
    """``drawdown_fields`` 由行情算出跌幅：窗口内最高价 12、末根收盘 6 → 0.5。"""
    highs = [10.0, 12.0, 11.0, 9.0, 6.0]
    prices = make_prices(highs)
    prices["high"] = highs  # make_prices 令 o=h=l=c，这里只要 high 与 close 不同

    fields = drawdown_fields([make_market("sh600000", prices)], lookback=5)

    got = fields[DRAWDOWN_FIELD]["sh600000"]
    assert pd.isna(got.iloc[3]), "不足 5 根处为缺失"
    assert got.iloc[-1] == pytest.approx(1.0 - 6.0 / 12.0)


def test_drawdown_fields_rejects_an_empty_universe():
    with pytest.raises(ValueError, match="至少要有一个标的"):
        drawdown_fields([])


# --- 选股规则：买入条件（旧版，留作对照） --------------------------------------


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
    from examples.strategies import UndervaluedGrowth

    markets = [make_market("sh600000", make_prices([10.0] * 8))]
    index = markets[0].prices.index
    signals = valuation_frames(index, ["sh600000"], **overrides)

    return run_portfolio_backtest(
        markets,
        UndervaluedGrowth,
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
    from examples.strategies import UndervaluedGrowth

    markets = [make_market("sh600000", make_prices([10.0] * 8))]
    index = markets[0].prices.index
    signals = valuation_frames(index, ["sh600000"])

    result = run_portfolio_backtest(
        markets,
        UndervaluedGrowth,
        cash=100_000.0,
        max_positions=1,
        rules=zero_cost_rules,
        universe_rules=UniverseRules(min_trading_days=0),
        # 一个什么都不选的闸门
        screen=Screen(filters=(lambda panel: panel["close"] < 0.0,)),
        signals=signals,
    )

    assert len(result.trades) == 0
