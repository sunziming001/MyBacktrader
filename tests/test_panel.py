"""行情面板：**字段 → 标的宽表**的映射（ADR-0009，票据 #14）。

面板要挡住的是**静默错位**：跨字段计算（ATR 要用 high/low/close 同步对齐）的全部正确性
都建立在各字段同日对齐上，而一旦不对齐，结果是错的数而非报错。故这里的重点在
「错位必须报错」与「只带显式声明的字段」两条。
"""

from __future__ import annotations

import pandas as pd
import pytest

from mbt.data import MarketDataError, Panel, assemble_panel
from mbt.data.panel import coverage, latest_complete_day


def test_assemble_panel_uses_symbols_as_columns_and_keeps_their_order(make_prices, make_market):
    """面板的列是**标的**、行是交易日；标的顺序按传入顺序保留。"""
    a = make_market("sh600000", make_prices([10.0, 11.0]))
    b = make_market("sz000001", make_prices([20.0, 21.0]))

    got = assemble_panel([a, b], ["close"])

    assert got.field_names == ("close",)
    assert list(got.symbols) == ["sh600000", "sz000001"]
    assert got["close"]["sh600000"].tolist() == [10.0, 11.0]
    assert got["close"]["sz000001"].tolist() == [20.0, 21.0]


def test_only_declared_fields_are_carried_never_inferred(make_prices, make_market):
    """字段是**声明**的意图，不是数据源的副产品：没声明的字段不进来。

    ``make_prices`` 造出的字段宽表含 ``volume`` 与 ``amount``，两者都不该被无声带进计算。
    """
    market = make_market("sh600000", make_prices([10.0, 11.0]))

    got = assemble_panel([market], ["high", "low", "close"])

    assert got.field_names == ("high", "low", "close")
    assert "amount" not in got
    assert "volume" not in got


def test_a_field_missing_on_a_symbol_is_an_error_naming_both(make_prices, make_market):
    """缺字段要报错并指出是哪个标的缺哪个字段，而不是静默少一列。"""
    frame = make_prices([10.0, 11.0]).drop(columns=["high"])
    market = make_market("sh600000", frame)

    with pytest.raises(MarketDataError, match="sh600000 的行情缺少字段 'high'"):
        assemble_panel([market], ["high"])


def test_symbols_with_different_trading_days_align_to_the_union_without_filling(
    make_prices, make_market
):
    """各标的的交易日可以不同（停牌、次新股）：按并集对齐，缺处保持缺失。

    这不是填充——缺失仍是缺失，只是换了个位置表达（ADR-0005 的缺口纪律）。
    """
    a = make_market("sh600000", make_prices([10.0, 11.0], start="2024-01-02"))
    b = make_market("sz000001", make_prices([20.0], start="2024-01-03"))

    got = assemble_panel([a, b], ["close"])["close"]

    assert got.shape == (2, 2)
    assert got.loc["2024-01-02", "sh600000"] == pytest.approx(10.0)
    assert pd.isna(got.loc["2024-01-02", "sz000001"])
    assert got.loc["2024-01-03", "sz000001"] == pytest.approx(20.0)


def test_a_duplicated_symbol_is_rejected_rather_than_silently_collapsing(make_prices, make_market):
    """同一标的装两次若不拦，组装时会把后一份**悄悄覆盖**前一份——静默丢数据。

    拦它的是 ``Panel`` 的列唯一性校验（组装器用 ``pd.concat`` 保留重复，而不是用字典）。
    """
    a = make_market("sh600000", make_prices([10.0]))
    b = make_market("sh600000", make_prices([99.0]))

    with pytest.raises(MarketDataError, match="columns 有重复"):
        assemble_panel([a, b], ["close"])


def test_empty_markets_or_empty_fields_are_rejected(make_prices, make_market):
    """空面板没有意义，且多半是上游出了错，故报错而不是返回一个空壳。"""
    market = make_market("sh600000", make_prices([10.0]))

    with pytest.raises(MarketDataError, match="至少要有一个标的"):
        assemble_panel([], ["close"])
    with pytest.raises(MarketDataError, match="至少要声明一个字段"):
        assemble_panel([market], [])


def test_fields_with_divergent_indexes_are_rejected():
    """字段间的 index 错位必须报错——跨字段计算据此对齐，错位即静默错答。"""
    left = pd.DataFrame({"sh600000": [1.0, 2.0]}, index=pd.bdate_range("2024-01-02", periods=2))
    right = pd.DataFrame({"sh600000": [1.0, 2.0]}, index=pd.bdate_range("2024-01-03", periods=2))

    with pytest.raises(MarketDataError, match="index 不一致"):
        Panel({"high": left, "low": right})


def test_fields_with_divergent_columns_are_rejected():
    """字段间的 columns 错位（标的集合不同）同样必须报错。"""
    index = pd.bdate_range("2024-01-02", periods=2)
    left = pd.DataFrame({"sh600000": [1.0, 2.0]}, index=index)
    right = pd.DataFrame({"sz000001": [1.0, 2.0]}, index=index)

    with pytest.raises(MarketDataError, match="columns 不一致"):
        Panel({"high": left, "low": right})


def test_a_duplicated_column_inside_a_field_is_rejected():
    """手工构造的面板里若同一标的出现两次，截面统计会被重复计入。"""
    index = pd.bdate_range("2024-01-02", periods=2)
    frame = pd.DataFrame([[1.0, 1.0], [2.0, 2.0]], index=index, columns=["sh600000", "sh600000"])

    with pytest.raises(MarketDataError, match="columns 有重复"):
        Panel({"close": frame})


def test_a_non_frame_field_is_rejected():
    """字段值必须是标的宽表；传进去别的东西要立刻报错，而不是等到计算时才炸。"""
    with pytest.raises(MarketDataError, match="必须是 DataFrame"):
        Panel({"close": [1.0, 2.0]})


def test_the_panel_is_read_only(panel):
    """面板一旦建成即不可改——字段映射只读，避免调用方在背后换掉一张表。"""
    got = panel({"close": {"sh600000": [1.0, 2.0]}})

    with pytest.raises(TypeError):
        got.fields["close"] = None


def test_asking_for_an_undeclared_field_lists_what_is_declared(panel):
    """取未声明的字段要报错并列出已声明的——比裸 KeyError 好查。"""
    got = panel({"close": {"sh600000": [1.0, 2.0]}})

    assert "close" in got
    assert "high" not in got
    with pytest.raises(KeyError, match="面板没有字段 'high'"):
        got["high"]


# --- 选股该按哪一天评估 -------------------------------------------------------
#
# **末根残桩**：本机真实数据的最后一根 K 线（2026-09-11）只有 **5/4,752** 只标的有价，
# 而它前面每一天都是约 4,718 只。这不是偶发——它是「通达信还没把当天数据写全」的常态。
# 一个每日选股的工具若按「最后一根」评估，就会静静地选出一个 5 只标的的日子，
# 而结果看起来只是「今天候选很少」。故「哪一天算数」必须由数据本身回答。


def _coverage_panel(counts: list[int], *, width: int = 100):
    """造一个「第 ``i`` 行只有 ``counts[i]`` 只标的有价」的面板（其余为缺失）。"""
    dates = pd.bdate_range("2024-01-01", periods=len(counts))
    columns = [f"sh{600000 + i:06d}" for i in range(width)]
    data = [[1.0] * count + [float("nan")] * (width - count) for count in counts]
    return Panel({"close": pd.DataFrame(data, index=dates, columns=columns)})


def test_coverage_counts_the_symbols_with_a_price_per_day():
    got = coverage(_coverage_panel([3, 1, 2]))

    assert got.tolist() == [3, 1, 2]
    assert got.index.equals(pd.bdate_range("2024-01-01", periods=3))


def test_the_trailing_stub_bar_is_not_taken_as_the_evaluation_day():
    """尾巴上那根只有少数标的有价的 K 线要**跳过去**——这正是本机数据现在的样子。"""
    panel_ = _coverage_panel([100] * 10 + [5])

    got = latest_complete_day(panel_)

    assert got == pd.bdate_range("2024-01-01", periods=10)[-1], "应回退到残桩之前那天"


def test_an_ordinary_tail_is_kept_as_is():
    """数据是全的时候不许乱回退——否则这条规则本身就会变成一个新坑。"""
    panel_ = _coverage_panel([100] * 10)

    assert latest_complete_day(panel_) == pd.bdate_range("2024-01-01", periods=10)[-1]


def test_it_returns_the_last_complete_day_not_the_first():
    """回退要走**尽可能少**的步数：连着几天残缺时取最晚的那个合格日。"""
    panel_ = _coverage_panel([100] * 5 + [7, 9, 4])

    assert latest_complete_day(panel_) == pd.bdate_range("2024-01-01", periods=5)[-1]


def test_a_day_with_half_the_names_is_still_a_trading_day():
    """门槛是**宽的**：正常交易日不会有一半标的缺席，故半个市场停牌不该触发回退。"""
    panel_ = _coverage_panel([100] * 5 + [60])

    assert latest_complete_day(panel_) == pd.bdate_range("2024-01-01", periods=6)[-1]


def test_the_threshold_is_relative_to_the_panel_so_a_small_universe_still_works():
    """门槛按**同一份面板的中位数**算，故 ``--limit 5`` 那种小样本照样给出最后一天。

    绝对的「至少 N 只」在这里一定是错的：试跑时全市场只有几只，而「只有几只」不等于
    「数据没写全」。故这条同时也钉住了**不许**把门槛写成常数。
    """
    small = _coverage_panel([5, 5, 5], width=5)

    assert latest_complete_day(small) == pd.bdate_range("2024-01-01", periods=3)[-1]


def test_a_day_is_never_invented_when_there_is_no_data_at_all():
    """空面板没有「最后一天」可言——返回 ``None``，让调用方自己决定怎么收场。"""
    empty = Panel({"close": pd.DataFrame(index=pd.DatetimeIndex([]), dtype="float64")})

    assert coverage(empty).empty
    assert latest_complete_day(empty) is None


def test_asking_for_a_field_the_panel_lacks_names_what_is_declared():
    """缺字段要报错并说清该怎么办——与 ``Panel.__getitem__`` 同口径。"""
    got = _coverage_panel([1, 2])

    with pytest.raises(KeyError, match="面板没有字段 'volume'"):
        latest_complete_day(got, field="volume")
