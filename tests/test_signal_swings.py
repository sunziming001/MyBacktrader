"""摆动点：已确认拐点的位置、配对关系、以及**时点正确性**。

本文件的重点是最后一条。拐点定义天然带着前视风险——「这是峰值」这件事只有回头看才知道，
所以「在峰值当根就报出峰值」是一个看起来对、实际用了未来的实现。故这里既手算钉住数值，
也逐根检查「报出来的拐点在当时是否已经可确认」。
"""

from __future__ import annotations

import pandas as pd
import pytest

from mbt.signals import Swings, swings


def swing_frame(symbol_frame):
    """标准样本：跌到 8 → 涨到 12 → 回撤超过 10% 确认峰值 12。

    故自索引 7 起，已完成的上涨是 8 → 12，``peak_age`` 为 1、``trough_age`` 为 5。
    """
    return symbol_frame({"sh600000": [10.0, 9.0, 8.0, 9.0, 10.0, 11.0, 12.0, 10.5]})


def test_swings_reports_a_peak_only_after_the_retracement_confirms_it(symbol_frame):
    """涨幅不够回撤阈值时，峰值**尚未确认**，故整列缺失。

    样本一路上涨到 12（回撤阈值 10%）：12 的 10% 回撤要跌破 10.8 才算确认，而这组数据
    从未跌破。故即便序列里显然有个高点，本函数也**不报**——这就是「不给乐观答案」。
    """
    prices = symbol_frame({"sh600000": [8.0, 9.0, 10.0, 11.0, 12.0]})

    got = swings(prices, retracement=0.10)

    assert got.peak_price["sh600000"].isna().all()
    assert got.trough_price["sh600000"].isna().all()


def test_swings_reports_the_confirmed_advance_once_the_retracement_happens(symbol_frame):
    """补上一根 10.5 之后峰值被确认，且报出的是**与它配对的起涨点**。

    价格路径 ``[10, 9, 8, 9, 10, 11, 12, 10.5]``（阈值 10%）：
    跌到 8 后涨过 8.8 确认低点 8；随后连涨到 12；10.5 < 12 × 0.9 = 10.8 确认峰值 12。
    故 ``peak_price`` 自第 8 根（索引 7）起为 12、``peak_age`` 为 1；
    ``trough_price`` 为 8、``trough_age`` 为 5（8 在第 3 根，索引 2）。
    """
    got = swings(swing_frame(symbol_frame), retracement=0.10)

    peak = got.peak_price["sh600000"]
    trough = got.trough_price["sh600000"]

    assert peak.iloc[:7].isna().all(), "峰值在确认之前就被报出来了——那是前视偏差"
    assert peak.iloc[7] == pytest.approx(12.0)
    assert got.peak_age["sh600000"].iloc[7] == pytest.approx(1.0)

    assert trough.iloc[7] == pytest.approx(8.0)
    assert got.trough_age["sh600000"].iloc[7] == pytest.approx(5.0)


def test_swings_pairs_the_trough_with_the_peak_not_with_the_newest_low(symbol_frame):
    """``trough_*`` 是**与当前峰值配对**的起涨点，不是「最近确认的低点」。

    峰值 12 确认后继续跌，跌破 8 的路径上又会确认一个新低点（如 6）。那个新低点描述的是
    **正在形成**的下一段上涨，与「已完成的那一段」不是一回事；若它把 ``trough_price``
    顶掉，判据里的「上涨幅度」会被算成 12/6−1 = 100% 而不是 12/8−1 = 50%。

    样本：``[10, 9, 8, 9, 10, 11, 12, 10.5, 9, 7, 6, 7.5]``（阈值 10%）。
    峰值 12 于索引 7 确认；其后跌到 6（索引 10），并从 6 涨过 6.6（索引 11 的 7.5）确认新低点 6。
    """
    prices = symbol_frame(
        {"sh600000": [10.0, 9.0, 8.0, 9.0, 10.0, 11.0, 12.0, 10.5, 9.0, 7.0, 6.0, 7.5]}
    )

    got = swings(prices, retracement=0.10)

    # 新低点 6 在索引 11 被确认，但它**没有**顶掉已配对的那个起涨点。
    assert got.trough_price["sh600000"].iloc[11] == pytest.approx(8.0)
    assert got.trough_age["sh600000"].iloc[11] == pytest.approx(9.0)
    assert got.peak_price["sh600000"].iloc[11] == pytest.approx(12.0)


def test_swings_never_reports_a_half_pair(symbol_frame):
    """``trough_*`` 与 ``peak_*`` 同时缺失或同时有值——不存在半截状态。

    后半段样本 ``[10, 9, 8, 9]`` 是只有低点被确认、峰值还没出现的情形（跌到 8 后涨过 8.8）。
    若那时把低点单独报出来，调用方就得自己猜这段算不算数。
    """
    only_trough = symbol_frame({"sh600000": [10.0, 9.0, 8.0, 9.0]})
    got = swings(only_trough, retracement=0.10)
    assert got.trough_price["sh600000"].isna().all(), "低点被单独报出来了"

    both = swing_frame(symbol_frame)
    got = swings(both, retracement=0.10)
    for column in both.columns:
        assert got.peak_price[column].isna().equals(got.trough_price[column].isna())
        assert got.peak_age[column].isna().equals(got.trough_age[column].isna())


def test_swings_does_not_report_an_advance_whose_start_was_never_confirmed(symbol_frame):
    """价格自序列首根直线上涨时，这一段的起点从未被确认为低点，故整段不报。

    宁可漏报也不猜起点：若把序列首根当成起涨点，算出的「上涨幅度」会随切片起点变化，
    于是同一段上涨在两个不同的取数区间里得到两个值——那正是本项目反复防的不可复现。
    """
    prices = symbol_frame({"sh600000": [10.0, 11.0, 12.0, 13.0, 14.0, 12.5]})

    got = swings(prices, retracement=0.10)

    assert got.peak_price["sh600000"].isna().all()


def test_swings_rejects_a_retracement_outside_the_open_unit_interval(symbol_frame):
    """阈值必须落在 ``(0, 1)``：0 会让每个点都被当成拐点，1 及以上则永远不确认。"""
    prices = symbol_frame({"sh600000": [10.0, 9.0, 8.0]})

    for bad in (0.0, 1.0, -0.1, 1.5):
        with pytest.raises(ValueError, match=r"\(0, 1\)"):
            swings(prices, retracement=bad)


def test_swings_keeps_the_trailing_window_number_of_bars_correct(symbol_frame):
    """``age`` 是自拐点起经过的**根数**，随每根递增，且不在缺口处跳号。

    涨到 12 后回落，峰值在索引 6 确认；其后每一根 age 应连续 +1（缺口那根也占一个序号，
    因为停牌日同样是日历上过去的一天）。
    """
    nan = float("nan")
    prices = symbol_frame(
        {"sh600000": [10.0, 9.0, 8.0, 9.0, 10.0, 11.0, 12.0, 10.5, nan, 10.0, 9.5]}
    )

    ages = swings(prices, retracement=0.10).peak_age["sh600000"]

    assert ages.iloc[7] == pytest.approx(1.0)
    assert ages.iloc[9] == pytest.approx(3.0)
    assert ages.iloc[10] == pytest.approx(4.0)


def test_swings_reports_nothing_on_a_bar_whose_price_is_unknown(symbol_frame):
    """缺口当根的价未知，故当根拒绝给出「距峰值多少根」这类判断——一律缺失。

    但结构**不**被停牌抹掉：缺口之后仍报出同一个峰值。这与 :func:`~mbt.signals.ema`
    的处理不同，理由是拐点结构描述的是**过去的**价格，而过去的价格仍然已知。
    """
    nan = float("nan")
    prices = symbol_frame({"sh600000": [10.0, 9.0, 8.0, 9.0, 10.0, 11.0, 12.0, 10.5, nan]})

    got = swings(prices, retracement=0.10)

    assert pd.isna(got.peak_price["sh600000"].iloc[8])
    assert pd.isna(got.peak_age["sh600000"].iloc[8])
    assert got.peak_price["sh600000"].iloc[7] == pytest.approx(12.0)


def test_swings_keeps_the_symbol_frame_shape_and_float_dtype(symbol_frame):
    """四条输出各是合法的标的宽表：列、索引、dtype 与输入一致（ADR-0009）。"""
    prices = symbol_frame(
        {
            "sh600000": [10.0, 9.0, 8.0, 9.0, 10.0, 11.0, 12.0, 10.5],
            "sz000001": [20.0, 18.0, 16.0, 18.0, 20.0, 22.0, 24.0, 21.0],
        }
    )

    got = swings(prices, retracement=0.10)

    assert isinstance(got, Swings)
    for series in (got.peak_price, got.peak_age, got.trough_price, got.trough_age):
        assert list(series.columns) == ["sh600000", "sz000001"]
        assert series.index.equals(prices.index)
        assert all(dtype.kind == "f" for dtype in series.dtypes)


def test_swings_computes_each_symbol_independently(symbol_frame):
    """一个标的的拐点不得影响另一个——状态是按列各自持有的。

    两个标的都用「先跌后涨再回撤」的完整形状，但幅度不同：sh600000 的低点 8 / 峰值 12，
    sz000001 的低点 12 / 峰值 16。若状态在列之间串了，两个峰值会互相污染。
    """
    prices = symbol_frame(
        {
            "sh600000": [10.0, 9.0, 8.0, 9.0, 10.0, 11.0, 12.0, 10.5],
            "sz000001": [14.0, 13.0, 12.0, 13.0, 14.0, 15.0, 16.0, 14.0],
        }
    )

    got = swings(prices, retracement=0.10)

    assert got.peak_price["sh600000"].iloc[7] == pytest.approx(12.0)
    assert got.trough_price["sh600000"].iloc[7] == pytest.approx(8.0)
    assert got.peak_price["sz000001"].iloc[7] == pytest.approx(16.0)
    assert got.trough_price["sz000001"].iloc[7] == pytest.approx(12.0)


def test_swings_rejects_a_non_monotonic_index_like_the_other_signals_do(symbol_frame):
    """契约由所有公开函数共同遵守，顺序扫描的也不例外。"""
    idx = pd.to_datetime(["2024-01-03", "2024-01-02", "2024-01-04"])
    prices = pd.DataFrame({"sh600000": [10.0, 11.0, 12.0]}, index=idx)

    with pytest.raises(ValueError, match="升序"):
        swings(prices, retracement=0.10)


def test_swings_of_an_all_missing_series_is_all_missing(symbol_frame):
    """全为缺失值的序列不产生拐点，也不报错。"""
    nan = float("nan")
    prices = symbol_frame({"sh600000": [nan, nan, nan]})

    got = swings(prices, retracement=0.10)

    assert got.peak_price["sh600000"].isna().all()
    assert got.peak_age["sh600000"].isna().all()
