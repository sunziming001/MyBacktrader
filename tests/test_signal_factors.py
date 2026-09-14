"""排序因子：同一时刻可横向比较大小的数值（``CONTEXT.md``），票据 #5。

因子的用处与指标不同：不是「某标的的序列」，而是「同一行里谁更靠前」。故这里的断言除了
数值本身，还要**在同一行内比较两个标的**——那才是它的消费方向。
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from mbt.signals import (
    Swings,
    distance_to_high,
    momentum,
    reward_risk_ratio,
    swings,
    yellow_line,
    yellow_proximity,
)


def test_momentum_is_the_return_over_the_trailing_n_bars(symbol_frame):
    """n 日动量 = 当根 / n 根前 − 1，手算锁定：12/10−1 与 13/11−1。"""
    prices = symbol_frame({"sh600000": [10.0, 11.0, 12.0, 13.0]})

    got = momentum(prices, n=2)["sh600000"]

    assert pd.isna(got.iloc[0])
    assert pd.isna(got.iloc[1])
    assert got.iloc[2] == pytest.approx(0.2)
    assert got.iloc[3] == pytest.approx(13.0 / 11.0 - 1.0)


def test_distance_to_high_is_zero_at_a_new_high_and_negative_below_it(symbol_frame):
    """距 N 日高点的距离：恰在最高收盘价处为 0，回落则为负——方向是「越大越强」。"""
    prices = symbol_frame({"sh600000": [10.0, 12.0, 11.0]})

    got = distance_to_high(prices, n=2)["sh600000"]

    assert pd.isna(got.iloc[0])
    assert got.iloc[1] == pytest.approx(0.0)
    assert got.iloc[2] == pytest.approx(11.0 / 12.0 - 1.0)


def test_factors_are_comparable_across_symbols_on_the_same_row(symbol_frame):
    """同一行的两个数值可直接比大小——这正是排序因子存在的理由。"""
    prices = symbol_frame({"sh600000": [10.0, 20.0], "sz000001": [10.0, 15.0]})

    got = momentum(prices, n=1)

    assert got.loc[got.index[1], "sh600000"] > got.loc[got.index[1], "sz000001"]


def test_factors_keep_missing_values_missing_rather_than_filling_zero(symbol_frame):
    """缺失不填零：填了就会在截面排序里占到一个具体位置，那是凭空造出的排名。"""
    nan = float("nan")
    prices = symbol_frame({"sh600000": [10.0, nan, nan]})

    got = momentum(prices, n=1)

    assert got["sh600000"].isna().all()


def test_factors_are_floats_not_booleans(symbol_frame):
    """因子与过滤信号的返回类型是契约的一部分，不可混用（过滤信号见同层测试）。"""
    prices = symbol_frame({"sh600000": [10.0, 11.0, 12.0]})

    got = momentum(prices, n=1)

    assert all(dtype.kind == "f" for dtype in got.dtypes)


# --- 黄线贴近度（B1 策略的排序因子） --------------------------------------------


def test_yellow_proximity_is_larger_when_the_close_is_closer_to_the_line(symbol_frame):
    """``黄线 ÷ |收盘 − 黄线|``：离黄线越近值越大。

    取一条滞后的黄线（MA3）与两组收盘，比较同一个观测日上的取值——**横截面**上的排序正是
    这个因子的用途，故按「谁大谁靠前」来断言。
    """
    index = pd.bdate_range("2024-01-02", periods=5)
    near = pd.DataFrame({"a": [10.0, 11.0, 12.0, 12.0, 12.02]}, index=index)
    far = pd.DataFrame({"b": [10.0, 11.0, 12.0, 12.0, 14.0]}, index=index)
    prices = pd.concat([near, far], axis=1)

    got = yellow_proximity(prices, windows=(3,))
    lines = yellow_line(prices, windows=(3,))

    last = got.iloc[-1]
    assert last["a"] > last["b"], "贴着黄线的那只必须排更前"
    # 手算 a：黄线 = MA3 = mean(12, 12, 12.02) = 12.006666…，|12.02 − 12.006666| = 0.0133333…
    expected_a = float(lines.loc[index[-1], "a"]) / abs(12.02 - float(lines.loc[index[-1], "a"]))
    assert last["a"] == pytest.approx(expected_a)


def test_yellow_proximity_is_infinite_when_the_close_sits_exactly_on_the_line(symbol_frame):
    """收盘**恰好**等于黄线时分母为 0 → **正无穷**。

    这不是边界瑕疵，而是定义的结果：它是「最近的」那一档，故必须排在所有有限值之前。
    与 ``volume_ratio`` 在基准为 0 时返回无穷同一先例——消费方按大小排序即可，无需特判。
    """
    prices = symbol_frame({"sh600000": [10.0, 10.0, 10.0, 10.0]})

    got = yellow_proximity(prices, windows=(2,))["sh600000"]

    assert got.iloc[-1] == float("inf")


def test_yellow_proximity_is_missing_while_the_line_is_missing(symbol_frame):
    """黄线窗口不足处为缺失，因子随之缺失——``Screen`` 会把缺失的标的排除在取前 N 之外。"""
    prices = symbol_frame({"sh600000": [10.0, 11.0, 12.0, 13.0]})

    got = yellow_proximity(prices, windows=(3,))["sh600000"]

    assert got.iloc[:2].isna().all()
    assert pd.notna(got.iloc[2])
    assert pd.notna(got.iloc[3])


def test_yellow_proximity_takes_the_absolute_distance_so_a_break_below_stays_positive(
    symbol_frame,
):
    """分母取绝对值：**跌破**黄线时因子仍为正，不会变成负数。

    构造对称的两组（最后一根分别在黄线上下 0.1）。两组的黄线**不相等**（MA2 = 10.1 与 9.9，
    因为最后一根本身不同），故因子也不相等——它们度量的是**相对**贴近度，这一点由
    :func:`test_yellow_proximity_is_not_scale_dependent` 另行钉住。

    这条真正要钉的是**符号**：若分母漏了 ``abs()``，下面那一组会得到负数，从而在横截面排序
    里被排到最后——而「跌破黄线但只差 0.1」本该是最靠前的一档。
    """
    index = pd.bdate_range("2024-01-02", periods=4)
    above = pd.DataFrame({"a": [10.0, 10.0, 10.0, 10.2]}, index=index)
    below = pd.DataFrame({"b": [10.0, 10.0, 10.0, 9.8]}, index=index)
    prices = pd.concat([above, below], axis=1)

    got = yellow_proximity(prices, windows=(2,)).iloc[-1]
    lines = yellow_line(prices, windows=(2,)).iloc[-1]

    assert got["b"] > 0, "跌破黄线却得到负因子——分母漏了 abs()"
    assert got["b"] == pytest.approx(float(lines["b"]) / 0.1)
    assert got["a"] == pytest.approx(float(lines["a"]) / 0.1)


def test_yellow_proximity_is_not_scale_dependent(symbol_frame):
    """价格整体乘以常数，因子**不变**——它是两个同量纲量的比值。

    这条有实际意义：后复权把整条序列放大，而基于该因子的排序不受影响。
    """
    base = symbol_frame({"sh600000": [10.0, 11.0, 12.0, 11.5, 11.2]})

    plain = yellow_proximity(base, windows=(2,))
    scaled = yellow_proximity(base * 7.5, windows=(2,))

    pd.testing.assert_frame_equal(plain, scaled, rtol=1e-12)


def test_yellow_proximity_rejects_a_non_monotonic_index(symbol_frame):
    """契约由所有公开函数共同遵守，因子也不例外。"""
    idx = pd.to_datetime(["2024-01-03", "2024-01-02", "2024-01-04"])
    prices = pd.DataFrame({"sh600000": [10.0, 11.0, 12.0]}, index=idx)

    with pytest.raises(ValueError, match="升序"):
        yellow_proximity(prices, windows=(2,))


# --- 交易盈亏比（B1 策略的过滤条件与排序因子） -----------------------------------

NAN = float("nan")


def anchors_of(symbol_frame, *, peak_price, peak_age):
    """只造本函数读到的两条（``peak_price`` / ``peak_age``），``trough_*`` 填同形占位。

    这样用例可以直接**摆出**段边界，把「盈亏比怎么算」与「拐点怎么认」分成两件事测——
    后者归 ``test_signal_swings.py``。本函数不读 ``trough_*``，故占位不会掩盖任何东西。
    """
    prices = symbol_frame(peak_price)
    ages = symbol_frame(peak_age)
    return Swings(peak_price=prices, peak_age=ages, trough_price=prices, trough_age=ages)


def test_the_yellow_line_is_the_stop_when_it_sits_above_the_prior_low(panel, symbol_frame):
    """黄线比前低 × 0.99 高时，止损位就是黄线——手算 **6.0**。

    收盘 12.6；黄线 = MA3 = (12 + 13 + 12.6) ÷ 3 = 12.5333…；前低 = min(12.0, 12.4) × 0.99
    = 11.88；故亏头 = 12.6 − 12.5333… = 0.06666…，赚头 = 前高 13 − 12.6 = 0.4 → **6.0**。

    前 4 根为缺失：前 2 根黄线窗口不足，后 2 根峰值尚未确认。
    """
    built = panel(
        {
            "close": {"sh600000": [10.0, 11.0, 12.0, 13.0, 12.6]},
            "low": {"sh600000": [9.0, 10.0, 11.0, 12.0, 12.4]},
        }
    )
    got = reward_risk_ratio(
        built,
        anchors_of(
            symbol_frame,
            peak_price={"sh600000": [NAN] * 4 + [13.0]},
            peak_age={"sh600000": [NAN] * 4 + [1.0]},
        ),
        windows=(3,),
        stop_buffer=0.01,
    )["sh600000"]

    assert got.iloc[:4].isna().all()
    assert got.iloc[-1] == pytest.approx(0.4 / (12.6 - 37.6 / 3))


def test_the_prior_low_is_the_stop_when_it_sits_above_the_yellow_line(panel, symbol_frame):
    """前低 × 0.99 比黄线高时，止损位是前低——手算 **1 ÷ 0.229**。

    收盘 13.0；黄线 = MA5 = (5 + 5 + 12 + 13 + 13) ÷ 5 = 9.6；前低 = min(12.9, 12.9) × 0.99
    = 12.771；故亏头 = 13 − 12.771 = 0.229，赚头 = 前高 14 − 13 = 1 → **1 ÷ 0.229**。

    这段构造顺带说明「前低在上」需要什么：**黄线窗口比前低窗口长得多**。黄线是窗口内收盘的
    均值，而前低只看峰值之后那一段——窗口一短，均值必然高于段内最低价，前低就永远当不上止损位。
    """
    built = panel(
        {
            "close": {"sh600000": [5.0, 5.0, 12.0, 13.0, 13.0]},
            "low": {"sh600000": [4.9, 4.9, 11.9, 12.9, 12.9]},
        }
    )
    got = reward_risk_ratio(
        built,
        anchors_of(
            symbol_frame,
            peak_price={"sh600000": [NAN] * 4 + [14.0]},
            peak_age={"sh600000": [NAN] * 4 + [1.0]},
        ),
        windows=(5,),
        stop_buffer=0.01,
    )["sh600000"]

    assert got.iloc[-1] == pytest.approx(1.0 / (13.0 - 12.9 * 0.99))


def test_the_prior_low_starts_at_the_peak_so_an_earlier_low_is_not_counted(panel, symbol_frame):
    """窗口是「峰值 → 当根」，峰值**之前**的最低价不算——那段属于上一波，不是这次的支撑。

    最低价 1.0 落在峰值（第 2 根）之前。前低 = min(5.0, 6.0, 6.0) × 0.99 = 4.95；若窗口误
    从序列开头算起，前低会变成 0.99、盈亏比从 **2 ÷ 1.05** 变成 2 ÷ 1.5 —— 一眼可辨。
    """
    built = panel(
        {
            "close": {"sh600000": [1.0, 5.0, 6.0, 6.0]},
            "low": {"sh600000": [1.0, 5.0, 6.0, 6.0]},
        }
    )
    got = reward_risk_ratio(
        built,
        anchors_of(
            symbol_frame,
            peak_price={"sh600000": [NAN] * 3 + [8.0]},
            peak_age={"sh600000": [NAN] * 3 + [2.0]},
        ),
        windows=(4,),
        stop_buffer=0.01,
    )["sh600000"]

    assert got.iloc[-1] == pytest.approx(2.0 / (6.0 - 5.0 * 0.99))


def test_a_bar_that_makes_a_new_low_pulls_the_prior_low_down_to_its_own_low(panel, symbol_frame):
    """含当根的代价：当根创出调整期新低时，前低就是**它自己的最低价**，亏头被压到 1% 的收盘价。

    最低价逐日下移 8.9 → 7.9 → 7.0，峰值在第 1 根。前低 = 7.0 × 0.99 = 6.93，亏头 =
    7 − 6.93 = 0.07（恰是 ``stop_buffer`` × 收盘价），赚头 = 9 − 7 = 2 → **2 ÷ 0.07 ≈ 28.6**，
    门槛 4.0 轻松通过。

    这正是「正在跌破支撑的那一根反而容易通过门槛」：当根的最低点成了止损位，若它同时**收在
    最低价附近**，亏头就只剩 ``stop_buffer`` × 收盘价。实测（141 只 × 8,558 个交易日）这类
    格子在放行的 29,667 格里有 2,300 格，其中 1,551 格换成「不含当根」的窗口就直接给不出正
    亏头——两种口径各有代价，故由调用方选（见函数文档的 warning）。
    """
    built = panel(
        {
            "close": {"sh600000": [1.0, 9.0, 8.0, 7.0]},
            "low": {"sh600000": [1.0, 8.9, 7.9, 7.0]},
        }
    )
    got = reward_risk_ratio(
        built,
        anchors_of(
            symbol_frame,
            peak_price={"sh600000": [NAN] * 3 + [9.0]},
            peak_age={"sh600000": [NAN] * 3 + [2.0]},
        ),
        windows=(4,),
        stop_buffer=0.01,
    )["sh600000"]

    assert got.iloc[-1] == pytest.approx(2.0 / 0.07)


def test_the_ratio_remembers_that_it_is_a_factor_so_bigger_comes_first(panel, symbol_frame):
    """同一行可比，且**越大越靠前**。两只标的的收盘与止损位完全相同，只有前高不同。

    ``远`` 的前高更高 → 赚头更大 → 排更前。这正是因子契约（``Screen`` 不提供反转开关）。
    """
    built = panel(
        {
            "close": {"近": [1.0, 9.0, 8.0, 7.0], "远": [1.0, 9.0, 8.0, 7.0]},
            "low": {"近": [1.0, 8.9, 7.9, 7.0], "远": [1.0, 8.9, 7.9, 7.0]},
        }
    )
    got = reward_risk_ratio(
        built,
        anchors_of(
            symbol_frame,
            peak_price={"近": [NAN] * 3 + [9.0], "远": [NAN] * 3 + [14.0]},
            peak_age={"近": [NAN] * 3 + [2.0], "远": [NAN] * 3 + [2.0]},
        ),
        windows=(4,),
        stop_buffer=0.01,
    ).iloc[-1]

    assert got["远"] > got["近"], "赚头更大的那只必须排更前"


def test_a_close_below_the_stop_gives_a_missing_ratio_not_a_negative_one(panel, symbol_frame):
    """分母 ≤ 0（已跌穿止损位）→ **缺失**，既不给负数也不给无穷。

    给负数会在横截面排序里排到最末，看着像「最差的一档」，而它其实是「这笔交易不成立」；
    给无穷则看着像「最好的一档」。两者都是把一个不存在判断编成具体数值。
    """
    built = panel(
        {
            "close": {"sh600000": [10.0, 10.0, 9.0]},
            "low": {"sh600000": [9.5, 9.5, 8.9]},
        }
    )
    got = reward_risk_ratio(
        built,
        anchors_of(
            symbol_frame,
            peak_price={"sh600000": [NAN] * 2 + [12.0]},
            peak_age={"sh600000": [NAN] * 2 + [1.0]},
        ),
        windows=(2,),
        stop_buffer=0.01,
    )["sh600000"]

    assert pd.isna(got.iloc[-1]), "收盘 9.0 已在黄线 9.5 下方，比值没有含义"


def test_an_unconfirmed_peak_gives_a_missing_ratio(panel, symbol_frame):
    """峰值未确认就没有前高，也没有前低——整格缺失，且仍是浮点（因子契约）。"""
    built = panel(
        {
            "close": {"sh600000": [10.0, 11.0, 12.0]},
            "low": {"sh600000": [9.0, 10.0, 11.0]},
        }
    )
    got = reward_risk_ratio(
        built,
        anchors_of(
            symbol_frame,
            peak_price={"sh600000": [NAN, NAN, NAN]},
            peak_age={"sh600000": [NAN, NAN, NAN]},
        ),
        windows=(2,),
        stop_buffer=0.01,
    )

    assert got["sh600000"].isna().all()
    assert all(dtype.kind == "f" for dtype in got.dtypes)


def test_a_gap_inside_the_window_is_skipped_rather_than_blanking_the_window(panel, symbol_frame):
    """窗内的缺失根（停牌）**跳过**：不参与取最小值，也不把整窗判为缺失（ADR-0005）。

    第 3 根缺价。前低取 min(5.0, 〔缺〕, 7.0) = 5.0 → 4.95。若把缺口当成「不知道」而整窗作废，
    这一格会变成缺失；若把缺口当成 0，前低会变成 0。
    """
    built = panel(
        {
            "close": {"sh600000": [1.0, 6.0, 5.0, 7.0]},
            "low": {"sh600000": [1.0, 5.0, NAN, 7.0]},
        }
    )
    got = reward_risk_ratio(
        built,
        anchors_of(
            symbol_frame,
            peak_price={"sh600000": [NAN] * 3 + [9.0]},
            peak_age={"sh600000": [NAN] * 3 + [2.0]},
        ),
        windows=(4,),
        stop_buffer=0.01,
    )["sh600000"]

    assert got.iloc[-1] == pytest.approx(2.0 / (7.0 - 5.0 * 0.99))


def test_the_monotonic_queue_matches_a_naive_scan_over_a_real_swing_series(panel, symbol_frame):
    """拿**朴素写法**（每根重扫整段窗口）对账单调队列——队列是这里唯一有技巧的零件。

    段边界不走占位，而是真的由 :func:`~mbt.signals.swings.swings` 给出，故这条同时验了
    「前低窗口就是 ``peak_age + 1`` 根，且左端是峰值」这件事与信号的因果性一致。
    """
    closes = [
        10.0,
        11.0,
        13.0,
        12.0,
        11.0,
        10.5,
        10.2,
        11.0,
        12.0,
        13.5,
        12.5,
        11.5,
        11.0,
        10.8,
        12.0,
        13.0,
        14.0,
        13.0,
        12.0,
        11.5,
    ]
    lows = [value - 0.1 for value in closes]
    values = {"sh600000": closes}
    frame = symbol_frame(values)
    anchors = swings(frame, retracement=0.05)
    lines = yellow_line(frame, windows=(3,))

    got = reward_risk_ratio(
        panel({"close": values, "low": {"sh600000": lows}}),
        anchors,
        windows=(3,),
        stop_buffer=0.01,
    )

    naive = pd.DataFrame(
        {
            "sh600000": [
                _naive_ratio(row, closes, lows, anchors, lines["sh600000"])
                for row in range(len(closes))
            ]
        },
        index=frame.index,
    )
    pd.testing.assert_frame_equal(got, naive, check_exact=False, rtol=1e-12)


def _naive_ratio(row: int, closes, lows, anchors, line) -> float:
    """单调队列的对照实现：**每根重扫整段窗口**，不做任何优化。"""
    age = anchors.peak_age["sh600000"].iloc[row]
    if not np.isfinite(age):
        return NAN
    window = [value for value in lows[row - int(age) : row + 1] if np.isfinite(value)]
    if not window:
        return NAN
    stop = max(float(line.iloc[row]), min(window) * 0.99)
    risk = closes[row] - stop
    if risk <= 0.0:
        return NAN
    return (float(anchors.peak_price["sh600000"].iloc[row]) - closes[row]) / risk


def test_reward_risk_ratio_rejects_a_stop_buffer_outside_the_unit_interval(panel, symbol_frame):
    """缓冲比例不在 ``[0, 1)`` 内时，止损位要么不降反升、要么非正——报错而不是静默算。"""
    built = panel(
        {
            "close": {"sh600000": [10.0, 11.0]},
            "low": {"sh600000": [9.0, 10.0]},
        }
    )
    anchors = anchors_of(
        symbol_frame,
        peak_price={"sh600000": [NAN, 12.0]},
        peak_age={"sh600000": [NAN, 1.0]},
    )

    for bad in (-0.01, 1.0):
        with pytest.raises(ValueError, match="stop_buffer"):
            reward_risk_ratio(built, anchors, windows=(2,), stop_buffer=bad)


def test_reward_risk_ratio_rejects_anchors_that_do_not_match_the_panel(panel, symbol_frame):
    """段边界与行情必须同源：对不上会把段边界对到别的日子上，而那种错不会报错。"""
    built = panel(
        {
            "close": {"sh600000": [10.0, 11.0]},
            "low": {"sh600000": [9.0, 10.0]},
        }
    )
    elsewhere = anchors_of(
        symbol_frame,
        peak_price={"sz000001": [NAN, 12.0]},
        peak_age={"sz000001": [NAN, 1.0]},
    )

    with pytest.raises(ValueError, match="标的"):
        reward_risk_ratio(built, elsewhere, windows=(2,), stop_buffer=0.01)
