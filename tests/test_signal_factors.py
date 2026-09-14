"""排序因子：同一时刻可横向比较大小的数值（``CONTEXT.md``），票据 #5。

因子的用处与指标不同：不是「某标的的序列」，而是「同一行里谁更靠前」。故这里的断言除了
数值本身，还要**在同一行内比较两个标的**——那才是它的消费方向。
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from mbt.signals import (
    distance_to_high,
    j_oversold,
    kdj,
    momentum,
    reward_risk_ratio,
    white_line,
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


# --- 交易盈亏比：(白线 − 收盘) ÷ (收盘 − 黄线) ---------------------------------


def pullback_frame(symbol_frame, rate, symbol="sh600000", bars=140):
    """**先涨后浅回调**的一段序列：涨到 20，再按 ``rate`` 逐根回落 ``bars - 130`` 根。

    这个形状是刻意的：盈亏比的分母是「收盘 − 黄线」，故要让它为正，收盘必须**仍在黄线上方**，
    而分子要为正又要求收盘**已跌到白线下方**——即价格落在两条线之间。那正是买点所在的位置
    （回调到支撑上），故这个夹具与策略的语义一致，不是为了凑数。

    用 140 根是因为黄线含 114 日均线：短于此它整列缺失（实测 40 根时黄线全是 NaN）。
    """

    closes = list(np.linspace(10.0, 20.0, 130))
    closes += [20.0 * (1.0 - rate * (i + 1)) for i in range(bars - 130)]
    return symbol_frame({symbol: closes[:bars]})


def test_reward_risk_ratio_is_the_reward_over_the_risk(symbol_frame):
    """手算锁定：``(白线 − 收盘) ÷ (收盘 − 黄线)``。

    期望值由 ``white_line`` / ``yellow_line`` **现算**，而不是写死数字——故两条线的口径若被
    改动，这条会跟着变红，不会因为「数字还对得上」而静默通过。
    """
    prices = pullback_frame(symbol_frame, 0.006)
    windows = (14, 28, 57, 114)

    got = reward_risk_ratio(prices, white_n=10, windows=windows)["sh600000"]
    white = white_line(prices, 10)["sh600000"]
    yellow = yellow_line(prices, windows=windows)["sh600000"]

    last = prices.index[-1]
    close = float(prices.loc[last, "sh600000"])
    reward = float(white.loc[last]) - close
    risk = close - float(yellow.loc[last])

    assert risk > 0 and reward > 0, "构造前提：收盘应落在黄线与白线之间"
    assert float(got.loc[last]) == pytest.approx(reward / risk)


def test_reward_risk_ratio_is_missing_when_the_close_is_below_the_yellow_line(symbol_frame):
    """收盘跌破黄线 → 分母 ≤ 0，取**缺失**而不是负数。

    刻意不给负数：那会在横截面排序里排到最末，看着像「最差的一档」；也不给无穷，那看着像
    「最好的一档」。这两种观感都在替调用方表态，而事实是「这个比值没有意义」。
    """
    prices = pullback_frame(symbol_frame, 0.010)
    windows = (14, 28, 57, 114)

    got = reward_risk_ratio(prices, white_n=10, windows=windows)["sh600000"]
    yellow = float(yellow_line(prices, windows=windows)["sh600000"].iloc[-1])

    assert float(prices["sh600000"].iloc[-1]) < yellow, "构造前提：收盘应在黄线下方"
    assert pd.isna(got.iloc[-1])


def test_reward_risk_ratio_goes_negative_when_the_close_is_above_the_white_line(symbol_frame):
    """收盘站上白线 → 分子 ≤ 0，**照常给负值**（那是一个真实可比的比值，由 ``> 门槛`` 挡掉）。

    构造：整段**单调上涨**，故收盘恒在两条线之上、分子恒为负。
    """

    prices = symbol_frame({"sh600000": list(np.linspace(10.0, 20.0, 140))})

    got = reward_risk_ratio(prices, white_n=10, windows=(14, 28, 57, 114))["sh600000"]

    assert pd.notna(got.iloc[-1])
    assert float(got.iloc[-1]) < 0


def test_reward_risk_ratio_is_missing_while_the_yellow_line_is_missing(symbol_frame):
    """黄线窗口不足处为缺失，比值随之缺失——缺失来自窗口，不是来自数据。"""
    short = symbol_frame({"sh600000": [10.0 + i for i in range(40)]})

    got = reward_risk_ratio(short, white_n=10, windows=(14, 28, 57, 114))["sh600000"]
    assert got.isna().all(), "黄线含 114 日均线，40 根里它整列缺失"

    longer = symbol_frame({"sh600000": [10.0 + i for i in range(140)]})
    assert pd.notna(
        reward_risk_ratio(longer, white_n=10, windows=(14, 28, 57, 114))["sh600000"].iloc[-1]
    )


def test_reward_risk_ratio_is_larger_when_the_close_sits_closer_to_the_yellow_line(symbol_frame):
    """越大越靠前：贴黄线的那只比值更大，故它能直接当排序因子用。

    这条把**方向**钉住——若哪天有人把分子分母写反，两者的相对大小会翻转。
    """
    shallow = pullback_frame(symbol_frame, 0.002, symbol="near")
    deep = pullback_frame(symbol_frame, 0.008, symbol="far")
    prices = pd.concat([shallow, deep], axis=1)

    got = reward_risk_ratio(prices, white_n=10, windows=(14, 28, 57, 114)).iloc[-1]

    assert got["far"] > got["near"], "贴黄线（亏头小、赚头大）的那只比值必须更大"
    assert got["far"] > 4.0, "深回调那只应当越过默认门槛 4.0，否则这条测的不是它"


def test_reward_risk_ratio_needs_only_closing_prices(symbol_frame):
    """它只吃收盘价的标的宽表——不再需要面板，也不再需要摆动点。

    这条是接口契约：前一版要 ``panel`` 与 ``anchors``，各自带来一处「必须与别处同源」的
    约束。简化之后那些约束一并消失。
    """
    prices = pullback_frame(symbol_frame, 0.006)

    got = reward_risk_ratio(prices, white_n=10, windows=(14, 28, 57, 114))

    assert list(got.columns) == ["sh600000"]
    assert got.index.equals(prices.index)
    assert all(dtype.kind == "f" for dtype in got.dtypes)


def test_reward_risk_ratio_rejects_an_empty_window_list(symbol_frame):
    """没有均线就谈不上黄线，故报错而不是静默给常数。"""
    prices = symbol_frame({"sh600000": [10.0, 10.0, 10.0]})

    with pytest.raises(ValueError, match="不能为空"):
        reward_risk_ratio(prices, white_n=2, windows=())


# --- J 值的超卖程度（B1 的排序因子） -------------------------------------------


def test_j_oversold_is_the_negated_j(panel):
    """``−J``：J 越小越超卖，取负号之后「越大越超卖」，才符合因子「越大越靠前」的约定。"""
    closes = [10.0, 10.0, 11.0, 10.0, 12.0]
    bars = panel(
        {
            "high": {"sh600000": [c + 1.0 for c in closes]},
            "low": {"sh600000": [c - 1.0 for c in closes]},
            "close": {"sh600000": closes},
        }
    )

    got = j_oversold(bars, 3, 3, 3)["sh600000"]
    j = kdj(bars, 3, 3, 3).j["sh600000"]

    pd.testing.assert_series_equal(got, -j, check_names=False)


def test_j_oversold_puts_the_more_oversold_symbol_first(panel):
    """横截面上的方向：J 更低的那只因子更大。

    两个标的走同一段行情，只在最后一根分开——一只继续跌（把 J 压低）、一只继续涨（把 J 抬高）。
    """
    falling = [10.0, 11.0, 12.0, 11.0, 10.0]
    rising = [10.0, 11.0, 12.0, 11.0, 13.0]
    bars = panel(
        {
            "high": {"falling": [c + 1.0 for c in falling], "rising": [c + 1.0 for c in rising]},
            "low": {"falling": [c - 1.0 for c in falling], "rising": [c - 1.0 for c in rising]},
            "close": {"falling": falling, "rising": rising},
        }
    )

    got = j_oversold(bars, 3, 3, 3).iloc[-1]

    assert got["falling"] > got["rising"], "J 更低的那只必须排更前"


def test_j_oversold_is_missing_where_the_j_is_missing(panel):
    """J 缺失处（窗口不足、分母为 0）因子随之缺失——不会因为取负号而变成 0。"""
    flat = panel(
        {
            "high": {"sh600000": [5.0, 5.0, 5.0, 5.0]},
            "low": {"sh600000": [5.0, 5.0, 5.0, 5.0]},
            "close": {"sh600000": [5.0, 5.0, 5.0, 5.0]},
        }
    )

    got = j_oversold(flat, 3, 3, 3)["sh600000"]

    assert got.isna().all()
