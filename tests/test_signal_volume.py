"""量能结构：按摆动点切段，度量上涨放量与回调缩量。

样本设计成**可手算**：价格路径与成交量都取整数，三个比值都能用纸笔复算，故实现改了哪一步
都会在这里露馅。
"""

from __future__ import annotations

import pandas as pd
import pytest

from mbt.signals import Swings, volume_contraction, volume_structure

# 价格：前 8 根平在 10，随后 10→9→8→9→10→11→12→10.5（回撤阈值 10%）。
# 状态机：index 10 处 8 跌破 10×0.9 确认下跌腿；index 11 处 9 涨过 8×1.1=8.8 确认低点 8；
# 随后连涨到 index 14 的 12；index 15 的 10.5 跌破 12×0.9=10.8 确认峰值 12。
# 故 index 15 处：peak_pos=14（age=1）、trough_pos=10（age=5）。
PRICES = [10.0] * 8 + [10.0, 9.0, 8.0, 9.0, 10.0, 11.0, 12.0, 10.5]

# 成交量：基准 [6,9] 与上涨段 [10,14] 都取 100，只在峰值那根（index 14）放 500，回调那根取 40。
# 于是可手算：基准均值 100、上涨段最大 500、上涨段均值 180、顶部两根均值 300、回调均值 40。
VOLUMES = [100.0] * 16
VOLUMES[14] = 500.0  # 峰值那根
VOLUMES[15] = 40.0  # 回调那根


def series(symbol_frame, values, index=None):
    frame = symbol_frame({"sh600000": values})
    if index is not None:
        frame = frame.reindex(index)
    return frame


def volume_sample(symbol_frame, prices=PRICES, volumes=None):
    """同一批 K 线同时给出价格与成交量两个标的宽表（列名一致，故可直接喂给结构函数）。"""
    price = symbol_frame({"sh600000": prices})
    vol = symbol_frame({"sh600000": volumes if volumes is not None else VOLUMES})
    return price, vol


def test_volume_structure_matches_the_hand_computed_ratios(symbol_frame):
    """三条比值逐一手算锁定。

    基准 = mean([6,9]) = 100；上涨段 = [10,14]，最大 500、均值 (100+100+100+100+500)/5 = 180；
    顶部两根 = [13,14]，均值 (100+500)/2 = 300；回调 = [15]，均值 40。
    故 surge = 500/100 = 5、top = 300/180 = 1.6667、pullback = 40/500 = 0.08。
    """
    price, vol = volume_sample(symbol_frame)
    anchors = swings_of(price)

    got = volume_structure(vol, anchors, edge_bars=2, base_bars=4)

    row = 15
    assert got.surge_ratio["sh600000"].iloc[row] == pytest.approx(5.0)
    assert got.top_ratio["sh600000"].iloc[row] == pytest.approx(300.0 / 180.0)
    assert got.pullback_ratio["sh600000"].iloc[row] == pytest.approx(0.08)


def test_volume_structure_is_missing_until_the_peak_is_confirmed(symbol_frame):
    """峰值确认之前没有段，故三条比值一律缺失——不因为「显然有个高点」就先报出来。"""
    price, vol = volume_sample(symbol_frame)
    anchors = swings_of(price)

    got = volume_structure(vol, anchors, edge_bars=2, base_bars=4)

    for series_ in (got.surge_ratio, got.top_ratio, got.pullback_ratio):
        assert series_["sh600000"].iloc[15:].notna().all()
        assert series_["sh600000"].iloc[:15].isna().all()


def test_volume_structure_pullback_ratio_uses_the_advance_max_not_its_mean(symbol_frame):
    """``pullback_ratio`` 的分母是上涨段的**最大值**，不是均值。

    这是实测定的口径：用均值时分母为 180，比值会是 40/180 = 0.222；用最大值是 40/500 = 0.08。
    这条把两种取法钉开——用均值会让「缩量」在真实数据上完全看不出来。
    """
    price, vol = volume_sample(symbol_frame)
    anchors = swings_of(price)

    ratio = volume_structure(vol, anchors, edge_bars=2, base_bars=4).pullback_ratio["sh600000"]

    assert ratio.iloc[15] == pytest.approx(0.08)
    assert ratio.iloc[15] != pytest.approx(40.0 / 180.0)


def test_volume_structure_is_missing_when_the_base_window_runs_off_the_start(symbol_frame):
    """起涨前不足 ``base_bars`` 根时，放量倍数的分母无从算起——整行缺失。

    起涨点在 index 10，故 ``base_bars=11`` 要求 index −1 起，越出序列。
    """
    price, vol = volume_sample(symbol_frame)
    anchors = swings_of(price)

    got = volume_structure(vol, anchors, edge_bars=2, base_bars=11)

    assert got.surge_ratio["sh600000"].isna().all()


def test_volume_structure_is_missing_when_the_advance_is_shorter_than_the_edges(symbol_frame):
    """上涨段短于 ``edge_bars`` 时，「起涨」与「顶部」会重叠——宁可整行缺失。

    本例上涨段 5 根，故 ``edge_bars=6`` 越界。
    """
    price, vol = volume_sample(symbol_frame)
    anchors = swings_of(price)

    got = volume_structure(vol, anchors, edge_bars=6, base_bars=4)

    assert got.surge_ratio["sh600000"].isna().all()
    # 恰好等于上涨段长度是允许的：那时顶部就是整段，top_ratio 恒为 1。
    ok = volume_structure(vol, anchors, edge_bars=5, base_bars=4)
    assert ok.top_ratio["sh600000"].iloc[15] == pytest.approx(1.0)


def test_volume_structure_rejects_non_positive_windows(symbol_frame):
    """窗口必须为正；0 或负数会让切片语义变得没有意义，故报错而不是静默取空。"""
    price, vol = volume_sample(symbol_frame)
    anchors = swings_of(price)

    with pytest.raises(ValueError, match="edge_bars"):
        volume_structure(vol, anchors, edge_bars=0, base_bars=4)
    with pytest.raises(ValueError, match="base_bars"):
        volume_structure(vol, anchors, edge_bars=2, base_bars=0)


def test_volume_structure_rejects_mismatched_symbols(symbol_frame):
    """量能与拐点必须出自同一批标的——列不一致时对不上段边界，故报错。"""
    price, _ = volume_sample(symbol_frame)
    anchors = swings_of(price)
    other = symbol_frame({"sz000001": [100.0] * len(PRICES)})

    with pytest.raises(ValueError, match="列"):
        volume_structure(other, anchors, edge_bars=2, base_bars=4)


def test_volume_structure_computes_each_symbol_from_its_own_series(symbol_frame):
    """列之间不串味：第二个标的的成交量是另一个数量级，故它的比值必须按自己的数列算。

    用同一批价格（故段边界相同）、成交量放大 10 倍。三个比值都是**比值**，故应当**不变**。
    """
    price = symbol_frame({"sh600000": PRICES, "sz000001": PRICES})
    base_vol = symbol_frame({"sh600000": VOLUMES, "sz000001": [v * 10 for v in VOLUMES]})
    anchors = swings_of(price)

    got = volume_structure(base_vol, anchors, edge_bars=2, base_bars=4)

    assert got.surge_ratio["sz000001"].iloc[15] == pytest.approx(5.0)
    assert got.pullback_ratio["sz000001"].iloc[15] == pytest.approx(0.08)
    assert got.surge_ratio["sh600000"].iloc[15] == pytest.approx(5.0)


def test_volume_contraction_requires_both_conditions(symbol_frame):
    """两条判据是**合取**：任一条不成立就不算。

    样本的三条比值是 surge=5、pullback=0.08，故 (2, 0.5) 通过、(6, 0.5) 与 (2, 0.05) 都不通过。
    """
    price, vol = volume_sample(symbol_frame)
    anchors = swings_of(price)

    def fires(min_surge, max_pullback):
        return volume_contraction(
            vol, anchors, edge_bars=2, base_bars=4, min_surge=min_surge, max_pullback=max_pullback
        )["sh600000"].iloc[15]

    assert bool(fires(2.0, 0.5)) is True
    assert bool(fires(6.0, 0.5)) is False  # 放量倍数不够
    assert bool(fires(2.0, 0.05)) is False  # 缩量不够


def test_volume_contraction_is_strict_at_both_bounds(symbol_frame):
    """两条都是严格比较：恰好等于阈值不算（与 :func:`new_high`、:func:`volume_surge` 一致）。"""
    price, vol = volume_sample(symbol_frame)
    anchors = swings_of(price)

    def at(min_surge, max_pullback):
        return volume_contraction(
            vol, anchors, edge_bars=2, base_bars=4, min_surge=min_surge, max_pullback=max_pullback
        )["sh600000"].iloc[15]

    assert bool(at(5.0, 0.08)) is True  # 恰好相等
    assert bool(at(5.0 + 1e-9, 0.08)) is False
    assert bool(at(5.0, 0.08 - 1e-9)) is False


def test_volume_contraction_returns_plain_bools_and_never_missing(symbol_frame):
    """过滤器输出纯 ``bool``：段未确认处取「不合格」，不是「未知」。"""
    price = symbol_frame({"sh600000": PRICES, "sz000001": [1.0] * len(PRICES)})
    vol = symbol_frame({"sh600000": VOLUMES, "sz000001": [100.0] * len(PRICES)})
    anchors = swings_of(price)

    got = volume_contraction(
        vol, anchors, edge_bars=2, base_bars=4, min_surge=2.0, max_pullback=0.5
    )

    assert all(dtype.kind == "b" for dtype in got.dtypes)
    assert got["sz000001"].tolist() == [False] * len(PRICES)


def test_volume_structure_keeps_the_symbol_frame_shape_and_float_dtype(symbol_frame):
    """三条比值各是合法的标的宽表：列、索引、dtype 与输入一致（ADR-0009）。"""
    price, vol = volume_sample(symbol_frame)
    anchors = swings_of(price)

    got = volume_structure(vol, anchors, edge_bars=2, base_bars=4)

    for series_ in (got.surge_ratio, got.top_ratio, got.pullback_ratio):
        assert list(series_.columns) == ["sh600000"]
        assert series_.index.equals(vol.index)
        assert all(dtype.kind == "f" for dtype in series_.dtypes)


def test_volume_structure_without_anchors_is_all_missing(symbol_frame):
    """拐点全缺失（价格一路直上、从未确认低点）时，量能结构无从谈起——整表缺失。

    这条同时挡住「把序列首根当成起涨点」这种写法：那会让量能比值随切片起点变化。
    """
    price = symbol_frame({"sh600000": [10.0 + i for i in range(30)]})
    vol = symbol_frame({"sh600000": [100.0 + i for i in range(30)]})
    anchors = swings_of(price)

    got = volume_structure(vol, anchors, edge_bars=2, base_bars=4)

    assert got.surge_ratio["sh600000"].isna().all()


def swings_of(price_frame, retracement=0.10):
    from mbt.signals import swings

    return swings(price_frame, retracement=retracement)


def test_volume_structure_rejects_anchors_from_a_different_index(symbol_frame):
    """手工构造的拐点若与成交量不同日，应当报错而不是对到别的日子上。"""
    vol = symbol_frame({"sh600000": VOLUMES})
    idx = pd.to_datetime(["2024-01-03", "2024-01-02", "2024-01-04"])
    bogus = Swings(
        peak_price=pd.DataFrame({"sh600000": [1.0, 1.0, 1.0]}, index=idx),
        peak_age=pd.DataFrame({"sh600000": [1.0, 1.0, 1.0]}, index=idx),
        trough_price=pd.DataFrame({"sh600000": [1.0, 1.0, 1.0]}, index=idx),
        trough_age=pd.DataFrame({"sh600000": [2.0, 2.0, 2.0]}, index=idx),
    )

    with pytest.raises(ValueError, match="列|索引"):
        volume_structure(vol, bogus, edge_bars=2, base_bars=4)


def wave_frame(symbol_frame, bars=90):
    """一条反复涨跌的序列：确认出**多个**摆动，故量能结构在多行上都有值。

    契约测试用的那份 8 根样本在这件事上太稀薄——它只有 1 格非缺失，截断重算的判据几乎
    空转。故这里单列一条曲线，把「用到未来」这类缺陷真正暴露出来。
    """
    import math

    prices = [100.0 + 20.0 * math.sin(i / 5.0) for i in range(bars)]
    volumes = [1000.0 + 500.0 * math.cos(i / 3.0) for i in range(bars)]
    return symbol_frame({"sh600000": prices}), symbol_frame({"sh600000": volumes})


def test_volume_structure_never_depends_on_bars_after_the_evaluation_day(symbol_frame):
    """截断重算不变性：把输入截到第 k 根重算，前 k 根必须与全长结果逐值相同。

    这是**独立于实现**的因果性判据——任何「用了未来」的写法都会在这里露馅。本函数尤其
    需要它：段聚合是跨多根的滑窗，最容易不小心吃到当根之后的行。
    """
    price, vol = wave_frame(symbol_frame)
    anchors_full = swings_of(price, retracement=0.05)
    full = volume_structure(vol, anchors_full, edge_bars=3, base_bars=5)

    checked = 0
    for k in range(1, len(price) + 1):
        price_k = price.iloc[:k]
        vol_k = vol.iloc[:k]
        anchors_k = swings_of(price_k, retracement=0.05)
        got = volume_structure(vol_k, anchors_k, edge_bars=3, base_bars=5)
        for name in ("surge_ratio", "top_ratio", "pullback_ratio"):
            expected = getattr(full, name).iloc[:k]
            pd.testing.assert_frame_equal(getattr(got, name), expected, check_exact=True)
        checked += int(full.surge_ratio.iloc[:k].notna().to_numpy().sum())

    assert checked > 100, f"判据几乎空转：只比对了 {checked} 个有值的格子"


def test_volume_structure_is_invariant_to_scaling_the_volume_series(symbol_frame):
    """成交量整体放大同一个倍数，三个比值**不变**。

    这条不只是个性质，它还解释了**为什么这里没有那条惯用的「自检」测试**：上游契约测试里
    用「除以整列均值」来伪造未来函数，靠的是信号是**水平量**；而本函数的三条输出都是比值，
    分子分母同乘一个常数必然约掉，故那种伪造在这里**不可能**改变输出。也就是说，那条自检
    对本函数恒为绿，拿它当保证等于没有保证。

    真正防空转的是上一个测试末尾的 ``checked > 100``：它直接断言「比对了足够多有值的格子」。

    容差说明：约掉是数学结论，但在浮点上是**同一比值用不同的分子分母再除一次**，故末位
    会差一个 ULP（实测差 1e-16 量级）。这里用 ``rtol=1e-12`` 而不是逐位相等——把
    「数学上相等、浮点上差了末位」写成「不相等」是另一种失真。
    """
    price, vol = wave_frame(symbol_frame)
    anchors = swings_of(price, retracement=0.05)

    plain = volume_structure(vol, anchors, edge_bars=3, base_bars=5)
    scaled = volume_structure(vol * 7.5, anchors, edge_bars=3, base_bars=5)

    for name in ("surge_ratio", "top_ratio", "pullback_ratio"):
        pd.testing.assert_frame_equal(getattr(scaled, name), getattr(plain, name), rtol=1e-12)
    assert plain.surge_ratio.notna().to_numpy().sum() > 20, "样本太稀薄，这条性质没被真正检验"
