"""量能结构：按摆动点切段，度量上涨放量与回调缩量。

样本设计成**可手算**：价格路径与成交量都取整数，三个比值都能用纸笔复算，故实现改了哪一步
都会在这里露馅。
"""

from __future__ import annotations

import pandas as pd
import pytest

from mbt.data import Panel
from mbt.signals import Swings, atr, volume_contraction, volume_pattern, volume_structure

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

    for series_ in (
        got.surge_ratio,
        got.top_ratio,
        got.pullback_ratio,
        got.pullback_vs_top_ratio,
    ):
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
    """两条都是严格比较：恰好等于阈值不算（与 :func:`new_high`、:func:`volume_surge` 一致）。

    缩量那条的边界值是 **40/500**——即回归后的口径（回调均量 ÷ 上涨段**单日最大量**）。
    """
    price, vol = volume_sample(symbol_frame)
    anchors = swings_of(price)
    ratio = 40.0 / 500.0

    def at(min_surge, max_pullback):
        return volume_contraction(
            vol, anchors, edge_bars=2, base_bars=4, min_surge=min_surge, max_pullback=max_pullback
        )["sh600000"].iloc[15]

    assert bool(at(5.0, ratio)) is True  # 恰好相等
    assert bool(at(5.0 + 1e-9, ratio)) is False
    assert bool(at(5.0, ratio - 1e-9)) is False


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

    for series_ in (
        got.surge_ratio,
        got.top_ratio,
        got.pullback_ratio,
        got.pullback_vs_top_ratio,
    ):
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
        for name in ("surge_ratio", "top_ratio", "pullback_ratio", "pullback_vs_top_ratio"):
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

    for name in ("surge_ratio", "top_ratio", "pullback_ratio", "pullback_vs_top_ratio"):
        pd.testing.assert_frame_equal(getattr(scaled, name), getattr(plain, name), rtol=1e-12)
    assert plain.surge_ratio.notna().to_numpy().sum() > 20, "样本太稀薄，这条性质没被真正检验"


def wavy_panel(symbol_frame, symbols, bars=120):
    """每个标的走**各自不同相位**的波浪行情，故它们的摆动点互不相同。

    这正是分组实现最容易出错的那一类输入。共用一条价格序列的样本测不出这种错——那样所有标的
    落在**同一个**摆动对上，分组退化成一组，于是「分错组」不可能发生。
    """
    import math

    prices = {
        f"s{i:05d}": [100.0 + 20.0 * math.sin(j / 5.0 + i) for j in range(bars)]
        for i in range(symbols)
    }
    volumes = {
        f"s{i:05d}": [1000.0 + 500.0 * math.cos(j / 3.0 + i) for j in range(bars)]
        for i in range(symbols)
    }
    return symbol_frame(prices), symbol_frame(volumes)


def test_output_of_a_symbol_does_not_depend_on_which_other_symbols_are_present(symbol_frame):
    """一个标的的取值**只取决于它自己那一列**，与同批还有谁在完全无关。

    这不是可有可无的性质，而是一次优化的**许可条件**：实现现在只对被问到的那几列算窗口
    （见 ``_window_aggregates`` 的说明），而这只有在「每一列独立」成立时才与整体算逐位相同。
    故这条同时盯两件事——列之间不串味，以及那条优化没把「不串味」弄丢。
    """
    price, volume = wavy_panel(symbol_frame, symbols=4)
    many = volume_structure(volume, swings_of(price, retracement=0.05), edge_bars=3, base_bars=5)

    non_missing = 0
    for i in range(4):
        symbol = f"s{i:05d}"
        # 单独跑第 i 个标的：只留它一列，段边界按它自己那条曲线重算。
        alone = volume_structure(
            symbol_frame({symbol: volume[symbol].tolist()}),
            swings_of(symbol_frame({symbol: price[symbol].tolist()}), retracement=0.05),
            edge_bars=3,
            base_bars=5,
        )
        for name in ("surge_ratio", "top_ratio", "pullback_ratio", "pullback_vs_top_ratio"):
            pd.testing.assert_frame_equal(
                getattr(alone, name), getattr(many, name)[[symbol]], check_exact=True
            )
            non_missing += int(getattr(alone, name).notna().to_numpy().sum())

    assert non_missing > 50, f"判据几乎空转：只比对了 {non_missing} 个有值的格子"


def test_a_fully_missing_symbol_does_not_poison_its_group(symbol_frame):
    """成交量整列缺失的标的自己给 NaN，且**不拖累**与它同组的标的。

    长期停牌的标的列就是这样（整段无有效值），而按列算窗口之后会多出「这一组整段没有有效值」
    这么一条分支——它若把整组一起置空，同组的邻居就被无辜抹掉了。
    """
    prices = symbol_frame({"sh600000": PRICES, "sz000001": PRICES, "sz000002": PRICES})
    blanks = [float("nan")] * len(PRICES)
    volumes = symbol_frame({"sh600000": VOLUMES, "sz000001": VOLUMES, "sz000002": blanks})

    got = volume_structure(volumes, swings_of(prices), edge_bars=2, base_bars=4)

    assert pd.isna(got.surge_ratio["sz000002"].iloc[15]), "整列缺失的标的应当缺失"
    assert got.surge_ratio["sh600000"].iloc[15] == pytest.approx(5.0), "同组的邻居被拖累了"
    assert got.surge_ratio["sz000001"].iloc[15] == pytest.approx(5.0)


def test_the_work_grows_with_the_number_of_symbols_not_with_its_square(symbol_frame, monkeypatch):
    """标的数翻倍，窗口计算量只该翻倍——**这条盯着的是一个真实出现过的二次开销**。

    曾经每加一个标的，代价按标的数**平方**增长，来源有两处：每行先铺一张「摆动对 × 标的」的
    表再挑回一列；窗口按全部标的算，而实际只有一两只用到。全市场实测因此从 5 秒涨到一个多
    小时，且现象是「不报错、只是永远跑不完」——正是最需要一条测试钉住的那类缺陷。

    判据取**比值**而不是绝对秒数：2 = 线性，4 = 二次。绝对耗时依赖机器，比值不依赖，故这条
    测试不会因为换了台电脑就变红。
    """
    from mbt.signals import volume as volume_module

    def work_for(symbols):
        counted = {"elements": 0}
        real = volume_module._window_aggregates

        def counting(vol, start, end, columns):
            rows = max(0, min(end, vol.shape[0] - 1) - max(start, 0) + 1)
            counted["elements"] += rows * int(columns.size)
            return real(vol, start, end, columns)

        monkeypatch.setattr(volume_module, "_window_aggregates", counting)
        price, volume = wavy_panel(symbol_frame, symbols=symbols)
        volume_structure(volume, swings_of(price, retracement=0.05), edge_bars=3, base_bars=5)
        monkeypatch.undo()
        return counted["elements"]

    small = work_for(4)
    large = work_for(8)
    assert small > 0, "没数到任何窗口计算——计数器没接上，这条测试在空转"

    growth = large / small
    assert growth < 3.0, (
        f"标的数翻倍（4→8），窗口计算量却涨了 {growth:.1f} 倍："
        f"线性应当在 2 倍上下，接近 4 倍说明二次开销又回来了"
    )


# --- 回调缩量：参照物取「顶部段」而不是「上涨段最大值」 ----------------------------


def test_pullback_vs_top_ratio_divides_by_the_top_segment_not_the_advance_max(symbol_frame):
    """新比值的分母是**顶部段均量**，与 ``pullback_ratio``（分母是上涨段最大值）不是同一个量。

    夹具同前：基准 [6,9] = 100、上涨段 [10,14] 最大 500 / 均值 180、顶部两根 [13,14] 均值 300、
    回调 [15] = 40。故

    - ``pullback_ratio``        = 40 / **500** = 0.08（分母是上涨段**最大**量）
    - ``pullback_vs_top_ratio`` = 40 / **300** = 0.1333…（分母是**顶部段均量**）

    两者刻意放在一条测试里断言：它们是**两个不同的量**（不是彼此的换标度），若实现把新字段
    也算成 40/500，这条会红；若把旧字段一起改掉，前半句会红。
    """
    price, vol = volume_sample(symbol_frame)
    anchors = swings_of(price)

    got = volume_structure(vol, anchors, edge_bars=2, base_bars=4)

    row = 15
    assert got.pullback_ratio["sh600000"].iloc[row] == pytest.approx(0.08)
    assert got.pullback_vs_top_ratio["sh600000"].iloc[row] == pytest.approx(40.0 / 300.0)


def test_volume_contraction_judges_the_spike_referenced_ratio(symbol_frame):
    """``volume_contraction`` 的缩量那条看的是 **``pullback_ratio``**（÷ 上涨段单日最大量）。

    夹具上两个比值分居一个门槛的两侧，故这条能判别实现用的是哪一个：

    - ``pullback_ratio``（÷ 500）        = 0.08  <= 0.10 → 放行
    - ``pullback_vs_top_ratio``（÷ 300） = 0.133 >  0.10 → 不放行

    取 ``max_pullback=0.10`` 断言为 ``True``，即「仍在看单日爆量那个分母」。

    **为什么不是顶部口径**：顶部口径实现过、也跑过全市场对照，结果是**更差**——逐笔从 8,601
    降到 3,145、收益率均值 +0.164% → +0.116%、均值/标准误 3.90 → 1.68，而被它排掉的那 5,969 笔
    反而是更好的交易（均值 +0.190%、均值/标准误 +3.74）。即「相对顶部安静」与收益**反向**。
    故判据改回，``pullback_vs_top_ratio`` 只作为**数据**保留（见 ADR-0011）。
    """
    price, vol = volume_sample(symbol_frame)
    anchors = swings_of(price)

    def fires(max_pullback):
        return volume_contraction(
            vol,
            anchors,
            edge_bars=2,
            base_bars=4,
            min_surge=2.0,
            max_pullback=max_pullback,
        )["sh600000"].iloc[15]

    assert bool(fires(0.10)) is True, "0.08 <= 0.10，看的是爆量口径"
    assert bool(fires(0.07)) is False, "0.08 > 0.07，收紧就该挡住——证明上一条不是恒真"


# --- 新口径（ADR-0012）的形态读数 -------------------------------------------------------
#
# 这一组盯两件事，缺一不可：
#
# 1. **口径**——顶部是「最高价那根」而不是收盘口径峰值、分母是「上涨段除去顶部那根」、
#    影取正值。这些只有**手算**能钉住，故第一组样本刻意做成整数、且让最高价那根落在
#    收盘口径峰值**之前**一根（两个口径分叉的情形，正是新口径改的那一处）。
# 2. **那三处省时间的手法**——列优先的「上一次窗口」缓存、顶部那根的行进 argmax、
#    顶部取值的花式索引。这些手算钉不住（手算样本只有十根、一条列），故另写一份
#    **朴素到不可能优化错**的实现，在各自相位的波浪行情上逐格对。

# 最高价那根在 index 5（12.0），收盘口径峰值在 index 6——两个口径**分叉**。
HAND_CLOSE = [5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 9.0, 8.0, 7.5]
HAND_OPEN = [4.5, 5.5, 6.5, 7.5, 8.5, 9.5, 10.5, 8.5, 7.5, 7.0]
HAND_HIGH = [5.2, 6.2, 7.2, 8.2, 9.2, 12.0, 11.5, 9.2, 8.5, 8.0]
HAND_LOW = [4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 8.0, 7.0, 6.5]
HAND_VOLUME = [10.0, 20.0, 30.0, 40.0, 50.0, 100.0, 60.0, 30.0, 20.0, 10.0]


def hand_pattern_panel(panel):
    """十根**可手算**的样本，OHLC 合法且上下影各不等。"""
    return panel(
        {
            "open": {"sh600000": HAND_OPEN},
            "high": {"sh600000": HAND_HIGH},
            "low": {"sh600000": HAND_LOW},
            "close": {"sh600000": HAND_CLOSE},
            "volume": {"sh600000": HAND_VOLUME},
        }
    )


def hand_anchors(symbol_frame):
    """手工敲定的拐点：``T = 3``、收盘口径峰值 ``P = 6``，峰值确认（第 6 根）之前一律缺失。

    刻意手工构造而不是跑 ``swings``：这条测试要钉的是**读数的口径**，把摆动识别也拉进来
    只会让失败时看不出是哪一层错了。价格列本函数不读，故留缺失。
    """
    blank = [float("nan")] * 6
    return Swings(
        peak_price=symbol_frame({"sh600000": [float("nan")] * 10}),
        peak_age=symbol_frame({"sh600000": blank + [0.0, 1.0, 2.0, 3.0]}),
        trough_price=symbol_frame({"sh600000": [float("nan")] * 10}),
        trough_age=symbol_frame({"sh600000": blank + [3.0, 4.0, 5.0, 6.0]}),
    )


def test_the_pattern_readings_match_the_hand_computed_ratios(symbol_frame, panel):
    """第 9 根上四条读数逐一手算锁定。

    ``T = 3``、``P = 6``、顶部那根 = 5（最高价 12.0）、``R = max(P, 顶) = 6``、
    ``S = 9``，``base_bars = 2``：

    - 起涨前基准 ``[1,2]`` = 20、30 → 均值 25
    - 上涨段 ``[3,6]`` = 40、50、100、60 → 最大 100
    - 上涨段除去顶部那根 = 40、50、**60** → 最大 60、均值 50
    - 回调段 ``[7,9]`` = 30、20、10 → 均值 20
    - 顶部的影 = 12.0 − max(9.5, 10.0) = 2.0

    故 ``surge = 100/25 = 4``、``top_calm = 100/60``、``pullback = 20/50 = 0.4``。
    """
    built = hand_pattern_panel(panel)
    anchors = hand_anchors(symbol_frame)

    got = volume_pattern(built, anchors, base_bars=2, atr_n=3)

    symbol = "sh600000"
    assert got.surge_vs_base[symbol].iloc[9] == pytest.approx(4.0)
    assert got.top_calm[symbol].iloc[9] == pytest.approx(100.0 / 60.0)
    assert got.pullback_vs_advance[symbol].iloc[9] == pytest.approx(0.4)
    assert got.top_shadow_atr[symbol].iloc[9] == pytest.approx(2.0 / atr(built, 3)[symbol].iloc[5])


def test_the_top_bar_is_the_highest_high_not_the_close_peak(symbol_frame, panel):
    """「顶部」是**最高价**那根，不是收盘口径峰值——本样本里两者正好差一根。

    收盘口径峰值在 index 6（收盘 11.0 是最高收盘），但最高价 12.0 在 index 5。新口径
    取后者，且上涨段的右端取 ``max(P, 顶) ``故**不会缩短**；``top_after_peak`` 记下这个差
    （``-1`` = 最高价那根在收盘峰值**之前**一根）。
    """
    built = hand_pattern_panel(panel)
    anchors = hand_anchors(symbol_frame)

    got = volume_pattern(built, anchors, base_bars=2, atr_n=3)

    symbol = "sh600000"
    assert got.top_after_peak[symbol].iloc[9] == pytest.approx(-1.0)
    # 顶部那根的成交量是 100；若错把收盘峰值那根（60）当顶部，这个比值就变成 60/100。
    assert got.top_calm[symbol].iloc[9] == pytest.approx(100.0 / 60.0)


def test_the_shadow_is_positive_so_that_smaller_is_better(symbol_frame, panel):
    """影取**正值**（``high − max(开, 收)``），故「越小越好」；写成反向会得到一个恒非正的数。

    恒非正会让秩归一整个反过来：长上影会被当成「最好」。
    """
    built = hand_pattern_panel(panel)
    anchors = hand_anchors(symbol_frame)

    got = volume_pattern(built, anchors, base_bars=2, atr_n=3)

    values = got.top_shadow_atr["sh600000"].dropna()
    assert (values >= 0.0).all(), f"影必须非负，收到 {values.tolist()}"
    assert values.iloc[-1] == pytest.approx(2.0 / atr(built, 3)["sh600000"].iloc[5])


def test_the_pattern_readings_are_missing_before_the_peak_is_confirmed(symbol_frame, panel):
    """峰值确认之前没有段，五条读数一律缺失——不因为「显然有个高点」就先报出来。"""
    built = hand_pattern_panel(panel)
    anchors = hand_anchors(symbol_frame)

    got = volume_pattern(built, anchors, base_bars=2, atr_n=3)

    # 回调段在第 6 根为空（`S == R`），故这一条从第 7 根起才有值。
    for name in ("surge_vs_base", "top_calm", "top_shadow_atr", "top_after_peak"):
        frame = getattr(got, name)["sh600000"]
        assert frame.iloc[6:].notna().all(), f"{name} 确认之后应当有值"
        assert frame.iloc[:6].isna().all(), f"{name} 确认之前不应当有值"
    pullback = got.pullback_vs_advance["sh600000"]
    assert pullback.iloc[7:].notna().all()
    assert pullback.iloc[:7].isna().all()


def test_the_pullback_segment_is_missing_on_the_bar_the_advance_ends(symbol_frame, panel):
    """回调段为空（``S == R``）时只有 ``pullback_vs_advance`` 缺失，其余三条照算。

    这条防的是「一段没有数据就把整格扔掉」的写法：那样第 6 根上``surge`` / ``top_calm``
    也会跟着消失，而它们与回调段无关。
    """
    built = hand_pattern_panel(panel)
    anchors = hand_anchors(symbol_frame)

    got = volume_pattern(built, anchors, base_bars=2, atr_n=3)
    symbol = "sh600000"

    assert pd.isna(got.pullback_vs_advance[symbol].iloc[6]), "第 6 根回调段为空"
    assert got.surge_vs_base[symbol].iloc[6] == pytest.approx(100.0 / 25.0)
    assert got.top_calm[symbol].iloc[6] == pytest.approx(100.0 / 60.0)


def test_the_base_window_running_off_the_start_only_costs_the_surge_ratio(symbol_frame, panel):
    """起涨前不足 ``base_bars`` 根时，只有 ``surge_vs_base`` 缺失，其余三条照算。"""
    built = hand_pattern_panel(panel)
    # 把起涨点挪到 index 1（故 ``trough_age`` 从第 3 根起是 2、3、…），而 base_bars=2 要求
    # 窗口到 [-1, 0]——跑到序列开头之外，基准算不出来。
    blank = [float("nan")] * 3
    ages = blank + [2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]
    anchors = Swings(
        peak_price=symbol_frame({"sh600000": [float("nan")] * 10}),
        peak_age=symbol_frame({"sh600000": blank + [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0]}),
        trough_price=symbol_frame({"sh600000": [float("nan")] * 10}),
        trough_age=symbol_frame({"sh600000": ages}),
    )

    got = volume_pattern(built, anchors, base_bars=2, atr_n=3)
    symbol = "sh600000"

    assert pd.isna(got.surge_vs_base[symbol].iloc[3]), "起涨前不足 2 根，基准算不出来"
    assert pd.notna(got.top_calm[symbol].iloc[3]), "顶部无量与基准窗口无关"


def test_the_pattern_rejects_non_positive_windows(symbol_frame, panel):
    """``base_bars`` / ``atr_n`` 非正应当报错，而不是算出一堆缺失让人以为「就是没值」。"""
    built = hand_pattern_panel(panel)
    anchors = hand_anchors(symbol_frame)

    with pytest.raises(ValueError, match="base_bars"):
        volume_pattern(built, anchors, base_bars=0, atr_n=3)
    with pytest.raises(ValueError, match="atr_n"):
        volume_pattern(built, anchors, base_bars=2, atr_n=0)


def test_the_pattern_names_the_fields_it_is_missing(symbol_frame, panel):
    """面板缺字段时点名缺的是哪个，而不是让它冒成一个 KeyError。"""
    built = hand_pattern_panel(panel)
    anchors = hand_anchors(symbol_frame)
    thin = panel(
        {
            "open": {"sh600000": HAND_OPEN},
            "close": {"sh600000": HAND_CLOSE},
            "volume": {"sh600000": HAND_VOLUME},
        }
    )

    with pytest.raises(ValueError, match="high"):
        volume_pattern(thin, anchors, base_bars=2, atr_n=3)
    assert pd.notna(volume_pattern(built, anchors, base_bars=2, atr_n=3).top_calm.iloc[9, 0])


def test_the_pattern_rejects_anchors_from_a_different_index(symbol_frame, panel):
    """拐点与行情不同日应当报错——对到别的日子上不会报错，只会静默算错。"""
    built = hand_pattern_panel(panel)
    idx = pd.to_datetime(["2024-01-03", "2024-01-02", "2024-01-04"])
    bogus = Swings(
        peak_price=pd.DataFrame({"sh600000": [1.0, 1.0, 1.0]}, index=idx),
        peak_age=pd.DataFrame({"sh600000": [1.0, 1.0, 1.0]}, index=idx),
        trough_price=pd.DataFrame({"sh600000": [1.0, 1.0, 1.0]}, index=idx),
        trough_age=pd.DataFrame({"sh600000": [2.0, 2.0, 2.0]}, index=idx),
    )

    with pytest.raises(ValueError, match="列|索引"):
        volume_pattern(built, bogus, base_bars=2, atr_n=3)


def ohlcv_of(prices, volumes):
    """把一条**收盘价**曲线撑成合法 OHLCV：开盘取上一根收盘，最高/最低在实体外留一段余量。

    余量**不等**（随 i 变）是刻意的：若上下影恒定，最高价那根就会与实体最大那根重合，
    而「顶部 = 最高价那根」与「收盘口径峰值」正是要能分开的两件事，样本必须让它们分叉。
    """
    import math

    fields = {name: {} for name in ("open", "high", "low", "close", "volume")}
    for symbol in prices.columns:
        close = prices[symbol].tolist()
        open_ = [close[0]] + close[:-1]
        pad = [0.3 + 0.6 * math.sin(i / 2.0) ** 2 for i in range(len(close))]
        fields["open"][symbol] = open_
        fields["close"][symbol] = close
        fields["high"][symbol] = [max(o, c) + p for o, c, p in zip(open_, close, pad, strict=True)]
        fields["low"][symbol] = [
            min(o, c) - p / 2.0 for o, c, p in zip(open_, close, pad, strict=True)
        ]
        fields["volume"][symbol] = volumes[symbol].tolist()
    return fields


def brute_force_pattern(built, anchors, *, base_bars, atr_n):
    """**另写一遍**的朴素实现：逐格、逐窗口、不缓存、不花式索引。

    它与 :func:`volume_pattern` 唯一的共同点是段边界（都取自 ``anchors``）与 ATR（都调库里的
    :func:`atr`）——故两者一致才有意义。实现里那三处省时间的手法（列优先的「上一次窗口」
    缓存、顶部那根的行进 argmax、顶部取值的花式索引）都会被它逮住。

    刻意不写成「按位置扫一遍」：那样会和被测实现的思路撞车，撞车之后就审不出同一个错。
    """
    import math

    import numpy as np

    high = built["high"].to_numpy(dtype=float)
    open_ = built["open"].to_numpy(dtype=float)
    close = built["close"].to_numpy(dtype=float)
    vol = built["volume"].to_numpy(dtype=float)
    rows, columns = vol.shape
    trough_age = anchors.trough_age.to_numpy(dtype=float)
    peak_age = anchors.peak_age.to_numpy(dtype=float)
    atr_frame = atr(built, atr_n).to_numpy(dtype=float)

    names = ("surge_vs_base", "top_calm", "pullback_vs_advance", "top_shadow_atr", "top_after_peak")
    out = {name: np.full((rows, columns), math.nan) for name in names}

    for row in range(rows):
        for column in range(columns):
            age_trough = trough_age[row, column]
            age_peak = peak_age[row, column]
            if not (math.isfinite(age_trough) and math.isfinite(age_peak)):
                continue
            trough = int(row - age_trough)
            peak = int(row - age_peak)
            if trough < 0 or peak < trough:
                continue
            window = high[trough : row + 1, column]
            if not np.isfinite(window).any():
                continue
            top = trough + int(np.nanargmax(window))
            right = max(peak, top)

            # 逐项都用最直白的窗口写法：显式构造「除去顶部那根」的那一段。
            advance = vol[trough : right + 1, column]
            rest = np.delete(advance, top - trough)
            if trough - base_bars >= 0:
                base = vol[trough - base_bars : trough, column]
            else:
                base = np.array([], dtype=float)
            pull = vol[right + 1 : row + 1, column]

            out["top_after_peak"][row, column] = float(top - peak)
            base_mean = np.nanmean(base) if np.isfinite(base).any() else math.nan
            if np.isfinite(base_mean) and base_mean > 0 and np.isfinite(advance).any():
                out["surge_vs_base"][row, column] = np.nanmax(advance) / base_mean
            rest_max = np.nanmax(rest) if np.isfinite(rest).any() else math.nan
            if np.isfinite(rest_max) and rest_max > 0 and np.isfinite(vol[top, column]):
                out["top_calm"][row, column] = vol[top, column] / rest_max
            rest_mean = np.nanmean(rest) if np.isfinite(rest).any() else math.nan
            pull_mean = np.nanmean(pull) if np.isfinite(pull).any() else math.nan
            if np.isfinite(rest_mean) and rest_mean > 0 and np.isfinite(pull_mean):
                out["pullback_vs_advance"][row, column] = pull_mean / rest_mean
            atr_value = atr_frame[top, column]
            if np.isfinite(atr_value) and atr_value > 0:
                body = max(open_[top, column], close[top, column])
                out["top_shadow_atr"][row, column] = (high[top, column] - body) / atr_value

    return out


def test_the_pattern_readings_match_a_brute_force_pass_over_every_cell(symbol_frame, panel):
    """逐格朴素实现与库里那份优化过的实现必须处处相同。

    「处处」不是客套：三处省时间的手法（列优先的窗口缓存、行进 argmax、花式索引）各自
    只在**特定输入**下才会露馅——缓存要同一窗口连续多行不变、argmax 要顶部那根换列换段、
    花式索引要 ``valid`` 为真却落在别的列上。各自相位的波浪行情同时给了这三类输入。
    """
    prices, volumes = wavy_panel(symbol_frame, symbols=4, bars=120)
    built = panel(ohlcv_of(prices, volumes))
    anchors = swings_of(prices, retracement=0.05)

    got = volume_pattern(built, anchors, base_bars=5, atr_n=3)
    want = brute_force_pattern(built, anchors, base_bars=5, atr_n=3)

    filled = 0
    for name in want:
        mine = getattr(got, name)
        pd.testing.assert_frame_equal(
            mine,
            pd.DataFrame(want[name], index=mine.index, columns=mine.columns),
            check_exact=False,
            rtol=1e-12,
            atol=1e-12,
            check_names=False,
        )
        filled += int(mine.notna().to_numpy().sum())

    assert filled > 100, f"判据几乎空转：只比对了 {filled} 个有值的格子"


def test_the_pattern_readings_never_depend_on_bars_after_the_evaluation_day(symbol_frame, panel):
    """截断重算不变性：把输入截到第 k 根重算，前 k 根必须与全长结果逐值相同。

    这条对**列优先的窗口缓存**尤其要紧：缓存键里少写一项（例如左侧窗口只按 ``T`` 记，
    而忘了 ``顶`` 也跟着变），前 k 根的值就会取决于「第 k 根之后有没有出现过新的摆动点」——
    那正是这类「记住上一次结果」的手法最容易漏掉未来的地方。逐列扫描的顶部 argmax 同理。
    """
    prices, volumes = wavy_panel(symbol_frame, symbols=3, bars=90)
    built = panel(ohlcv_of(prices, volumes))
    full = volume_pattern(built, swings_of(prices, retracement=0.05), base_bars=5, atr_n=3)

    checked = 0
    for k in range(1, len(prices) + 1):
        truncated = Panel({name: built[name].iloc[:k] for name in built.field_names})
        got = volume_pattern(
            truncated, swings_of(prices.iloc[:k], retracement=0.05), base_bars=5, atr_n=3
        )
        for name in full._fields:
            want = getattr(full, name).iloc[:k]
            pd.testing.assert_frame_equal(getattr(got, name), want, check_exact=True)
            checked += int(want.notna().to_numpy().sum())

    assert checked > 100, f"判据几乎空转：只比对了 {checked} 个有值的格子"


# --- 窗口内部的缺失：必须「跳过」，不能当 0 ------------------------------------


def _pattern_with_pullback_volumes(symbol_frame, panel, pullback_volumes):
    """把回调段那三根（index 7/8/9）的成交量换成给定值，其余照手算夹具。"""
    volumes = list(HAND_VOLUME)
    volumes[7:10] = pullback_volumes
    built = panel(
        {
            "open": {"sh600000": HAND_OPEN},
            "high": {"sh600000": HAND_HIGH},
            "low": {"sh600000": HAND_LOW},
            "close": {"sh600000": HAND_CLOSE},
            "volume": {"sh600000": volumes},
        }
    )
    return volume_pattern(built, hand_anchors(symbol_frame), base_bars=2, atr_n=3)


def test_a_missing_bar_inside_the_pullback_window_is_skipped_not_counted_as_zero(
    symbol_frame, panel
):
    """回调窗口**中间**缺一根时，均值按剩下的有限值算——不是把缺失当 0 计入。

    手算：回调段 ``[7,9]`` 原为 30、**20**、10（均值 20）。把中间那根换成缺失之后：

    - **跳过**（正确）：``(30 + 10) / 2 = 20``
    - 当 0 计入（错误）：``(30 + 0 + 10) / 3 = 13.33``

    两种处置给出不同的数，故这条断言能判别。它与 :func:`~mbt.signals.volume._finite_stats`
    的语义一致（只统计有限值），也与本项目「缺失不是 0」的一贯口径一致。

    为什么要单独钉：回调段是唯一**每格都要重算**的窗口（右端就是当前行），也是最容易被
    换成别的求和方式的一处；换的时候「缺失怎么算」极易跟着变味。
    """
    got = _pattern_with_pullback_volumes(symbol_frame, panel, [30.0, float("nan"), 10.0])

    assert got.pullback_vs_advance["sh600000"].iloc[9] == pytest.approx(20.0 / 50.0)


def test_a_pullback_window_that_is_entirely_missing_gives_missing_not_zero(symbol_frame, panel):
    """回调窗口**整段**缺失时给缺失，不给 0——给 0 会让「没有数据」看起来像「极度缩量」。

    这条与上一条成对：上一条钉「部分缺失按有限值算」，这一条钉「一个有限值都没有时不编数」。
    其余三条读数与回调段无关，故照常有值。
    """
    got = _pattern_with_pullback_volumes(
        symbol_frame, panel, [float("nan"), float("nan"), float("nan")]
    )

    symbol = "sh600000"
    assert pd.isna(got.pullback_vs_advance[symbol].iloc[9]), "整段缺失不该给 0"
    assert got.top_calm[symbol].iloc[9] == pytest.approx(100.0 / 60.0)
    assert got.surge_vs_base[symbol].iloc[9] == pytest.approx(4.0)


def pullback_anchors(symbol_frame):
    """把手算夹具的峰值挪到 index 5，使**回调段有 4 根**（``[6, 9]``）。

    原 ``hand_anchors`` 的峰值在 index 6，故回调段只有 ``[7, 9]`` 三根、且第 9 根时只剩一根。
    要测「窗口**内部**有缺失」，窗口得够长。

    手算依据（``HAND_VOLUME``，``edge_bars=2``、``base_bars=2``）：
    ``T = 3``、``P = 5``、顶部那根 = 5、``R = max(P, 顶) = 5``，于是

    - 上涨段 ``[3, 5]`` = 40、50、100 → 最大 **100**（``pullback_ratio`` 的分母）
    - 回调段 ``[6, 9]`` = 60、30、20、10 → 均值 **30**
    - 故 ``pullback_ratio`` = 30 / 100 = **0.30**
    """
    blank = [float("nan")] * 5
    return Swings(
        peak_price=symbol_frame({"sh600000": [float("nan")] * 10}),
        peak_age=symbol_frame({"sh600000": blank + [0.0, 1.0, 2.0, 3.0, 4.0]}),
        trough_price=symbol_frame({"sh600000": [float("nan")] * 10}),
        trough_age=symbol_frame({"sh600000": blank + [2.0, 3.0, 4.0, 5.0, 6.0]}),
    )


def test_a_missing_bar_inside_the_structure_pullback_window_is_skipped_not_counted_as_zero(
    symbol_frame,
):
    """`volume_structure` 的回调窗口**内部**缺一根时，均值按剩下的有限值算。

    取回调段第一根（60）为缺失，两种处置给出不同的数：

    - **跳过**（正确）：``(30 + 20 + 10) / 3 = 20`` → ``20 / 100 = 0.20``
    - 当 0 计入（错误）：``(0 + 30 + 20 + 10) / 4 = 15`` → 0.15
    - 若把缺失当分母不算、分子算 0：``90 / 4 = 22.5`` → 0.225

    三种处置互不相同，故这条断言能判别实现走了哪一条。
    """
    volumes = list(HAND_VOLUME)
    volumes[6] = float("nan")  # 回调段第一根（60）
    got = volume_structure(
        symbol_frame({"sh600000": volumes}),
        pullback_anchors(symbol_frame),
        edge_bars=2,
        base_bars=2,
    )

    assert got.pullback_ratio["sh600000"].iloc[9] == pytest.approx(20.0 / 100.0)


def test_the_complete_window_still_gives_the_hand_computed_ratio(symbol_frame):
    """上一条的对照：不插缺失时仍是手算的 0.30——证明上一条的 0.20 来自缺失，不是别处的改动。"""
    got = volume_structure(
        symbol_frame({"sh600000": list(HAND_VOLUME)}),
        pullback_anchors(symbol_frame),
        edge_bars=2,
        base_bars=2,
    )

    assert got.pullback_ratio["sh600000"].iloc[9] == pytest.approx(0.30)


def test_a_structure_pullback_window_that_is_entirely_missing_gives_missing_not_zero(
    symbol_frame,
):
    """`volume_structure` 的窗口**整段**缺失时给缺失，不给 0。"""
    volumes = list(HAND_VOLUME)
    for index in range(6, 10):
        volumes[index] = float("nan")
    got = volume_structure(
        symbol_frame({"sh600000": volumes}),
        pullback_anchors(symbol_frame),
        edge_bars=2,
        base_bars=2,
    )

    assert pd.isna(got.pullback_ratio["sh600000"].iloc[9]), "整段缺失不该给 0"
