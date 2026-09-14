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
