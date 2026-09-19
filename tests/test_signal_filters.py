"""过滤信号：对**单一标的**的是/否判断（``CONTEXT.md``），票据 #5。

两条约定在此锁定：

1. 输出是**纯 ``bool``** 的标的宽表，缺失一律取 ``False``——语义是「不合格」。
   不引入 nullable boolean：下游（回测、选股）要的是「能不能买」这个二值答案，
   把「未知」再传下去只会让每个消费者各写一遍填充逻辑。
2. 判断只在**已知**上成立。窗口不足或输入缺失时不给乐观答案，一律 ``False``。
"""

from __future__ import annotations

import pandas as pd
import pytest

from mbt.signals import (
    above_ma,
    above_white,
    above_yellow,
    below_white,
    below_yellow_streak,
    close_above_high,
    high_above_white,
    j_below,
    ma_cross_up,
    new_high,
    no_contained_run,
    pullback_after_advance,
    rising_streak,
    volume_surge,
    white_above_yellow,
    white_line,
    yellow_line,
)


def test_new_high_is_strict_so_exactly_matching_the_prior_high_does_not_count(symbol_frame):
    """恰好等于前 n 日最高价**不算**新高——边界在此，差一分才算越出。"""
    prices = symbol_frame({"sh600000": [10.0, 11.0, 12.0, 12.0, 12.01]})

    got = new_high(prices, n=3)["sh600000"]

    assert got.tolist() == [False, False, False, False, True]


def test_new_high_never_fires_when_the_window_has_no_known_price(symbol_frame):
    """全为缺失值的窗口不产生信号，也不被当成「低于现值」而放行（缺口纪律）。"""
    nan = float("nan")
    prices = symbol_frame({"sh600000": [nan, nan, nan, nan]})

    got = new_high(prices, n=3)

    assert got["sh600000"].tolist() == [False, False, False, False]


def test_one_missing_bar_inside_the_window_suppresses_the_signal(symbol_frame):
    """窗口里**有一根**缺失即不足以判定，故不给信号。

    这一条比「全为缺失」更紧：它同时挡住「把缺失填成 0」与「把缺失跳过、用其余 n-1 根
    凑一个最大值」两种写法——后者会把停牌期当成「没有更高的价」，从而凭空造出新高。
    """
    nan = float("nan")
    prices = symbol_frame({"sh600000": [10.0, nan, nan, 20.0]})

    got = new_high(prices, n=3)["sh600000"]

    assert got.tolist() == [False, False, False, False]


def test_above_ma_compares_the_close_with_its_own_trailing_average(symbol_frame):
    """站上均线：当根收盘高于 n 日均线；窗口未满时不给答案。"""
    prices = symbol_frame({"sh600000": [10.0, 11.0, 12.0, 11.0]})

    got = above_ma(prices, n=2)["sh600000"]

    assert got.tolist() == [False, True, True, False]


def test_ma_cross_up_fires_on_the_crossing_bar_only(symbol_frame):
    """均线上穿是**事件**而非状态：越过的那一根为真，其后继续在上方则为假。"""
    prices = symbol_frame({"sh600000": [10.0, 9.0, 8.0, 9.5, 9.6]})

    got = ma_cross_up(prices, n=2)["sh600000"]

    assert got.tolist() == [False, False, False, True, False]


def test_rising_streak_needs_n_increases_hence_n_plus_one_bars(symbol_frame):
    """连续上涨 n 日 = 最近 n 个交易日**每一次**收盘都高于前一日，故需要 n+1 根 K 线。

    取 n=3、六根 K 线：[1,2,3,4,4,5] 的日间变动是 [—,↑,↑,↑,平,↑]。第一根合格的
    K 线在第 4 根（索引 3，"1→4" 那一段），此前不足以判定，索引 4 因持平而中断。
    """
    prices = symbol_frame({"sh600000": [1.0, 2.0, 3.0, 4.0, 4.0, 5.0]})

    got = rising_streak(prices, n=3)["sh600000"]

    assert got.tolist() == [False, False, False, True, False, False]


def test_volume_surge_is_strict_so_exactly_k_times_the_average_does_not_count(symbol_frame):
    """放量 k 倍：当根量必须**严格高于**前 n 日均量的 k 倍——恰好 k 倍不算。

    第 4 根 200 恰为前 3 日均量 100 的 2 倍，故为假；第 5 根 300 高于其基准
    mean(100,100,200)=133.3 的 2 倍（266.7），故为真。
    """
    volumes = symbol_frame({"sh600000": [100.0, 100.0, 100.0, 200.0, 300.0]})

    got = volume_surge(volumes, k=2.0, n=3)["sh600000"]

    assert got.tolist() == [False, False, False, False, True]


def test_volume_surge_uses_the_prior_average_not_one_including_the_current_bar(symbol_frame):
    """基准取自**前 n 根**：含当根会把当根的放量本身算进基准，放量越猛越难触发。"""
    volumes = symbol_frame({"sh600000": [100.0, 100.0, 100.0, 400.0]})

    got = volume_surge(volumes, k=3.0, n=3)["sh600000"]

    # 400 > 3 × 100 → 真。若基准含当根则为 mean(100,100,400)=200，400 > 600 为假。
    assert got.tolist() == [False, False, False, True]


def test_filters_return_plain_bools_never_missing(symbol_frame):
    """过滤器输出纯 ``bool``：缺失被解释为「不合格」，不是「未知」。"""
    nan = float("nan")
    prices = symbol_frame({"sh600000": [nan, nan, 12.0, 13.0], "sz000001": [1.0, 2.0, 3.0, 4.0]})

    got = above_ma(prices, n=2)

    assert list(got.columns) == ["sh600000", "sz000001"]
    assert got.index.equals(prices.index)
    assert all(dtype.kind == "b" for dtype in got.dtypes)
    assert bool(got.loc[got.index[0], "sz000001"]) is False
    assert bool(got.loc[got.index[0], "sh600000"]) is False


def test_a_filter_rejects_a_non_monotonic_index_like_the_indicators_do(symbol_frame):
    """契约由所有公开函数共同遵守，过滤器也不例外。"""
    idx = pd.to_datetime(["2024-01-03", "2024-01-02", "2024-01-04"])
    prices = pd.DataFrame({"sh600000": [10.0, 11.0, 12.0]}, index=idx)

    with pytest.raises(ValueError, match="升序"):
        new_high(prices, n=2)


# --- 白线在黄线上、J 值偏低（提示 2 与 3 的过滤器） -----------------------------


def j_panel(panel):
    """手算样本（n=3）：RSV = [缺失, 缺失, 2/3, 1/3, 1]。

    故 J = [缺失, 缺失, 200/3, 1100/27, 6500/81]，即 ≈ [—, —, 66.67, 40.74, 80.25]。
    与 ``test_signal_indicators.py`` 里那份是同一个样本，两处各自手算、不共享 fixture。
    """
    return panel(
        {
            "high": {"sh600000": [10.0, 11.0, 12.0, 11.0, 12.0]},
            "low": {"sh600000": [9.0, 9.0, 10.0, 10.0, 11.0]},
            "close": {"sh600000": [10.0, 10.0, 11.0, 10.0, 12.0]},
        }
    )


def test_white_above_yellow_is_true_in_an_uptrend_and_false_in_a_downtrend(symbol_frame):
    """上升趋势里快线在慢线上方，下降趋势里反之。

    这不是同义反复：对**线性**序列可以手算两条线的滞后——单层 EMA(n=10) 滞后 (n-1)/2=4.5 根，
    双层滞后 9.0 根；黄线四条的滞后是 (14+28+57+114)/4-1)/... 即 26.125 根。故上涨时白线
    （价格-9.0）高于黄线（价格-26.125），下跌时相反。这条能区分「白线是否真的被算成双层」。
    """
    up = symbol_frame({"sh600000": [100.0 + i for i in range(200)]})
    down = symbol_frame({"sh600000": [300.0 - i for i in range(200)]})

    assert (
        bool(white_above_yellow(up, n=10, windows=(14, 28, 57, 114))["sh600000"].iloc[-1]) is True
    )
    assert (
        bool(white_above_yellow(down, n=10, windows=(14, 28, 57, 114))["sh600000"].iloc[-1])
        is False
    )


def test_white_above_yellow_never_fires_before_the_longest_window_is_full(symbol_frame):
    """黄线含 114 日均线，故只有 60 根 K 线的次新股一律为 False——不因为白线算得出就放行。"""
    prices = symbol_frame({"sh600000": [100.0 + i for i in range(60)]})

    got = white_above_yellow(prices, n=10, windows=(14, 28, 57, 114))

    assert got["sh600000"].tolist() == [False] * 60


def test_j_below_flags_only_the_bars_whose_j_is_below_the_threshold(panel):
    """逐值钉住：J = [缺失, 缺失, 200/3, 1100/27, 6500/81] ≈ [—, —, 66.7, 40.7, 80.2]，
    故阈值 50 只在第 4 根为真。窗口不满的两根判为 ``False``（不合格），不是「未知」。
    """
    got = j_below(j_panel(panel), threshold=50.0, n=3, m1=3, m2=3)

    assert got["sh600000"].tolist() == [False, False, False, True, False]
    assert got.dtypes.tolist() == ["bool"]


def test_j_below_is_strict_so_exactly_the_threshold_does_not_count(panel):
    """恰好落在阈值上不算——与 :func:`new_high`、:func:`volume_surge` 的严格比较一致。

    样本取 ``n=m1=m2=1`` 且收盘等于窗口最高价，此时 RSV=K=D=J=100（整数、可精确比较）。
    """
    at_the_top = panel(
        {
            "high": {"sh600000": [10.0, 10.0]},
            "low": {"sh600000": [8.0, 8.0]},
            "close": {"sh600000": [10.0, 10.0]},
        }
    )

    assert j_below(at_the_top, threshold=100.0, n=1, m1=1, m2=1)["sh600000"].tolist() == [
        False,
        False,
    ]
    assert j_below(at_the_top, threshold=100.5, n=1, m1=1, m2=1)["sh600000"].tolist() == [
        True,
        True,
    ]


def test_j_below_is_false_where_the_window_has_no_range(panel):
    """窗口内无波动时 RSV 无定义，故「J 很低」这种判断不给答案——判为不合格。"""
    flat = panel(
        {
            "high": {"sh600000": [5.0, 5.0, 5.0, 5.0]},
            "low": {"sh600000": [5.0, 5.0, 5.0, 5.0]},
            "close": {"sh600000": [5.0, 5.0, 5.0, 5.0]},
        }
    )

    got = j_below(flat, threshold=0.0, n=3, m1=3, m2=3)

    assert got["sh600000"].tolist() == [False, False, False, False]


def test_below_yellow_streak_counts_consecutive_breaks_of_a_slow_line(symbol_frame):
    """用一条真正滞后的黄线来测「连续」。

    样本收盘 ``[10, 12, 20, 18, 16, 13, 11, 9]``，黄线取 ``MA(3)``：
    ``[缺失, 缺失, 14, 50/3, 18, 47/3, 40/3, 11]`` ≈ ``[—, —, 14, 16.67, 18, 15.67, 13.33, 11]``。
    故「收盘 < 黄线」= ``[F, F, F, F, T, T, T, T]``，连续 2 根要等到索引 5 才成立
    （索引 4 只是第一根跌破）。
    """
    prices = symbol_frame({"sh600000": [10.0, 12.0, 20.0, 18.0, 16.0, 13.0, 11.0, 9.0]})

    got = below_yellow_streak(prices, n=1, windows=(3,), days=2)["sh600000"]

    assert got.tolist() == [False, False, False, False, False, True, True, True]


def test_below_yellow_streak_is_strict_about_the_line(symbol_frame):
    """恰好等于黄线**不算**跌破——与 :func:`new_high`、:func:`volume_surge` 的严格比较一致。"""
    prices = symbol_frame({"sh600000": [10.0, 10.0, 10.0, 10.0]})

    got = below_yellow_streak(prices, n=1, windows=(2,), days=2)["sh600000"]

    assert got.tolist() == [False, False, False, False]


def test_below_yellow_streak_never_fires_before_the_yellow_line_exists(symbol_frame):
    """黄线窗口不满处为缺失，比较即假，故连续成立需要 ``days`` 根**可用**的黄线。"""
    prices = symbol_frame({"sh600000": [10.0, 9.0, 8.0, 7.0]})

    got = below_yellow_streak(prices, n=1, windows=(3,), days=2)["sh600000"]

    assert got.iloc[:4].tolist() == [False, False, False, True]


def test_below_yellow_streak_with_days_one_is_just_the_single_bar_break(symbol_frame):
    """``days=1`` 退化为「当根跌破」这一状态，与 :func:`white_above_yellow` 互补。"""
    prices = symbol_frame({"sh600000": [10.0, 12.0, 20.0, 18.0, 16.0, 13.0]})

    got = below_yellow_streak(prices, n=1, windows=(3,), days=1)["sh600000"]

    assert got.tolist() == [False, False, False, False, True, True]


# --- 「一波上涨之后的下跌阶段」（提示 3 的位置条件） ---------------------------


def pullback_frame(symbol_frame):
    """一段可手算的涨跌：8 →(涨到) 12 →(回撤到) 10。

    阈值 10% 下：低点 8 在索引 2 确认、峰值 12 在索引 6 确认（索引 7 的 10.5 回撤了 12.5%）。
    故索引 7 起可判：上涨幅度 = 12/8 − 1 = 50%。
    """
    return symbol_frame({"sh600000": [10.0, 9.0, 8.0, 9.0, 10.0, 11.0, 12.0, 10.5]})


def test_pullback_after_advance_is_true_inside_a_measured_pullback(symbol_frame):
    """四个条件全成立时为真：涨 50%、回调 12.5%(=10.5/12−1)、峰值距今 1 根。"""
    prices = pullback_frame(symbol_frame)

    got = pullback_after_advance(
        prices,
        retracement=0.10,
        min_advance=0.30,
        min_drop=0.08,
        max_drop=0.35,
        min_peak_age=1,
        max_peak_age=10,
    )["sh600000"]

    assert bool(got.iloc[7]) is True


def test_pullback_after_advance_requires_each_bound_and_the_misses_are_narrow(symbol_frame):
    """四条边界各自单独越一次，确认真的是**合取**而不是某一条在主导。

    同一个样本（索引 7 处：涨幅 50%、回调 12.5%、峰值距今 1 根），只改动一条边界：
    要求涨幅 ≥ 60% / 回调 ≥ 20% / 回调 ≤ 10% / 峰值距今 ≥ 2 根，四者都应让它变假。
    """
    prices = pullback_frame(symbol_frame)
    base = dict(
        retracement=0.10,
        min_advance=0.30,
        min_drop=0.08,
        max_drop=0.35,
        min_peak_age=1,
        max_peak_age=10,
    )

    assert (
        bool(pullback_after_advance(prices, **{**base, "min_advance": 0.60})["sh600000"].iloc[7])
        is False
    )
    assert (
        bool(pullback_after_advance(prices, **{**base, "min_drop": 0.20})["sh600000"].iloc[7])
        is False
    )
    assert (
        bool(pullback_after_advance(prices, **{**base, "max_drop": 0.10})["sh600000"].iloc[7])
        is False
    )
    assert (
        bool(pullback_after_advance(prices, **{**base, "min_peak_age": 2})["sh600000"].iloc[7])
        is False
    )
    assert (
        bool(pullback_after_advance(prices, **{**base, "max_peak_age": 0})["sh600000"].iloc[7])
        is False
    )


def test_pullback_after_advance_is_false_while_the_peak_is_still_unconfirmed(symbol_frame):
    """拐点未确认处一律为假——不因为「显然是个高点」就放行（不给乐观答案）。

    样本与上面同源，但砍掉最后一根：那时回撤还没发生，峰值 12 **尚不可知**。
    """
    prices = symbol_frame({"sh600000": [10.0, 9.0, 8.0, 9.0, 10.0, 11.0, 12.0]})

    got = pullback_after_advance(
        prices,
        retracement=0.10,
        min_advance=0.30,
        min_drop=0.01,
        max_drop=0.90,
        min_peak_age=0,
        max_peak_age=999,
    )

    assert got["sh600000"].tolist() == [False] * 7


def test_pullback_after_advance_is_false_while_the_price_is_at_new_highs(symbol_frame):
    """回调深度为 0（还在新高）时不算「下跌阶段」——`min_drop` 是必须的。"""
    prices = pullback_frame(symbol_frame)

    got = pullback_after_advance(
        prices,
        retracement=0.10,
        min_advance=0.30,
        min_drop=0.01,
        max_drop=0.35,
        min_peak_age=0,
        max_peak_age=10,
    )["sh600000"]

    # 峰值确认当根（索引 7）收盘 10.5 已在峰值下方 12.5%，故为真；
    # 但若把回调深度要求提到「必须回调」之上就成了假——这里用 min_drop=0.13 验证。
    assert bool(got.iloc[7]) is True
    assert (
        bool(
            pullback_after_advance(
                prices,
                retracement=0.10,
                min_advance=0.30,
                min_drop=0.13,
                max_drop=0.35,
                min_peak_age=0,
                max_peak_age=10,
            )["sh600000"].iloc[7]
        )
        is False
    )


def test_pullback_after_advance_returns_plain_bools(symbol_frame):
    """过滤器输出纯 ``bool``：缺失被解释为「不合格」，不是「未知」。"""
    prices = pullback_frame(symbol_frame)
    prices["sz000001"] = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]

    got = pullback_after_advance(
        prices,
        retracement=0.10,
        min_advance=0.30,
        min_drop=0.05,
        max_drop=0.40,
        min_peak_age=1,
        max_peak_age=20,
    )

    assert all(dtype.kind == "b" for dtype in got.dtypes)
    assert got["sz000001"].tolist() == [False] * 8


# --- 白线两侧：卖出规则用的两个判据（提示：8% 减半 / 跌破清仓） -----------------


def test_above_white_uses_the_margin_and_is_strict_at_the_threshold(symbol_frame):
    """``C > 白线 × (1 + margin)``，且**严格**：恰好等于门槛不算。

    用一条平线让白线等于收盘价本身（白线 = EMA(EMA(C))，平坦序列上恒等于 C），于是
    ``margin=0`` 时比较退化为 ``C > C``——恒假。这条同时把「严格」与「白线口径」钉住。
    再让最后一根跳高，检查 margin 的门槛位置。
    """
    flat = symbol_frame({"sh600000": [10.0, 10.0, 10.0, 10.0]})
    assert above_white(flat, n=2, margin=0.0)["sh600000"].tolist() == [False] * 4

    rising = symbol_frame({"sh600000": [10.0, 10.0, 10.0, 11.0]})
    got = above_white(rising, n=2, margin=0.0)["sh600000"]
    above = white_line(rising, n=2)["sh600000"]
    assert bool(got.iloc[3]) is (11.0 > float(above.iloc[3]))

    # 门槛定价：要求「高于白线 8%」，则 11.0 是否通过取决于白线 × 1.08。
    assert bool(above_white(rising, n=2, margin=0.08)["sh600000"].iloc[3]) is (
        11.0 > float(above.iloc[3]) * 1.08
    )


def test_above_white_has_no_warm_up_window_unlike_the_yellow_line(symbol_frame):
    """白线**从第一根起就有值**：``ema`` 以首根播种，没有「窗口填满」这一刻。

    故 ``above_white`` / ``below_white`` 不存在「窗口不足 → 取假」的那一段——它们在序列
    开头就能成立。这与黄线那一侧**不对称**（``below_yellow_streak`` 要等 114 日均线填满），
    而这个不对称有实际后果：建仓后「站上白线」可以立刻武装趋势离场，而「连续跌破黄线」的
    止损在建仓初期根本不可能触发。故意保留为不对称，是因为两者定义不同——把它抹平才是错的。
    """
    rising = symbol_frame({"sh600000": [10.0, 11.0, 12.0]})
    falling = symbol_frame({"sh600000": [10.0, 9.0, 8.0]})

    assert above_white(rising, n=10, margin=0.0)["sh600000"].tolist() == [False, True, True]
    assert below_white(falling, n=10)["sh600000"].tolist() == [False, True, True]


def test_below_yellow_streak_does_have_a_warm_up_window_by_contrast(symbol_frame):
    """对照：黄线用 ``sma``（``min_periods``），故那条止损在窗口填满之前**不可能**触发。

    这条与上一条合起来说明：同一套卖出规则里，趋势离场从第一天就可武装，而止损要等黄线
    的历史攒够。若哪天有人把两条线的口径改成一致，这两条会同时变红。
    """
    falling = symbol_frame({"sh600000": [10.0, 9.0, 8.0, 7.0, 6.0, 5.0]})

    got = below_yellow_streak(falling, n=10, windows=(5,), days=2)

    assert got["sh600000"].tolist() == [False, False, False, False, False, True]


def test_above_white_rejects_a_margin_that_makes_the_threshold_non_positive(symbol_frame):
    """``margin <= -1`` 会让门槛价非正、比较恒真——那不是放松条件而是条件消失，故报错。"""
    prices = symbol_frame({"sh600000": [10.0, 11.0]})

    for bad in (-1.0, -1.5):
        with pytest.raises(ValueError, match="margin"):
            above_white(prices, n=2, margin=bad)


def test_below_white_is_the_plain_strict_break(symbol_frame):
    """``C < 白线``，严格：恰好收在白线上不算跌破。

    平线序列上白线恒等于 C，故恒假——这恰好是「恰好等于不算」的极端情形。
    """
    flat = symbol_frame({"sh600000": [10.0, 10.0, 10.0]})
    assert below_white(flat, n=2)["sh600000"].tolist() == [False, False, False]

    falling = symbol_frame({"sh600000": [10.0, 10.0, 10.0, 9.0]})
    got = below_white(falling, n=2)["sh600000"]
    below = white_line(falling, n=2)["sh600000"]
    assert bool(got.iloc[3]) is (9.0 < float(below.iloc[3]))


def test_above_white_and_below_white_partition_the_line_except_at_equality(symbol_frame):
    """两者互补，**只差恰好相等那一点**——它同时为假，故不重不漏。

    这条把「严格」这件事的后果写清楚：恰好收在白线上时两条都不成立，于是趋势离场不会
    在那一根触发（要等真正跌破）。
    """
    prices = symbol_frame({"sh600000": [10.0, 10.0, 10.0, 10.0, 11.0, 9.0]})

    above = above_white(prices, n=2, margin=0.0)
    below = below_white(prices, n=2)

    both = (above & below).to_numpy().any()
    assert not both, "两条不可能同时成立"


def test_above_yellow_is_strict_and_missing_safe(symbol_frame):
    """收盘**严格**高于黄线：恰好等于不算；窗口不足处取假。

    用 windows=(1,) 让黄线等于当日收盘价，则比较退化为 ``C > C``——恒假。这恰好把
    「恰好等于不算」钉成极端情形。再让最后一根跳高，检查它能成立。
    """
    flat = symbol_frame({"sh600000": [10.0, 10.0, 10.0]})
    assert above_yellow(flat, windows=(1,))["sh600000"].tolist() == [False, False, False]

    rising = symbol_frame({"sh600000": [10.0, 10.0, 10.0, 11.0]})
    got = above_yellow(rising, windows=(3,))["sh600000"]
    line = yellow_line(rising, windows=(3,))["sh600000"]
    assert pd.isna(line.iloc[0]) and pd.isna(line.iloc[1])
    assert got.iloc[:2].tolist() == [False, False], "窗口不足处应当是假（黄线缺失）"
    assert bool(got.iloc[-1]) is (11.0 > float(line.iloc[-1]))


def test_above_yellow_is_not_implied_by_white_above_yellow(symbol_frame):
    """两条**独立**：白线在黄线上，不代表收盘也在黄线上。

    构造一段**长期缓涨**（约 120 根从 10 涨到 30）之后**一根急跌**到 20：

    - 黄线含 114 日均线、滞后约 26 根，缓涨时它稳稳在价格**下方**，故白线（滞后约 9 根）
      在它上方——``white_above_yellow`` 成立；
    - 但那根急跌让收盘直接落到黄线**下方**，而均线还没跟上——``above_yellow`` 不成立。

    若哪天有人把 ``above_yellow`` 写成 ``white_above_yellow`` 的别名，或者以为前者蕴含
    后者而删掉一条，这条会变红。
    """
    rise = [10.0 + 20.0 * i / 119.0 for i in range(120)]
    prices = symbol_frame({"sh600000": [*rise, 20.0]})
    windows = (14, 28, 57, 114)

    white_over = white_above_yellow(prices, n=10, windows=windows)
    close_over = above_yellow(prices, windows=windows)

    last = prices.index[-1]
    line = float(yellow_line(prices, windows=windows).loc[last, "sh600000"])
    assert float(prices.loc[last, "sh600000"]) < line, "构造前提：收盘应在黄线下方"
    assert bool(white_over["sh600000"].iloc[-1]) is True, "白线仍应在黄线上方（慢线没跟上）"
    assert bool(close_over["sh600000"].iloc[-1]) is False, "收盘已跌到黄线下方"


def test_above_yellow_is_the_single_bar_counterpart_of_below_yellow_streak(symbol_frame):
    """``above_yellow`` 是 :func:`below_yellow_streak` 的**单根**对照，两者互为反面。

    「连续 days 根在下」成立时，当根必然在下方，故 ``above_yellow`` 必假。反过来不成立：
    当根在上方时，前面几根仍可能在下方（days 根连续要求整段）。
    """
    falling = symbol_frame({"sh600000": [12.0, 11.0, 10.0, 9.0, 8.0, 7.0]})

    above = above_yellow(falling, windows=(3,))
    streak = below_yellow_streak(falling, n=2, windows=(3,), days=2)

    both = (above & streak).to_numpy().any()
    assert not both, "同一根不可能既在黄线上方、又连续两根在下方"


def test_above_yellow_rejects_a_non_monotonic_index(symbol_frame):
    """契约由所有公开函数共同遵守。"""
    idx = pd.to_datetime(["2024-01-03", "2024-01-02", "2024-01-04"])
    prices = pd.DataFrame({"sh600000": [10.0, 11.0, 12.0]}, index=idx)

    with pytest.raises(ValueError, match="升序"):
        above_yellow(prices, windows=(2,))


def test_high_above_white_uses_the_high_not_the_close(panel):
    """与 :func:`above_white` 的分水岭：最高价摸到白线而收盘又落回下方时，本条为真、那条为假。

    构造：前三根收 10.0（白线 = 10.0），第四根的**最高价** 11.0 冲过白线、**收盘**回到 9.5。
    故 ``high_above_white`` 为真而 ``above_white`` 为假——两类信号各答一个问题。
    """
    closes = [10.0, 10.0, 10.0, 9.5]
    bars = panel(
        {
            "high": {"sh600000": [10.0, 10.0, 10.0, 11.0]},
            "low": {"sh600000": [9.0, 9.0, 9.0, 9.0]},
            "close": {"sh600000": closes},
        }
    )
    close_frame = pd.DataFrame({"sh600000": closes}, index=bars["close"].index)

    touched = high_above_white(bars, n=2)["sh600000"]
    closed_above = above_white(close_frame, n=2, margin=0.0)["sh600000"]

    assert bool(touched.iloc[-1]) is True, "最高价冲过白线，本条应为真"
    assert bool(closed_above.iloc[-1]) is False, "收盘落在白线下方，那条应为假"


def test_high_above_white_is_strict_at_the_line(panel):
    """最高价**恰好等于**白线不算破——与 :func:`new_high` 的严格比较一致。

    平坦序列上白线恒等于该常数，故最高价（同值）与它相等，应当取假。
    """
    flat = panel(
        {
            "high": {"sh600000": [10.0, 10.0, 10.0]},
            "low": {"sh600000": [10.0, 10.0, 10.0]},
            "close": {"sh600000": [10.0, 10.0, 10.0]},
        }
    )

    got = high_above_white(flat, n=2)["sh600000"]

    assert got.tolist() == [False, False, False]


def test_high_above_white_takes_false_where_the_high_is_missing(panel):
    """``high`` 缺失（停牌）处取假——「摸到过白线」是一个判断，不是默认状态。

    注意白线**没有预热窗口**（``ema`` 以首根播种，见 :func:`~mbt.signals.indicators.ema`），
    故第 0、1 根照常有值、照常可比；只有 ``high`` 本身缺失的那一根取假。
    """
    nan = float("nan")
    bars = panel(
        {
            "high": {"sh600000": [10.0, 11.0, nan]},
            "low": {"sh600000": [9.0, 9.5, nan]},
            "close": {"sh600000": [9.5, 10.5, 12.0]},
        }
    )

    got = high_above_white(bars, n=10)["sh600000"]

    assert got.tolist() == [True, True, False], "只有 high 缺失那一根取假"


def test_high_above_white_rejects_a_non_monotonic_index():
    """契约由所有公开函数共同遵守。"""
    from mbt.data import Panel

    idx = pd.to_datetime(["2024-01-03", "2024-01-02", "2024-01-04"])
    frame = pd.DataFrame({"sh600000": [10.0, 11.0, 12.0]}, index=idx)

    with pytest.raises(ValueError, match="升序"):
        high_above_white(Panel({"high": frame, "low": frame, "close": frame}), n=2)


# --- 调整期内不得有「连续 days 根内含」 ----------------------------------------


def contained_panel(panel, flags, symbol="sh600000", count=20):
    """造一个「内含」标志完全可控的面板。

    手法：把每根的高/低固定为 100 / 90，于是「收盘落在前一根区间内」等价于
    ``90 <= 收盘 <= 100``。故 ``flags[t]=True`` 时收盘给 95、``False`` 时给 105
    （上破前一根的最高价）——这样逐根的内含标志与 ``flags`` 一致，便于手算。
    """
    high = [100.0] * count
    low = [90.0] * count
    close = [95.0 if flag else 105.0 for flag in flags]
    return panel({"high": {symbol: high}, "low": {symbol: low}, "close": {symbol: close}})


def anchors_with_peak(symbol="sh600000", peak_at=2, count=20):
    """手造摆动点：峰值固定在 ``peak_at``，故第 ``t`` 根的 ``peak_age = t - peak_at``。

    第 ``peak_at`` 根之前为缺失（那时峰值尚未确认）——这正是本过滤器「没有调整期就不给
    乐观答案」那条分界。手造而不调 :func:`swings`，是为了让段起点完全可控、落点可手算。
    """
    from mbt.signals import Swings

    index = pd.bdate_range("2024-01-02", periods=count)
    age = [float("nan")] * peak_at + [float(t - peak_at) for t in range(peak_at, count)]
    blank = pd.DataFrame({symbol: [float("nan")] * count}, index=index, dtype=float)
    return Swings(
        peak_price=blank.copy(),
        peak_age=pd.DataFrame({symbol: age}, index=index, dtype=float),
        trough_price=blank.copy(),
        trough_age=blank.copy(),
    )


def test_no_contained_run_says_qualified_when_the_pullback_has_no_such_stretch(panel):
    """**取向**：它的 True 表示「合格」= 调整期内**没有**连续 ``days`` 根内含。

    这一条单独存在是因为取向写反过一次：函数名叫 ``no_...`` 而实现返回了「存在这样的串」，
    于是单测与实现一起错、彼此印证，谁也没抓住它——直到它在真实行情上把所有格子都挡掉
    （夹具上 134~140 全是 False）才暴露。故这条断言只做最直白的一件事：给一串**没有**横盘的
    标志，整列必须为真。
    """
    flags = [True, False] * 10  # 内含与不内含交替，最长连续只有 1 根
    got = no_contained_run(contained_panel(panel, flags), anchors_with_peak(peak_at=2), days=3)[
        "sh600000"
    ]

    assert got.iloc[3:].all(), "没有三连横盘时，应当是「合格」"


def test_no_contained_run_rejects_the_bar_once_the_window_fits_in_the_pullback(panel):
    """手算锁定边界：窗口必须**完整落在调整期内**，早一根就不算。

    峰值在索引 2，故调整期从索引 3 起。**内含标志**在索引 2~5 为真（``contained[t]`` 看的是
    ``close[t]``，故标志按索引给），取 ``days=3``：

    ======  ==========  ====================  ======================
    第几根   调整期      最近一个合格窗口      合格？
    ======  ==========  ====================  ======================
    3        [3, 3]      无（放不下窗口）       是 ← 确凿没有，不是猜
    4        [3, 4]      [2, 4]（左端 2）       是 ← 窗口跨出了调整期
    5        [3, 5]      [3, 5]（左端 3）       否
    6        [3, 6]      [3, 5]（左端 3）       否 ← 已出现的窗口仍在期内
    7        [3, 7]      [3, 5]（左端 3）       否
    ======  ==========  ====================  ======================

    索引 4 那一格是关键：合格的窗口确实存在（索引 2~4 都内含），但它的**左端在调整期之前**，
    故「调整期内连续 3 根」不成立——漏掉这个左端检查的实现会在这里误判为不合格。
    """
    flags = [False, False, True, True, True, True] + [False] * 14
    got = no_contained_run(contained_panel(panel, flags), anchors_with_peak(peak_at=2), days=3)[
        "sh600000"
    ]

    assert bool(got.iloc[3]) is True, "调整期只有一根，装不下窗口——确凿合格"
    assert bool(got.iloc[4]) is True, "窗口 [2,4] 跨出了调整期，不算"
    assert bool(got.iloc[5]) is False, "窗口 [3,5] 完整落在期内，不合格"
    assert bool(got.iloc[6]) is False, "已出现的窗口仍在期内，状态保持"
    assert bool(got.iloc[7]) is False


def test_no_contained_run_is_unqualified_before_the_peak_is_confirmed(panel):
    """峰值未确认处**不合格**——没有调整期就无从谈「期内」。

    与其余过滤器「缺失即不合格」同一契约（ADR-0005）：宁可漏判，也不凭空造出合格。
    注意这一条与「调整期过短则合格」是**两回事**：前者是无从判断，后者是确凿成立。
    """
    got = no_contained_run(
        contained_panel(panel, [True] * 20), anchors_with_peak(peak_at=2), days=3
    )["sh600000"]

    assert got.iloc[:2].tolist() == [False, False], "峰值确认之前不该有答案"
    assert bool(got.iloc[2]) is True, "峰值那一根的调整期是空集，空真成立"
    assert bool(got.iloc[3]) is True, "调整期 [3,3] 装不下窗口，确凿合格"
    assert bool(got.iloc[4]) is True, "调整期 [3,4] 只有两根，仍装不下"
    assert bool(got.iloc[5]) is False, "调整期到 [3,5] 就装得下且确实横盘了三根"


def test_no_contained_run_needs_consecutive_days_not_a_total(panel):
    """要的是**连续**，不是累计。

    标志取 ``[T,F,T,T,F]`` 重复四遍：真值不少（每五根有两处），但连续的不超过 2 根，
    故 ``days=3`` 在峰值确认之后整列**合格**。若实现误用「累计和 >= days」，这一列会变红。
    """
    flags = ([True, False, True, True, False] * 4)[:20]
    got = no_contained_run(contained_panel(panel, flags), anchors_with_peak(peak_at=2), days=3)[
        "sh600000"
    ]

    assert got.iloc[:2].tolist() == [False, False], "峰值确认之前不该有答案"
    assert got.iloc[2:].all(), "没有任何 3 连，确认之后应当全部合格"


def test_no_contained_run_forgets_a_run_that_belongs_to_the_previous_pullback(panel):
    """新峰值把调整期起点往前推之后，旧的那一段不再计入——由不合格翻回合格。

    构造：内含在索引 2~5 为真，峰值先在索引 2（调整期从 3 起），故索引 5 处凑满窗口
    ``[3,5]`` → 不合格。随后峰值改到索引 8（调整期从 9 起），旧窗口左端 3 已落在本期之外，
    且后面没有新的三连，故索引 9 起应当**合格**。
    """
    flags = [False, False, True, True, True, True] + [False] * 14
    count = 20
    index = pd.bdate_range("2024-01-02", periods=count)
    age = [float("nan")] * count
    for t in range(2, 8):
        age[t] = float(t - 2)
    for t in range(8, count):
        age[t] = float(t - 8)
    blank = pd.DataFrame({"sh600000": [float("nan")] * count}, index=index, dtype=float)

    from mbt.signals import Swings

    anchors = Swings(
        peak_price=blank.copy(),
        peak_age=pd.DataFrame({"sh600000": age}, index=index, dtype=float),
        trough_price=blank.copy(),
        trough_age=blank.copy(),
    )

    got = no_contained_run(contained_panel(panel, flags), anchors, days=3)["sh600000"]

    assert bool(got.iloc[5]) is False, "窗口 [3,5] 落在峰值 2 的调整期内"
    assert bool(got.iloc[9]) is True, "峰值改到 8 之后，旧窗口不再属于当期"


def test_no_contained_run_rejects_a_non_positive_window(panel):
    """``days`` 至少为 1：0 会让「窗口和 == 0」恒真，那不是放松条件而是条件消失。"""
    with pytest.raises(ValueError, match="至少为 1"):
        no_contained_run(contained_panel(panel, [True] * 20), anchors_with_peak(), days=0)


def test_no_contained_run_rejects_a_panel_that_does_not_match_the_anchors(panel):
    """面板与摆动点必须同日同标的——对不上会把段边界对到别的日子上，而那种错不报错。"""
    from mbt.data import Panel

    bars = contained_panel(panel, [True] * 20)
    short = Panel(
        {
            "high": bars["high"].iloc[:10],
            "low": bars["low"].iloc[:10],
            "close": bars["close"].iloc[:10],
        }
    )

    with pytest.raises(ValueError, match="索引"):
        no_contained_run(short, anchors_with_peak(count=20), days=3)


# --- 收盘高于前 n 根的**最高价**（跨字段） --------------------------------------
#
# 与 :func:`new_high` 的分水岭：那个比的是**最高收盘价**，这个比的是**最高价**（`high`）。
# 两者的第一个把门宽度不同——盘中摸到过的高点总是 ≥ 收出来过的高点，故本条更严。


def _high_close_panel(panel, highs, closes, symbol="sh600000"):
    """只给 ``high`` 与 ``close`` 的最小面板；``low`` 由两者取下沿，免得越界。"""
    return panel(
        {
            "high": {symbol: highs},
            "low": {symbol: [min(h, c) * 0.99 for h, c in zip(highs, closes, strict=True)]},
            "close": {symbol: closes},
        }
    )


def test_close_above_high_uses_the_high_not_the_prior_high_close(panel):
    """基准是**最高价**（`high`），不是最高收盘价——这条把它与 :func:`new_high` 分开。

    六根：前五根的收盘都低于 10.0，但第四根的**最高价**摸到 12.0。第六根收盘 11.0：
    它高于任何一根的**收盘**（故 ``new_high`` 为真），却低于前五根里的最高价 12.0
    （故本条为**假**）。两者在同一个格子上给出相反答案，正是要分开的理由。
    """
    highs = [10.0, 10.5, 10.2, 12.0, 10.4, 11.0]
    closes = [9.0, 9.5, 9.2, 9.8, 9.4, 11.0]
    bars = _high_close_panel(panel, highs, closes)
    close_frame = pd.DataFrame({"sh600000": closes}, index=bars["close"].index)

    above_high = close_above_high(bars, n=5)["sh600000"]
    makes_new_high = new_high(close_frame, n=5)["sh600000"]

    assert bool(above_high.iloc[-1]) is False, "11.0 低于前五根的最高价 12.0，本条该为假"
    assert bool(makes_new_high.iloc[-1]) is True, "11.0 高于前五根的最高收盘 9.8，那条该为真"


def test_close_above_high_excludes_the_current_bar(panel):
    """窗口取**前 n 根**（不含当根）——含了它就是一句恒假的废话。

    ``HHV(H, n)`` 至少等于当根的 ``high``，而 ``close ≤ high`` 恒成立，故「收盘高于含当根的
    n 日最高价」**永远为假**。含它写错不会报错，只会把每一只都筛掉。
    """
    # 当根自己就是最高价（收 20.0、高 20.5）；前两根的最高价只有 10.5。
    bars = _high_close_panel(panel, [10.0, 10.5, 20.5], [9.5, 10.0, 20.0])

    got = close_above_high(bars, n=2)["sh600000"]

    assert bool(got.iloc[-1]) is True, "20.0 > 前两根的最高价 10.5——若把当根算进窗口就恒假了"


def test_close_above_high_is_strict_at_the_prior_high(panel):
    """**恰好等于**前 n 根的最高价不算「高于」——与 :func:`new_high` 的严格比较一致。"""
    bars = _high_close_panel(panel, [10.0, 10.0, 10.0, 10.0], [9.0, 9.0, 10.0, 10.0])

    got = close_above_high(bars, n=3)["sh600000"]

    assert bool(got.iloc[3]) is False, "收盘恰好等于前高，不算高于"


def test_close_above_high_stays_false_while_the_window_is_not_full(panel):
    """窗口不满 n 根处取假——不足 n 根就无从谈「n 根内的最高价」。"""
    bars = _high_close_panel(panel, [10.0, 11.0, 12.0, 13.0], [9.0, 10.0, 11.0, 12.5])

    got = close_above_high(bars, n=3)["sh600000"]

    assert got.tolist()[:3] == [False, False, False], "前 3 根窗口不满"
    assert bool(got.iloc[3]) is True, "第 4 根：12.5 > 前三根的最高价 12.0"


def test_close_above_high_takes_false_where_a_bar_in_the_window_is_missing(panel):
    """窗口里**有一根**缺失即不足以判定，故不给信号（缺口纪律，与 :func:`new_high` 同）。

    这条挡住「把缺失跳过、用其余几根凑一个最大值」那种写法——它会把停牌期当成「没有更高的
    价」，从而凭空造出「高于前高」。
    """
    nan = float("nan")
    bars = _high_close_panel(panel, [10.0, nan, nan, 20.0], [9.0, nan, nan, 20.0])

    got = close_above_high(bars, n=3)["sh600000"]

    assert got.tolist() == [False, False, False, False], "窗口里有两根缺失，不该给信号"


def test_close_above_high_keeps_the_symbol_frame_shape_and_boolean_dtype(panel):
    """跨字段的过滤信号**返回布尔标的宽表**（ADR-0009）：消费方向与其余过滤器一致。"""
    bars = _high_close_panel(panel, [10.0, 11.0, 12.0, 13.0], [9.0, 10.0, 11.0, 12.5])

    got = close_above_high(bars, n=3)

    assert list(got.columns) == ["sh600000"]
    assert all(dtype.kind == "b" for dtype in got.dtypes)
