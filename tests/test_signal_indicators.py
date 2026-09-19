"""指标契约：标的宽表进出，纯函数（票据 #5）。

指标是「由**单一标的**的价量序列算出的数值序列，本身不含任何观点或判断」
（``CONTEXT.md``）。故这里的断言只关心数值与形状，不关心任何判断语义。
"""

from __future__ import annotations

import pandas as pd
import pytest

from mbt.signals import (
    KDJ,
    atr,
    ema,
    kdj,
    rolling_max,
    rolling_min,
    sma,
    upper_shadow_atr,
    volume_ratio,
    white_line,
    yellow_line,
)


def test_sma_averages_the_trailing_window(symbol_frame):
    """2 日均线：窗口不足处为缺失，其后是含当根的尾随均值。"""
    prices = symbol_frame({"sh600000": [10.0, 11.0, 12.0, 13.0]})

    got = sma(prices, n=2)["sh600000"]

    assert pd.isna(got.iloc[0])
    assert got.iloc[1] == pytest.approx(10.5)
    assert got.iloc[2] == pytest.approx(11.5)
    assert got.iloc[3] == pytest.approx(12.5)


def test_sma_keeps_the_symbol_frame_shape_and_float_dtype(symbol_frame):
    """形状、索引、列顺序一律保持不变——回测按列取时序、选股按行取截面。"""
    prices = symbol_frame({"sh600000": [10.0, 11.0, 12.0], "sz000001": [20.0, 21.0, 22.0]})

    got = sma(prices, n=2)

    assert list(got.columns) == ["sh600000", "sz000001"]
    assert got.index.equals(prices.index)
    assert all(dtype.kind == "f" for dtype in got.dtypes)


def test_sma_of_an_all_missing_series_stays_missing(symbol_frame):
    """全为缺失值的窗口不产生数值，也不被填充（ADR-0005 的缺口纪律）。"""
    nan = float("nan")
    prices = symbol_frame({"sh600000": [nan, nan, nan]})

    got = sma(prices, n=2)["sh600000"]

    assert got.isna().all()


def test_rolling_max_covers_the_trailing_n_bars_including_the_current_one(symbol_frame):
    """N 日最高价：窗口不足处为缺失；窗口内取最大，含当根。"""
    prices = symbol_frame({"sh600000": [1.0, 3.0, 2.0, 5.0]})

    got = rolling_max(prices, n=2)["sh600000"]

    assert pd.isna(got.iloc[0])
    assert got.iloc[1] == pytest.approx(3.0)
    assert got.iloc[2] == pytest.approx(3.0)
    assert got.iloc[3] == pytest.approx(5.0)


def test_a_non_monotonic_index_is_rejected_rather_than_silently_reordered(symbol_frame):
    """标的宽表必须按交易日升序。乱序会让「只用过去」的保证失效，故报错而非就地排序。"""
    idx = pd.to_datetime(["2024-01-03", "2024-01-02", "2024-01-04"])
    prices = pd.DataFrame({"sh600000": [10.0, 11.0, 12.0]}, index=idx)

    with pytest.raises(ValueError, match="升序"):
        sma(prices, n=2)


# --- 成交量比（单字段）与 ATR（跨字段，取面板） ---------------------------------


def test_volume_ratio_divides_by_the_mean_of_the_prior_n_bars(symbol_frame):
    """成交量比 = 当根量 ÷ **前 n 根**均量：400 ÷ mean(100,100,100) = 4。"""
    volumes = symbol_frame({"sh600000": [100.0, 100.0, 100.0, 400.0]})

    got = volume_ratio(volumes, n=3)["sh600000"]

    assert pd.isna(got.iloc[0])
    assert pd.isna(got.iloc[1])
    assert pd.isna(got.iloc[2])
    assert got.iloc[3] == pytest.approx(4.0)


def test_volume_ratio_excludes_the_current_bar_from_its_baseline(symbol_frame):
    """基准若含当根，同一个 400 会得到 2.0 而非 4.0——这条把口径钉死在「前 n 根」。"""
    volumes = symbol_frame({"sh600000": [100.0, 100.0, 100.0, 400.0]})

    got = volume_ratio(volumes, n=3)["sh600000"]

    assert got.iloc[3] == pytest.approx(4.0)
    assert got.iloc[3] != pytest.approx(400.0 / ((100.0 + 100.0 + 400.0) / 3))


def test_volume_ratio_over_a_zero_prior_mean_is_infinite_not_an_error(symbol_frame):
    """前面 n 根全无成交（长期停牌）时比值为正无穷；过滤器据此照常判为放量。"""
    from mbt.signals import volume_surge

    volumes = symbol_frame({"sh600000": [0.0, 0.0, 0.0, 5.0]})

    ratio = volume_ratio(volumes, n=3)["sh600000"]
    surge = volume_surge(volumes, k=2.0, n=3)["sh600000"]

    assert pd.isna(ratio.iloc[2])
    assert ratio.iloc[3] == float("inf")
    assert bool(surge.iloc[3]) is True


def test_volume_ratio_is_not_the_exchange_intraday_definition(symbol_frame):
    """它**不是**交易所口径的「量比」，这条把两者的定义性差别钉住。

    官方量比是「当日开盘后**每分钟平均**成交量 ÷ 过去 **5 个**交易日每分钟平均成交量」——
    窗口固定为 5 日，基准还是**盘中每分钟**均量。本函数则是「当根**整日**量 ÷ 前 n 根**整日**
    均量」，且 **n 由调用方给定**；故它对 n 的响应本身就是与量比的差别所在。
    """
    volumes = symbol_frame({"sh600000": [100.0, 200.0, 300.0, 400.0]})

    by_one = volume_ratio(volumes, n=1)["sh600000"]
    by_three = volume_ratio(volumes, n=3)["sh600000"]

    assert by_one.iloc[3] == pytest.approx(400.0 / 300.0)  # 基准只取前 1 根
    assert by_three.iloc[3] == pytest.approx(400.0 / 200.0)  # 基准取前 3 根


def test_volume_surge_is_strictly_the_ratio_above_k(symbol_frame):
    """``volume_surge`` 与 ``volume_ratio`` 口径一致：2.5 > 2 为真，1.0 > 2 为假。"""
    from mbt.signals import volume_surge

    volumes = symbol_frame({"sh600000": [100.0, 100.0, 100.0, 250.0, 150.0]})

    got = volume_surge(volumes, k=2.0, n=3)["sh600000"]

    assert got.tolist() == [False, False, False, True, False]


def atr_panel(panel):
    """用于手算的 5 根 K 线：真实波幅恰为 [1.0, 1.5, 1.5, 1.5, 2.5]。"""
    return panel(
        {
            "high": {"sh600000": [10.0, 11.0, 12.0, 11.0, 13.0]},
            "low": {"sh600000": [9.0, 9.5, 11.0, 10.0, 12.0]},
            "close": {"sh600000": [9.5, 10.5, 11.5, 10.5, 12.5]},
        }
    )


def test_atr_matches_the_hand_computed_wilder_values(panel):
    """手算锁定 Wilder 口径：种子 = mean(1.0, 1.5, 1.5) = 4/3，其后 (ATR×2 + TR) ÷ 3。

    第 4、5 根的期望值是**按公式手算出的常数**（1.388888…、1.759259…），不是照实现抄的。
    """
    got = atr(atr_panel(panel), n=3)["sh600000"]

    assert pd.isna(got.iloc[0])
    assert pd.isna(got.iloc[1])
    assert got.iloc[2] == pytest.approx(1.3333333333333333)
    assert got.iloc[3] == pytest.approx(1.3888888888888888)
    assert got.iloc[4] == pytest.approx(1.759259259259259)


def test_atr_is_not_the_simple_average_of_true_range(panel):
    """ATR 是 Wilder 平滑，**不是**真实波幅的简单均线——两者在这组样本上必须可区分。

    简单均线版会是 1.5 与 1.833333…，故这条能真正区分两种口径；若实现被改成简单均线，
    它会失败。（「ATR」不加限定常被读成简单均线版，故口径必须写明并锁住。）
    """
    got = atr(atr_panel(panel), n=3)["sh600000"]

    assert got.iloc[3] != pytest.approx(1.5)
    assert got.iloc[4] != pytest.approx(1.8333333333333333)


def test_atr_with_window_one_is_the_true_range_and_the_first_bar_uses_high_minus_low(panel):
    """n=1 时 ATR 退化为真实波幅本身；第 1 根没有前收，故取 high − low。

    这一条把首根约定钉死：若把缺失的前收当成 0，首根的真实波幅会是 10.0 而非 1.0。
    """
    got = atr(atr_panel(panel), n=1)["sh600000"]

    assert got.tolist() == pytest.approx([1.0, 1.5, 1.5, 1.5, 2.5])


def test_atr_keeps_the_symbol_frame_shape_and_float_dtype(panel):
    """跨字段的信号**返回标的宽表**（ADR-0009）：消费方向与其余指标一致。"""
    got = atr(atr_panel(panel), n=3)

    assert list(got.columns) == ["sh600000"]
    assert all(dtype.kind == "f" for dtype in got.dtypes)


def test_atr_of_an_all_missing_field_stays_missing(panel):
    """全为缺失值的字段不产生数值，也不被填充（ADR-0005 的缺口纪律）。"""
    nan = float("nan")
    got = atr(
        panel(
            {
                "high": {"sh600000": [nan, nan, nan, nan]},
                "low": {"sh600000": [nan, nan, nan, nan]},
                "close": {"sh600000": [nan, nan, nan, nan]},
            }
        ),
        n=2,
    )

    assert got["sh600000"].isna().all()


def test_atr_does_not_carry_the_previous_value_across_a_missing_bar(panel):
    """缺口处必须是**缺失**，不能沿用前值。

    真实波幅在缺口当根无从计算；沿用前值等于把停牌日伪造成一个有波幅的交易日（ADR-0005）。
    这条能真正区分实现：整列调 ``ewm(adjust=False)`` 会把缺失位置填成上一次的状态。

    样本（n=3）：真实波幅 ``[1.0, 1.5, 1.5, 缺失, 1.5, 2.5]``，故第 3 根播种为 4/3，
    缺口在第 4 根；缺口之后只剩 2 个真实波幅，不足以重新播种。
    """
    got = atr(
        panel(
            {
                "high": {"sh600000": [10.0, 11.0, 12.0, float("nan"), 11.0, 13.0]},
                "low": {"sh600000": [9.0, 9.5, 11.0, float("nan"), 10.0, 12.0]},
                "close": {"sh600000": [9.5, 10.5, 11.5, float("nan"), 10.5, 12.5]},
            }
        ),
        n=3,
    )["sh600000"]

    assert pd.isna(got.iloc[0])
    assert pd.isna(got.iloc[1])
    assert got.iloc[2] == pytest.approx(1.3333333333333333)
    assert pd.isna(got.iloc[3]), "缺口处沿用了前值——那是填充，不是缺失"
    assert pd.isna(got.iloc[4])
    assert pd.isna(got.iloc[5])


def test_atr_re_seeds_after_a_gap_from_consecutive_true_ranges_only(panel):
    """缺口把序列切成两段，每段各自播种；缺口之后需**重新积累**连续 n 个真实波幅。

    样本（n=3）：真实波幅 ``[1.0, 1.5, 缺失, 1.5, 1.5, 1.5]``。前段只有 2 个值，不足播种；
    后段 3 个值恰好在最后一根播种为 1.5——故 ATR 直到第 6 根才有值。

    这条同时区分两个缺陷：缺口之后的**原始**真实波幅（第 4 根的 1.5）不得被当成起点而直接
    报出，且不得沿用缺口前的 4/3。
    """
    got = atr(
        panel(
            {
                "high": {"sh600000": [10.0, 11.0, float("nan"), 12.0, 13.0, 14.0]},
                "low": {"sh600000": [9.0, 9.5, float("nan"), 11.0, 12.0, 13.0]},
                "close": {"sh600000": [9.5, 10.5, float("nan"), 11.5, 12.5, 13.5]},
            }
        ),
        n=3,
    )["sh600000"]

    assert got.iloc[:5].isna().all()
    assert got.iloc[5] == pytest.approx(1.5)


def test_atr_measures_the_gap_on_the_first_bar_after_a_suspension(panel):
    """复牌当根的真实波幅要**包含跳空**：参照收盘价取最近一个可得的那一个。

    停牌期间没有收盘价，但若让这一根的真实波幅作废，就丢掉了复牌跳空——那恰恰是真实波幅
    要度量的东西。这一条与上一条用的是同一份样本：正因为复牌当根算得出真实波幅（1.5），
    后段才凑得满 3 个值、第 6 根才有 ATR。若参照收盘价不跨缺口，后段只有 2 个值，ATR 会
    一直缺失到序列结束。
    """
    got = atr(
        panel(
            {
                "high": {"sh600000": [10.0, 11.0, float("nan"), 12.0, 13.0, 14.0]},
                "low": {"sh600000": [9.0, 9.5, float("nan"), 11.0, 12.0, 13.0]},
                "close": {"sh600000": [9.5, 10.5, float("nan"), 11.5, 12.5, 13.5]},
            }
        ),
        n=3,
    )["sh600000"]

    assert not pd.isna(got.iloc[5]), "复牌当根的真实波幅被作废了，跳空因此丢失"


def test_atr_of_a_series_shorter_than_the_window_is_all_missing_not_an_error(panel):
    """序列短于窗口（次新股、切片）时整列缺失——不报错，也不假装算得出。

    这条由截断重算不变性测试逼出来的：播种位置在序列之外时会越界，而越界是**抛错**，
    会让一个次新股把整次截面计算打断。
    """
    got = atr(
        panel(
            {
                "high": {"sh600000": [10.0, 11.0]},
                "low": {"sh600000": [9.0, 10.0]},
                "close": {"sh600000": [9.5, 10.5]},
            }
        ),
        n=3,
    )

    assert got["sh600000"].isna().all()


# --- EMA 与提示 1、2 的两条线 ---------------------------------------------------


def test_ema_seeds_at_the_first_bar_like_the_charting_software_does(symbol_frame):
    """EMA **以首根播种**（EMA_1 = C_1），而不是像 sma 那样把前 n-1 根留空。

    手算（n=2，alpha = 2/3）：1 -> 5/3 -> 23/9 -> 95/27 -> 365/81。
    """
    prices = symbol_frame({"sh600000": [1.0, 2.0, 3.0, 4.0, 5.0]})

    got = ema(prices, n=2)["sh600000"]

    assert got.iloc[0] == pytest.approx(1.0)
    assert got.iloc[1] == pytest.approx(5.0 / 3.0)
    assert got.iloc[2] == pytest.approx(23.0 / 9.0)
    assert got.iloc[3] == pytest.approx(95.0 / 27.0)
    assert got.iloc[4] == pytest.approx(365.0 / 81.0)


def test_ema_is_not_the_simple_average_of_the_window(symbol_frame):
    """递推平滑与尾随均值不是一回事——这条能真正区分两种口径。

    同一组样本上 sma(2) 为 [缺失, 1.5, 2.5, 3.5, 4.5]，与本函数的首值不同、其后也不等。
    """
    prices = symbol_frame({"sh600000": [1.0, 2.0, 3.0, 4.0, 5.0]})

    got = ema(prices, n=2)["sh600000"]
    averaged = sma(prices, n=2)["sh600000"]

    assert pd.isna(averaged.iloc[0]) and not pd.isna(got.iloc[0])
    assert got.iloc[3] != pytest.approx(averaged.iloc[3])


def test_ema_keeps_a_gap_missing_and_re_seeds_after_it(symbol_frame):
    """缺口当根必须是**缺失**，其后**重新播种**。

    整列 ewm 会把缺口位置填成上一根的值（已实测：span=2 下 [1,2,NaN,4,5] 得到
    [1, 1.667, 1.667, 3.667, 4.556]，第 3 位不是缺失）——那是把停牌日伪造成有指标值的
    交易日。故这里同时钉两条：缺口处缺失，且缺口之后以 4.0 重新播种（沿用前值会得到别的数）。
    """
    nan = float("nan")
    prices = symbol_frame({"sh600000": [1.0, 2.0, nan, 4.0, 5.0]})

    got = ema(prices, n=2)["sh600000"]

    assert pd.isna(got.iloc[2]), "缺口处沿用了前值——那是填充，不是缺失"
    assert got.iloc[3] == pytest.approx(4.0)
    assert got.iloc[4] == pytest.approx(14.0 / 3.0)


def test_ema_of_an_all_missing_series_stays_missing(symbol_frame):
    """全为缺失值的序列不产生数值，也不被填充（ADR-0005 的缺口纪律）。"""
    nan = float("nan")
    prices = symbol_frame({"sh600000": [nan, nan, nan]})

    assert ema(prices, n=2)["sh600000"].isna().all()


def test_white_line_collapses_to_the_close_at_window_one(symbol_frame):
    """n=1 时 alpha=1，两层都不改变输入，故白线等于收盘价本身。

    这条同时钉住「它是**两层** EMA」：单层 EMA 在 n=1 下也是收盘价，故再加一条 n>1 的滞后断言。
    """
    prices = symbol_frame({"sh600000": [1.0, 2.0, 3.0, 4.0]})

    assert white_line(prices, n=1)["sh600000"].tolist() == pytest.approx([1.0, 2.0, 3.0, 4.0])

    got = white_line(prices, n=2)["sh600000"]
    assert got.iloc[0] == pytest.approx(1.0)
    assert got.iloc[3] < 4.0
    # 两次平滑比一次更滞后，故白线低于单层 EMA。
    assert got.iloc[3] < ema(prices, n=2)["sh600000"].iloc[3]


def test_yellow_line_is_the_equal_weighted_average_of_its_mas(symbol_frame):
    """[1,2,3,4] 上 windows=(2,3)：MA2=[缺失,1.5,2.5,3.5]、MA3=[缺失,缺失,2,3]，
    故均值为 [缺失,缺失,2.25,3.25]。等权与加权在这里得出的数不同，故这条能区分两者。
    """
    prices = symbol_frame({"sh600000": [1.0, 2.0, 3.0, 4.0]})

    got = yellow_line(prices, windows=(2, 3))["sh600000"]

    assert pd.isna(got.iloc[0])
    assert pd.isna(got.iloc[1])
    assert got.iloc[2] == pytest.approx(2.25)
    assert got.iloc[3] == pytest.approx(3.25)


def test_yellow_line_stays_missing_until_its_longest_window_is_full(symbol_frame):
    """任一条均线缺失则结果缺失，故黄线的可用起点由**最长**那个窗口决定。"""
    prices = symbol_frame({"sh600000": [1.0, 2.0, 3.0, 4.0]})

    got = yellow_line(prices, windows=(2, 4))["sh600000"]

    assert pd.isna(got.iloc[0]) and pd.isna(got.iloc[1]) and pd.isna(got.iloc[2])
    assert got.iloc[3] == pytest.approx((3.5 + 2.5) / 2.0)


def test_yellow_line_rejects_an_empty_window_list(symbol_frame):
    """没有均线就谈不上均值；静默返回 0 会让下游把「没算」当成「算出来是 0」。"""
    with pytest.raises(ValueError, match="不能为空"):
        yellow_line(symbol_frame({"sh600000": [1.0, 2.0]}), windows=())


# --- KDJ（提示 3 的 J 值） -----------------------------------------------------


def kdj_panel(panel):
    """手算样本（n=3）：每一根的窗口内高低差都可直接手算。

    RSV：r2=(11-9)/(12-9)=2/3、r3=(10-9)/(12-9)=1/3、r4=(12-10)/(12-10)=1。
    """
    return panel(
        {
            "high": {"sh600000": [10.0, 11.0, 12.0, 11.0, 12.0]},
            "low": {"sh600000": [9.0, 9.0, 10.0, 10.0, 11.0]},
            "close": {"sh600000": [10.0, 10.0, 11.0, 10.0, 12.0]},
        }
    )


def test_kdj_matches_the_hand_computed_values(panel):
    """手算锁定通达信口径（n=m1=m2=3）：K=(X+2K_prev)/3，D=(K+2D_prev)/3，J=3K-2D。

    r2 播种 K=D=2/3*100；其后 K3=(1+2*2)/3=5/9*100、K4=(1+2*5/9)/3=70.370...；
    D3=(5/9+2*2/3)/3=17/27*100、D4=(19/27+2*17/27)/3=65.432...；J3=40.740...、J4=80.246...。
    """
    lines = kdj(kdj_panel(panel), n=3, m1=3, m2=3)

    k = lines.k["sh600000"]
    d = lines.d["sh600000"]
    j = lines.j["sh600000"]

    assert pd.isna(k.iloc[0]) and pd.isna(k.iloc[1])
    assert k.iloc[2] == pytest.approx(200.0 / 3.0)
    assert k.iloc[3] == pytest.approx(500.0 / 9.0)
    assert k.iloc[4] == pytest.approx(1900.0 / 27.0)

    assert d.iloc[2] == pytest.approx(200.0 / 3.0)
    assert d.iloc[3] == pytest.approx(1700.0 / 27.0)
    assert d.iloc[4] == pytest.approx(5300.0 / 81.0)

    assert j.iloc[2] == pytest.approx(200.0 / 3.0)
    assert j.iloc[3] == pytest.approx(1100.0 / 27.0)
    assert j.iloc[4] == pytest.approx(6500.0 / 81.0)


def test_kdj_uses_the_recursive_tdx_mean_not_a_moving_average(panel):
    """K 是递推均值，**不是** RSV 的移动平均——两者在这组样本上必须可区分。

    简单 3 根均线版会给出 mean(2/3, 1/3, 1)*100 = 66.67，而通达信口径是 70.370…
    """
    lines = kdj(kdj_panel(panel), n=3, m1=3, m2=3)

    assert lines.k["sh600000"].iloc[4] != pytest.approx(200.0 / 3.0)


def test_kdj_is_missing_while_the_window_is_not_full(panel):
    """窗口不满 n 根就无从谈最高/最低，故 RSV 与三条线一律缺失。

    与图上的差别要写明：行情软件的 LLV/HHV 在不足 n 根时按已有的几根算，故它从首根就有值；
    本项目的缺口纪律不给这种乐观答案（ADR-0005）。
    """
    lines = kdj(kdj_panel(panel), n=3, m1=3, m2=3)

    for series in (lines.k, lines.d, lines.j):
        assert pd.isna(series["sh600000"].iloc[0])
        assert pd.isna(series["sh600000"].iloc[1])


def test_kdj_is_missing_when_the_window_has_no_range(panel):
    """窗口内最高价等于最低价（一字板、长期停牌复牌）时 RSV 分母为 0，三条线一律缺失。

    此处刻意**不**沿用前值：沿用会把一个没有波幅的窗口说成有 KDJ 值。
    """
    flat = panel(
        {
            "high": {"sh600000": [5.0, 5.0, 5.0, 5.0]},
            "low": {"sh600000": [5.0, 5.0, 5.0, 5.0]},
            "close": {"sh600000": [5.0, 5.0, 5.0, 5.0]},
        }
    )

    lines = kdj(flat, n=3, m1=3, m2=3)

    assert lines.k["sh600000"].isna().all()
    assert lines.d["sh600000"].isna().all()
    assert lines.j["sh600000"].isna().all()


def test_kdj_re_seeds_after_a_gap_instead_of_carrying_the_previous_value(panel):
    """缺口把序列切成两段，第二段以该段第一个可用值播种。

    样本（n=3）：第 4 根缺失。缺口后到第 7 根才凑满一个窗口，RSV=100，故 K 在那一根为
    **100**（重新播种）；若把跨缺口的递推连起来，会得到 2/3*66.67+1/3*100 ≈ 77.78。
    """
    nan = float("nan")
    gapped = panel(
        {
            "high": {"sh600000": [10.0, 11.0, 12.0, nan, 12.0, 13.0, 14.0]},
            "low": {"sh600000": [9.0, 9.0, 10.0, nan, 11.0, 12.0, 13.0]},
            "close": {"sh600000": [10.0, 10.0, 11.0, nan, 12.0, 12.5, 14.0]},
        }
    )

    k = kdj(gapped, n=3, m1=3, m2=3).k["sh600000"]

    assert k.iloc[2] == pytest.approx(200.0 / 3.0)
    assert k.iloc[3:6].isna().all()
    assert k.iloc[6] == pytest.approx(100.0)


def test_kdj_returns_a_symbol_frame_per_line_with_the_usual_shape_and_dtype(panel):
    """三条线**各自**都是合法的标的宽表（ADR-0009）：列、索引、dtype 与输入一致。"""
    lines = kdj(kdj_panel(panel), n=3, m1=3, m2=3)

    assert isinstance(lines, KDJ)
    for series in (lines.k, lines.d, lines.j):
        assert list(series.columns) == ["sh600000"]
        assert series.index.equals(lines.k.index)
        assert all(dtype.kind == "f" for dtype in series.dtypes)


def test_kdj_rejects_a_non_monotonic_index_like_the_other_indicators_do():
    """契约由所有公开函数共同遵守，跨字段的也不例外。"""
    from mbt.data import Panel

    idx = pd.to_datetime(["2024-01-03", "2024-01-02", "2024-01-04"])
    prices = pd.DataFrame({"sh600000": [10.0, 11.0, 12.0]}, index=idx)

    with pytest.raises(ValueError, match="升序"):
        kdj(Panel({"high": prices, "low": prices, "close": prices}), n=2, m1=3, m2=3)


def test_rolling_min_covers_the_trailing_n_bars_including_the_current_one(symbol_frame):
    """N 日最低价：窗口不足处为缺失；窗口内取最小，含当根。喂 ``low`` 字段即是「前低」。"""
    prices = symbol_frame({"sh600000": [5.0, 3.0, 4.0, 1.0]})

    got = rolling_min(prices, n=2)["sh600000"]

    assert pd.isna(got.iloc[0])
    assert got.iloc[1] == pytest.approx(3.0)
    assert got.iloc[2] == pytest.approx(3.0)
    assert got.iloc[3] == pytest.approx(1.0)


def test_rolling_min_over_an_all_missing_window_stays_missing(symbol_frame):
    """全为缺失值的窗口不产生数值，也不被填充（ADR-0005 的缺口纪律）。"""
    nan = float("nan")
    prices = symbol_frame({"sh600000": [nan, nan, nan]})

    assert rolling_min(prices, n=2)["sh600000"].isna().all()


def test_rolling_min_is_not_the_min_of_the_closes(symbol_frame):
    """「前低」用的是 ``low``，不是收盘价的最小值——两者在带下影线的 K 线上不同。

    这一条把口径钉死：止损位若按收盘价的最低价算，在有长下影的样本上会**偏高**，
    于是止损被触发得更早。
    """
    lows = symbol_frame({"sh600000": [10.0, 7.0, 9.0, 8.0]})
    closes = symbol_frame({"sh600000": [10.5, 9.5, 9.6, 8.5]})

    assert rolling_min(lows, n=3)["sh600000"].iloc[3] == pytest.approx(7.0)
    assert rolling_min(closes, n=3)["sh600000"].iloc[3] == pytest.approx(8.5)


# --- 上影线（当根，除以 ATR） --------------------------------------------------
#
# 与 ``mbt.signals.volume`` 的 ``top_shadow_atr`` 同**口径**、不同**指哪根**：那个指一段上涨的
# 顶部那根，这个指**当根**。故两者都取 ``high − max(开, 收)`` 再除以 ATR——这个定义只此一处，
# 两边的差别只在取哪一行。

#: 上影线那一组 K 线：`影 = 高 − max(开, 收)` 恰为 [0.0, 1.0, 0.0, 2.0, 0.5]。
SHADOW_BARS = {
    "open": [10.0, 10.0, 11.0, 10.0, 12.0],
    "high": [10.0, 11.0, 11.0, 12.5, 12.5],
    "low": [9.0, 9.5, 10.0, 10.0, 11.0],
    "close": [9.5, 10.5, 11.0, 10.5, 12.0],
}


def shadow_panel(panel):
    return panel({field: {"sh600000": values} for field, values in SHADOW_BARS.items()})


def test_the_upper_shadow_is_the_high_above_the_body_over_atr(panel):
    """``(高 − max(开, 收)) ÷ ATR``——上面那组 K 线的影恰为 [0.0, 1.0, 0.0, 2.0, 0.5]。

    期望值用**现成的** :func:`atr` 算出来再除：ATR 本身另有手算常数钉住（见下），故这里测的是
    那条**公式**与「除的是哪一根的 ATR」，而不是把 ATR 重算一遍。
    """
    prices = shadow_panel(panel)
    shadows = [0.0, 1.0, 0.0, 2.0, 0.5]

    got = upper_shadow_atr(prices, n=3)["sh600000"]
    denominator = atr(prices, n=3)["sh600000"]

    for i, shadow in enumerate(shadows):
        expected = shadow / denominator.iloc[i] if denominator.iloc[i] > 0 else float("nan")
        if expected != expected:
            assert pd.isna(got.iloc[i]), f"第 {i} 根该缺失"
        else:
            assert got.iloc[i] == pytest.approx(expected), f"第 {i} 根算错了"


def test_a_longer_upper_shadow_gives_a_bigger_number(panel):
    """**越大越长**——方向不能写反（写成 ``max(开,收) − 高`` 会得到一个恒非正数）。

    只看 ATR 有值的那几根（``n=3`` 时从第 3 根起）：第 3 根影 2.0、第 4 根影 0.5、
    第 2 根影 0.0，故顺序是 3 > 4 > 2。
    """
    got = upper_shadow_atr(shadow_panel(panel), n=3)["sh600000"]

    assert got.iloc[3] > got.iloc[4] > got.iloc[2]
    assert (got.dropna() >= 0).all(), "影取正值，不该出现负数"


def test_no_upper_shadow_gives_exactly_zero(panel):
    """收在当日最高价（或无上影）时为**恰好 0**，而不是缺失——0 是「没有上影」这个事实。"""
    got = upper_shadow_atr(shadow_panel(panel), n=3)["sh600000"]

    assert got.iloc[2] == pytest.approx(0.0), "高 = max(开,收) 的那一根，影就是 0"


def test_a_zero_atr_yields_missing_not_a_division_by_zero(panel):
    """ATR 为 0（长期停牌复牌、一字板连着走）时**缺失**——不是 0，也不是无穷。"""
    flat = panel(
        {
            "open": {"sh600000": [10.0] * 6},
            "high": {"sh600000": [10.0] * 6},
            "low": {"sh600000": [10.0] * 6},
            "close": {"sh600000": [10.0] * 6},
        }
    )

    assert upper_shadow_atr(flat, n=3)["sh600000"].isna().all()


def test_it_reads_the_current_bar_so_it_needs_no_confirmed_swing(panel):
    """指着**当根**，故不需要任何已确认的摆动点。

    这条是它与 :func:`~mbt.signals.volume.volume_pattern` 的 ``top_shadow_atr`` 的实质差别：
    那个要等一段上涨的峰值被确认之后才有值，这个从窗口一满就有——一条从头到尾单调下跌的
    行情上，前者可以整段缺失，而后者照常有值。
    """
    falling = panel(
        {
            "open": {"sh600000": [20.0, 19.0, 18.0, 17.0, 16.0, 15.0]},
            "high": {"sh600000": [20.5, 19.5, 18.5, 17.5, 16.5, 15.5]},
            "low": {"sh600000": [19.0, 18.0, 17.0, 16.0, 15.0, 14.0]},
            "close": {"sh600000": [19.0, 18.0, 17.0, 16.0, 15.0, 14.0]},
        }
    )

    assert upper_shadow_atr(falling, n=3)["sh600000"].notna().any(), "整段下跌时也该有值"


def test_the_upper_shadow_keeps_the_symbol_frame_shape_and_float_dtype(panel):
    """跨字段的信号**返回标的宽表**（ADR-0009）：消费方向与其余指标一致。"""
    got = upper_shadow_atr(shadow_panel(panel), n=3)

    assert list(got.columns) == ["sh600000"]
    assert got.index.equals(pd.bdate_range("2024-01-02", periods=5))
    assert all(dtype.kind == "f" for dtype in got.dtypes)


def test_the_upper_shadow_rejects_a_missing_field_by_name(panel):
    """少一个字段要**报错并说出缺哪个**——不是悄悄按 NaN 算（那会两边都缺失、看着像没数据）。"""
    incomplete = panel(
        {
            "high": {"sh600000": [10.0, 11.0, 12.0]},
            "low": {"sh600000": [9.0, 10.0, 11.0]},
            "close": {"sh600000": [9.5, 10.5, 11.5]},
        }
    )

    with pytest.raises(KeyError, match="open"):
        upper_shadow_atr(incomplete, n=3)
