"""指标契约：标的宽表进出，纯函数（票据 #5）。

指标是「由**单一标的**的价量序列算出的数值序列，本身不含任何观点或判断」
（``CONTEXT.md``）。故这里的断言只关心数值与形状，不关心任何判断语义。
"""

from __future__ import annotations

import pandas as pd
import pytest

from mbt.signals import atr, rolling_max, sma, volume_ratio


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
