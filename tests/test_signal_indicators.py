"""指标契约：宽表进、宽表出，纯函数（票据 #5）。

指标是「由**单一标的**的价量序列算出的数值序列，本身不含任何观点或判断」
（``CONTEXT.md``）。故这里的断言只关心数值与形状，不关心任何判断语义。
"""

from __future__ import annotations

import pandas as pd
import pytest

from mbt.signals import rolling_max, sma


def test_sma_averages_the_trailing_window(wide):
    """2 日均线：窗口不足处为缺失，其后是含当根的尾随均值。"""
    prices = wide({"sh600000": [10.0, 11.0, 12.0, 13.0]})

    got = sma(prices, n=2)["sh600000"]

    assert pd.isna(got.iloc[0])
    assert got.iloc[1] == pytest.approx(10.5)
    assert got.iloc[2] == pytest.approx(11.5)
    assert got.iloc[3] == pytest.approx(12.5)


def test_sma_keeps_the_wide_shape_and_float_dtype(wide):
    """形状、索引、列顺序一律保持不变——回测按列取时序、选股按行取截面。"""
    prices = wide({"sh600000": [10.0, 11.0, 12.0], "sz000001": [20.0, 21.0, 22.0]})

    got = sma(prices, n=2)

    assert list(got.columns) == ["sh600000", "sz000001"]
    assert got.index.equals(prices.index)
    assert all(dtype.kind == "f" for dtype in got.dtypes)


def test_sma_of_an_all_missing_series_stays_missing(wide):
    """全为缺失值的窗口不产生数值，也不被填充（ADR-0005 的缺口纪律）。"""
    nan = float("nan")
    prices = wide({"sh600000": [nan, nan, nan]})

    got = sma(prices, n=2)["sh600000"]

    assert got.isna().all()


def test_rolling_max_covers_the_trailing_n_bars_including_the_current_one(wide):
    """N 日最高价：窗口不足处为缺失；窗口内取最大，含当根。"""
    prices = wide({"sh600000": [1.0, 3.0, 2.0, 5.0]})

    got = rolling_max(prices, n=2)["sh600000"]

    assert pd.isna(got.iloc[0])
    assert got.iloc[1] == pytest.approx(3.0)
    assert got.iloc[2] == pytest.approx(3.0)
    assert got.iloc[3] == pytest.approx(5.0)


def test_a_non_monotonic_index_is_rejected_rather_than_silently_reordered(wide):
    """宽表必须按交易日升序。乱序会让「只用过去」的保证失效，故报错而非就地排序。"""
    idx = pd.to_datetime(["2024-01-03", "2024-01-02", "2024-01-04"])
    prices = pd.DataFrame({"sh600000": [10.0, 11.0, 12.0]}, index=idx)

    with pytest.raises(ValueError, match="升序"):
        sma(prices, n=2)
