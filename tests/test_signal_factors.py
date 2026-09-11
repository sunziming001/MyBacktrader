"""排序因子：同一时刻可横向比较大小的数值（``CONTEXT.md``），票据 #5。

因子的用处与指标不同：不是「某标的的序列」，而是「同一行里谁更靠前」。故这里的断言除了
数值本身，还要**在同一行内比较两个标的**——那才是它的消费方向。
"""

from __future__ import annotations

import pandas as pd
import pytest

from mbt.signals import distance_to_high, momentum


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
