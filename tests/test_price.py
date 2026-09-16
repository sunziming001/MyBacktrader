"""涨跌停价：向量化实现必须与 Decimal 标量实现**逐位一致**（ADR-0002）。

为什么逐位而不是近似：限价既是撮合的判据、也是质检的判据，落在「恰好触限」那一分钱上。
ADR-0002 记着浮点 `round()` 与十进制半进位在 16 只主板股票上有 1,916/43,982 天分歧，
而市场**全部**落在十进制一侧。故向量化**不许**改变任何一个结果。

本文件用现有的标量 :func:`~mbt.rules.price.limit_price`（走 Decimal）当**准绳**——它是实现前
就存在的独立来源，故这些用例不是「照实现重算一遍」。
"""

from __future__ import annotations

import numpy as np
import pytest

from mbt.rules import limit_band, limit_price
from mbt.rules.price import limit_prices


def test_the_vectorized_version_matches_the_decimal_one_on_a_dense_grid():
    """逐分扫一遍价格区间，四种限幅、两个方向，结果必须**恰好相等**。

    取 0.01~30.00 的每一分（3,000 个值）——A 股绝大多数价格落在这里，且**恰好半分**的
    情形（十进制半进位与二进制四舍五入分歧的地方）密集出现。
    """
    cents = np.arange(1, 3001)
    prices = cents / 100.0

    for limit in (0.05, 0.10, 0.20, 0.30):
        for direction in (+1, -1):
            got = limit_prices(prices, limit, direction)
            want = np.array([limit_price(float(p), limit, direction) for p in prices])
            mismatch = np.flatnonzero(got != want)
            assert mismatch.size == 0, (
                f"limit={limit} direction={direction} 有 {mismatch.size} 处不一致，"
                f"首个：前收 {prices[mismatch[0]]:.2f} "
                f"向量化给 {got[mismatch[0]]:.2f}、Decimal 给 {want[mismatch[0]]:.2f}"
            )


def test_the_vectorized_version_keeps_the_decimal_half_up_case():
    """把 ADR-0002 那个**具名**的分歧点钉住：前收 14.45、跌停 10%。

    十进制半进位给 **13.01**，而浮点 `round(14.45 * 0.9, 2)` 给 13.00。向量化必须落在
    13.01 一侧——这一条独立于上面那张网格，因为它是文档里点名的例子。
    """
    got = limit_prices(np.array([14.45]), 0.10, -1)
    assert float(got[0]) == limit_price(14.45, 0.10, -1) == 13.01


def test_the_vectorized_version_gives_nan_for_bases_that_cannot_have_a_limit():
    """前收缺失或非正时给缺失——调用方据此跳过，而不是拿一个假价格去比较。

    ``st.py`` 的逐根循环原本用 ``base is None or base != base or base <= 0`` 跳过，
    向量化之后这个判断必须由「给 NaN」承担（``NaN == x`` 恒假，故比较自然为假）。
    """
    got = limit_prices(np.array([np.nan, 0.0, -1.0, 10.0]), 0.10, +1)
    assert np.isnan(got[:3]).all()
    assert float(got[3]) == limit_price(10.0, 0.10, +1)


def test_the_vectorized_version_rejects_a_limit_that_is_not_in_cents():
    """限幅必须是整数分（两位小数）——否则整数运算复刻不了 Decimal 的语义。

    规则表里的取值只有 0.05 / 0.10 / 0.20 / 0.30，故这是个**守卫**：将来有人往表里写个
    0.075 之类，这里要报错，而不是静默算错一分钱。
    """
    with pytest.raises(ValueError, match="两位小数"):
        limit_prices(np.array([10.0]), 0.075, +1)


def test_the_vectorized_version_rejects_a_direction_that_is_not_pm_one():
    """方向只可能是涨停或跌停；其余取值是调用方的错，不该静默取到某个价。"""
    for bad in (0, 2, -2):
        with pytest.raises(ValueError, match="direction"):
            limit_prices(np.array([10.0]), 0.10, bad)


def test_the_band_helper_agrees_with_the_vectorized_ends():
    """把两个方向的组合也对一遍：``limit_band`` 给的是 (跌停, 涨停)。"""
    prices = np.array([1.00, 3.33, 14.45, 99.99, 1234.56])
    lower, upper = (
        limit_prices(prices, 0.10, -1),
        limit_prices(prices, 0.10, +1),
    )
    for position, price in enumerate(prices):
        want = limit_band(float(price), 0.10)
        assert (float(lower[position]), float(upper[position])) == want
