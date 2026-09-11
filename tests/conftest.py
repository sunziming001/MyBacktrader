"""测试共享工具。

fixture 数据全部位于 ``tests/fixtures/`` 之下，测试自足——不依赖本机通达信安装路径。
"""

from pathlib import Path

import pandas as pd
import pytest

TESTS_DIR = Path(__file__).parent

#: 通达信数据源 fixture 根目录。测试把它当作「本机数据源根路径」注入。
FIXTURE_ROOT = TESTS_DIR / "fixtures" / "tdx"


@pytest.fixture
def fixture_root():
    """通达信数据源 fixture 根目录。"""
    return FIXTURE_ROOT


@pytest.fixture
def synthetic_rules():
    """合成规则表路径——只用于测试查表机制，数值非真实制度数据（测试缝 S3）。"""
    return TESTS_DIR / "fixtures" / "rules" / "synthetic.toml"


@pytest.fixture
def zero_cost_rules():
    """零费用规则表路径——让撮合机制测试与费用数值测试分离（测试缝 S3）。"""
    return TESTS_DIR / "fixtures" / "rules" / "zero-costs.toml"


@pytest.fixture
def limit_rules():
    """撮合约束夹具规则表路径：限幅取真实制度值，费用为零（测试缝 S3）。"""
    return TESTS_DIR / "fixtures" / "rules" / "limit-fixture.toml"


@pytest.fixture
def make_prices():
    """工厂：由收盘价序列构造最小 OHLCV 宽表。

    为让行为可手算，默认令 open = high = low = close，即每根 K 线无振幅。

    .. warning::

        正因为每根 K 线 ``o == h == l == c``，**相邻收盘若恰好相差一个限幅**
        （主板 10% 步进，如 10 → 11），该根 K 线就构成**一字板**，会被撮合约束
        正确挡下而改变成交日。要测净值/费用记账时请避开这种步进，或显式传入
        非 10% 的价差。相关判定见 ``mbt.backtest.costs.AStockBroker._limit_locked``。
    """

    def _make(closes, start="2024-01-02", volume=1000):
        idx = pd.bdate_range(start, periods=len(closes))
        return pd.DataFrame(
            {
                "open": list(closes),
                "high": list(closes),
                "low": list(closes),
                "close": list(closes),
                "volume": [volume] * len(closes),
            },
            index=idx,
        )

    return _make


@pytest.fixture
def day_bytes():
    """工厂：把原始整数记录编码为 ``.day`` 文件字节。

    记录为 ``(date, open, high, low, close, amount, volume)``，其中价格是
    「价格 × 100」的整数——**调用方直接给原始整数**，本工厂不做任何缩放换算。

    文件格式本身的正确性由真实 fixture 测试锁定；此工厂只用于构造边界情形。
    """
    import struct

    def _make(records):
        out = bytearray()
        for date, o, h, low, c, amount, volume in records:
            out += struct.pack("<IIIIIfII", date, o, h, low, c, amount, volume, 0)
        return bytes(out)

    return _make


@pytest.fixture
def make_source(tmp_path, day_bytes):
    """工厂：在临时目录搭一个只含单个标的的数据源，返回其根路径。

    这样边界情形测试无需提交 fixture 文件，也不会碰本机通达信安装。
    """

    def _make(records, symbol="sh600000"):
        root = tmp_path / "vipdoc"
        target = root / symbol[:2] / "lday"
        target.mkdir(parents=True)
        (target / f"{symbol}.day").write_bytes(day_bytes(records))
        return root

    return _make
