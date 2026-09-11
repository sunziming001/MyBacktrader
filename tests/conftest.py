"""测试共享工具。

fixture 数据全部位于 ``tests/fixtures/`` 之下，测试自足——不依赖本机通达信安装路径。
"""

from pathlib import Path

import pandas as pd
import pytest

TESTS_DIR = Path(__file__).parent

#: 通达信数据源 fixture 根目录。测试把它当作「本机数据源根路径」注入。
FIXTURE_ROOT = TESTS_DIR / "fixtures" / "tdx"

#: 权息事件文件 fixture：sh600000 与 sz000001 两段**真实记录**的逐字节拼接（见其生成说明）。
GBBQ_FIXTURE = TESTS_DIR / "fixtures" / "gbbq" / "gbbq"

#: 独立 oracle（pytdx）对上述记录的解读结果，仅作对照。
GBBQ_ORACLE = TESTS_DIR / "fixtures" / "gbbq" / "records_oracle.csv"


@pytest.fixture
def fixture_root():
    """通达信数据源 fixture 根目录。"""
    return FIXTURE_ROOT


@pytest.fixture
def gbbq_file():
    """gbbq 权息事件文件 fixture 路径。"""
    return GBBQ_FIXTURE


@pytest.fixture
def gbbq_oracle():
    """独立 oracle 产出的对照表路径。"""
    return GBBQ_ORACLE


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
    """工厂：由收盘价序列构造最小 OHLCV **字段宽表**（一个标的一份）。

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


@pytest.fixture
def symbol_frame():
    """工厂：由 ``{标的: 序列}`` 构造信号层的**标的宽表**输入（``日期 × 标的``）。

    缺失值请显式写成 ``float("nan")``——本工厂**不填充**，也不接受 ``None``：
    信号层把缺失当正常状态处理，填充会在测试里先把这条纪律破坏掉。
    """

    def _make(values, start="2024-01-02"):
        idx = pd.bdate_range(start, periods=len(next(iter(values.values()))))
        return pd.DataFrame(values, index=idx)

    return _make


@pytest.fixture
def make_market():
    """工厂：由**字段宽表**构造 :class:`MarketData`，不碰磁盘、也不跑越界检查。

    ``MarketData`` 是 frozen dataclass 且字段公开，故测试可以直接构造它——无需 mock，
    也无需在临时目录里摆一个真实 ``.day`` 文件（后者是解析层的测试才需要的）。
    """

    def _make(symbol, prices):
        from mbt.data import MarketData

        return MarketData(symbol=symbol, prices=prices, events=())

    return _make


@pytest.fixture
def panel(symbol_frame):
    """工厂：由 ``{字段: {标的: 序列}}`` 构造 :class:`Panel`（字段自动对齐）。

    字段间的**错位**要测时请自行构造数据帧再直接调 ``Panel(...)``——本工厂刻意只造合法的。
    """

    def _make(fields):
        from mbt.data import Panel

        return Panel({name: symbol_frame(values) for name, values in fields.items()})

    return _make


@pytest.fixture
def make_vipdoc(tmp_path):
    """工厂：在临时目录搭一个含多个标的、跨市场的 ``vipdoc`` 根，返回其根路径。

    ``extra_files`` 用于摆上不该被当成标的的杂项文件（如说明文件、非 ``.day`` 后缀）。
    """

    def _make(symbols, extra_files=(), start="2024-01-02", periods=3):
        root = tmp_path / "vipdoc"
        closes = [10.0 + i for i in range(periods)]
        for symbol in symbols:
            target = root / symbol[:2] / "lday"
            target.mkdir(parents=True, exist_ok=True)
            frame = pd.DataFrame(
                {
                    "open": closes,
                    "high": closes,
                    "low": closes,
                    "close": closes,
                    "volume": [1000] * periods,
                },
                index=pd.bdate_range(start, periods=periods),
            )
            frame.to_csv(target / f"{symbol}.day", index=False)
        for relative in extra_files:
            path = root / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("not a symbol", encoding="utf-8")
        return root

    return _make
