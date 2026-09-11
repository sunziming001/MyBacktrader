"""验证日线解析器：从通达信定长记录文件中读出**原始价**与成交量。

测试缝 S2：数据源根路径被注入，指向仓库内 fixture，因此测试自足、不依赖本机通达信安装。

fixture 为真实文件 ``sh600000.day`` 的最后 45 条记录（2026-07-10 → 2026-09-10），
其中的期望值由独立解码该文件的字节得到，而非由被测代码计算——故可作黄金值。
"""

import pandas as pd

from mbt.data import TdxDataSource

SYMBOL = "sh600000"


def test_reads_expected_number_of_rows(fixture_root):
    df = TdxDataSource(fixture_root).daily(SYMBOL)

    assert len(df) == 45
    assert isinstance(df.index, pd.DatetimeIndex)
    assert df.index[0] == pd.Timestamp("2026-07-10")
    assert df.index[-1] == pd.Timestamp("2026-09-10")


def test_reads_first_and_last_record_verbatim(fixture_root):
    df = TdxDataSource(fixture_root).daily(SYMBOL)

    first = df.iloc[0]
    assert first["open"] == 8.98
    assert first["high"] == 9.11
    assert first["low"] == 8.92
    assert first["close"] == 9.06
    assert first["volume"] == 76502191
    assert first["amount"] == 691188288.0

    last = df.iloc[-1]
    assert last["open"] == 9.22
    assert last["high"] == 9.36
    assert last["low"] == 9.19
    assert last["close"] == 9.35
    assert last["volume"] == 59774258


def test_prices_are_raw_not_adjusted(fixture_root):
    """除权日的原始跳空必须保留——复权是后续票据的职责，解析器不得擅自调整。"""
    df = TdxDataSource(fixture_root).daily(SYMBOL)

    assert df.loc["2026-07-15", "close"] == 9.31
    assert df.loc["2026-07-16", "open"] == 8.92
    assert df.loc["2026-07-16", "close"] == 8.85
