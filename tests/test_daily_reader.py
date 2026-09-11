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


def test_halt_day_is_absent_and_not_filled(fixture_root):
    """停牌日完全没有记录，且不得用前值填充。

    2026-04-22 是全市场交易日（已在 ``sh600000`` 等大盘股上交叉验证），而
    ``sz000609`` 缺该日——即当天停牌。相邻两日俱在，可证是缺一天而非区间缺失。
    """
    df = TdxDataSource(fixture_root).daily("sz000609")

    assert pd.Timestamp("2026-04-22") not in df.index
    assert pd.Timestamp("2026-04-21") in df.index
    assert pd.Timestamp("2026-04-23") in df.index
    # 若被填充，这两日之间会多出一条等于前收的记录
    assert len(df) == 16


def test_limit_up_one_word_board_is_preserved(fixture_root):
    """连续涨停一字板保留为 o==h==l==c 且成交量>0。

    票据 04 要据此判定「涨停买不进、跌停卖不出」；若解析器把这种 K 线抹平或
    补上振幅，该判定就失去依据。
    """
    df = TdxDataSource(fixture_root).daily("sz000609")

    for day in ("2026-04-20", "2026-04-21", "2026-04-23", "2026-04-30"):
        row = df.loc[day]
        assert row["open"] == row["high"] == row["low"] == row["close"], day
        assert row["volume"] > 0, day


def test_one_fixture_spans_a_limit_regime_change(fixture_root):
    """``sh600243`` 同一文件内跨了一次限幅变更——制度规则表必须按日期查表。

    2025-03-20 是 +10.06% 的涨停一字（主板 10% 限幅），2025-04-23 是 −4.86% 的
    跌停一字（约 5% 限幅，该标的此时已带 ST）。若把限幅写成常量，其中一根必被误判。
    该文件同时含停牌空档 2025-04-22。
    """
    df = TdxDataSource(fixture_root).daily("sh600243")

    up = df.loc["2025-03-20"]
    assert up["open"] == up["high"] == up["low"] == up["close"] == 3.61
    assert df.loc["2025-03-19", "close"] == 3.28

    down = df.loc["2025-04-23"]
    assert down["open"] == down["high"] == down["low"] == down["close"] == 2.35
    assert df.loc["2025-04-21", "close"] == 2.47

    assert pd.Timestamp("2025-04-22") not in df.index
    assert len(df) == 30
