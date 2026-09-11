"""验证缺口纪律（ADR-0005）：缺失跳过、异常报错、**永不填充**。

这些测试都在数据源根路径这条缝（S2）上，用临时目录构造边界情形。

关于「单日涨跌幅超出制度上限」这一类异常：它需要按日期与板块查表才知道上限，
属制度规则表的职责（票据 03）。此处只覆盖无需规则表即可判定的异常。
"""

import pandas as pd
import pytest

from mbt.data import MarketDataError, TdxDataSource

SYMBOL = "sh600000"


def test_missing_trading_days_are_left_missing(make_source):
    """停牌造成的空档不产生记录，也不得被补齐。"""
    root = make_source(
        [
            (20240102, 1000, 1010, 990, 1005, 1.0, 100),
            # 01-03 与 01-04 无记录（例如停牌）
            (20240105, 1005, 1020, 1000, 1015, 1.0, 120),
        ]
    )

    df = TdxDataSource(root).daily(SYMBOL)

    assert len(df) == 2
    assert list(df.index) == [pd.Timestamp("2024-01-02"), pd.Timestamp("2024-01-05")]
    assert df["close"].tolist() == [10.05, 10.15]


def test_zero_price_is_rejected(make_source):
    root = make_source([(20240102, 0, 1010, 0, 1005, 1.0, 100)])

    with pytest.raises(MarketDataError, match="价格"):
        TdxDataSource(root).daily(SYMBOL)


def test_negative_extreme_is_rejected(make_source):
    """价格字段为无符号整数，出现 0 即表示该记录不可信。"""
    root = make_source([(20240102, 1000, 1010, 990, 0, 1.0, 100)])

    with pytest.raises(MarketDataError, match="价格"):
        TdxDataSource(root).daily(SYMBOL)


def test_high_below_low_is_rejected(make_source):
    root = make_source([(20240102, 1000, 980, 1010, 1005, 1.0, 100)])

    with pytest.raises(MarketDataError, match="自洽"):
        TdxDataSource(root).daily(SYMBOL)


def test_close_outside_daily_range_is_rejected(make_source):
    root = make_source([(20240102, 1000, 1010, 990, 1050, 1.0, 100)])

    with pytest.raises(MarketDataError, match="自洽"):
        TdxDataSource(root).daily(SYMBOL)


def test_missing_symbol_file_is_rejected_as_market_data_error(make_source):
    """文件根本不存在，与文件损坏一样属于「数据不可用」。

    两者统一为 ``MarketDataError``，调用方无需为「数据取不到」捕捉两种异常类型。
    """
    root = make_source([(20240102, 1000, 1010, 990, 1005, 1.0, 100)])

    with pytest.raises(MarketDataError, match="未找到"):
        TdxDataSource(root).daily("sh999999")


def test_truncated_file_is_rejected(make_source):
    """文件长度不是记录长度的整数倍，说明格式假设已被打破。"""
    root = make_source([(20240102, 1000, 1010, 990, 1005, 1.0, 100)])
    path = root / "sh" / "lday" / f"{SYMBOL}.day"
    path.write_bytes(path.read_bytes()[:-1])

    with pytest.raises(MarketDataError, match="长度"):
        TdxDataSource(root).daily(SYMBOL)
