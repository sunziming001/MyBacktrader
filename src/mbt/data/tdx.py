"""通达信本地数据源。

解析 ``<root>/<market>/lday/<symbol>.day`` 定长记录文件：每条记录 32 字节、小端序，
价格字段为「价格 × 100」的整数，成交量单位为股、成交额单位为元。

文件存的是**原始价**，未经除权除息调整（ADR-0003）。本模块**不做任何价格调整**——
复权是复权层的职责，在此处调整会让解析结果无法复现。
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

#: 单条日线记录的字节数。
DAY_RECORD_SIZE = 32

#: 日线记录布局。字段顺序与字节宽度由通达信文件格式决定。
DAY_DTYPE = np.dtype(
    [
        ("date", "<u4"),  # YYYYMMDD
        ("open", "<u4"),
        ("high", "<u4"),
        ("low", "<u4"),
        ("close", "<u4"),
        ("amount", "<f4"),  # 成交额（元）
        ("volume", "<u4"),  # 成交量（股）
        ("reserved", "<u4"),
    ]
)

#: 价格字段的缩放：文件中存的是「价格 × 100」的整数。
PRICE_SCALE = 100.0


class MarketDataError(Exception):
    """行情数据不可信。

    缺口纪律（ADR-0005）：缺失跳过、**异常报错**、永不填充。异常在这里意味着
    「文件内容与预期不符，不能继续」，而不是「这个标的今天没有数据」——
    后者是正常状态（停牌），只表现为没有记录。
    """


class TdxDataSource:
    """通达信本地数据源。

    参数:
        root: 数据源根目录，即包含 ``sh`` / ``sz`` / ``bj`` 子目录的 ``vipdoc`` 目录。

    根目录由调用方注入，测试因此可以指向仓库内 fixture，而不依赖本机安装路径。
    """

    def __init__(self, root):
        self._root = Path(root)

    def daily(self, symbol: str) -> pd.DataFrame:
        """读取单个标的的日线**原始价**。

        ``symbol`` 形如 ``sh600000``（市场前缀 + 代码），与文件名一致。

        返回以交易日为索引的宽表，列含 ``open`` / ``high`` / ``low`` / ``close`` /
        ``volume`` / ``amount``。
        """
        market = symbol[:2]
        path = self._root / market / "lday" / f"{symbol}.day"
        return self._decode(path.read_bytes())

    @staticmethod
    def _decode(raw: bytes) -> pd.DataFrame:
        if len(raw) % DAY_RECORD_SIZE != 0:
            raise MarketDataError(
                f"文件长度 {len(raw)} 不是记录长度 {DAY_RECORD_SIZE} 的整数倍，"
                f"文件格式与预期不符"
            )

        records = np.frombuffer(raw, dtype=DAY_DTYPE)
        TdxDataSource._validate(records)

        dates = pd.to_datetime(records["date"].astype("int64").astype(str), format="%Y%m%d")

        frame = pd.DataFrame(
            {
                "open": records["open"] / PRICE_SCALE,
                "high": records["high"] / PRICE_SCALE,
                "low": records["low"] / PRICE_SCALE,
                "close": records["close"] / PRICE_SCALE,
                "volume": records["volume"].astype("int64"),
                "amount": records["amount"].astype("float64"),
            },
            index=dates,
        )
        frame.index.name = "date"
        return frame

    @staticmethod
    def _validate(records: np.ndarray) -> None:
        """异常值检查：价格为 0、K 线不自洽。

        涨跌幅是否超出当日制度上限同样属于异常，但判定它需要按日期与板块查表，
        属制度规则表的职责（票据 03），不在此处实现。
        """
        open_ = records["open"].astype("int64")
        high = records["high"].astype("int64")
        low = records["low"].astype("int64")
        close = records["close"].astype("int64")

        zero = (open_ == 0) | (high == 0) | (low == 0) | (close == 0)
        if zero.any():
            i = int(np.flatnonzero(zero)[0])
            raise MarketDataError(f"价格为 0 的记录不可信：记录 {i}，日期 {records['date'][i]}")

        inconsistent = (
            (high < low) | (open_ < low) | (open_ > high) | (close < low) | (close > high)
        )
        if inconsistent.any():
            i = int(np.flatnonzero(inconsistent)[0])
            raise MarketDataError(
                f"K 线不自洽，开盘/收盘越出当日高低区间：记录 {i}，日期 {records['date'][i]}"
            )
