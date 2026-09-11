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

from .errors import MarketDataError

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


class TdxDataSource:
    """通达信本地数据源。

    参数:
        root: 数据源根目录，即包含 ``sh`` / ``sz`` / ``bj`` 子目录的 ``vipdoc`` 目录。

    根目录由调用方注入，测试因此可以指向仓库内 fixture，而不依赖本机安装路径。
    """

    def __init__(self, root):
        self._root = Path(root)

    def path_for(self, symbol: str) -> Path:
        """标的日线文件路径。``symbol`` 形如 ``sh600000``，与文件名一致。"""
        return self._root / symbol[:2] / "lday" / f"{symbol}.day"

    def daily(self, symbol: str) -> pd.DataFrame:
        """读取单个标的的日线**原始价**。

        ``symbol`` 形如 ``sh600000``（市场前缀 + 代码），与文件名一致。

        返回以交易日为索引的宽表，列含 ``open`` / ``high`` / ``low`` / ``close`` /
        ``volume`` / ``amount``。

        文件不存在时抛 ``MarketDataError``——对调用方而言「这个标的取不到数据」
        与「数据损坏」是同一种处境，无需分别为之捕捉两种异常。
        """
        path = self.path_for(symbol)
        if not path.is_file():
            raise MarketDataError(f"未找到标的 {symbol} 的日线文件：{path}")
        return self._decode_day_records(path.read_bytes())

    @staticmethod
    def _decode_day_records(raw: bytes) -> pd.DataFrame:
        if len(raw) % DAY_RECORD_SIZE != 0:
            raise MarketDataError(
                f"文件长度 {len(raw)} 不是记录长度 {DAY_RECORD_SIZE} 的整数倍，"
                f"文件格式与预期不符"
            )

        records = np.frombuffer(raw, dtype=DAY_DTYPE)
        TdxDataSource._reject_anomalies(records)

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
    def _reject_anomalies(records: np.ndarray) -> None:
        """异常值检查：价格为 0、K 线不自洽。

        涨跌幅是否超出当日制度上限同样属于异常，但判定它需要按日期与板块查表，
        属制度规则表的职责（票据 03），不在此处实现。
        """
        open_ = records["open"].astype("int64")
        high = records["high"].astype("int64")
        low = records["low"].astype("int64")
        close = records["close"].astype("int64")

        TdxDataSource._reject(
            records,
            (open_ == 0) | (high == 0) | (low == 0) | (close == 0),
            "价格为 0 的记录不可信",
        )
        TdxDataSource._reject(
            records,
            (high < low) | (open_ < low) | (open_ > high) | (close < low) | (close > high),
            "K 线不自洽，开盘/收盘越出当日高低区间",
        )

    @staticmethod
    def _reject(records: np.ndarray, mask: np.ndarray, reason: str) -> None:
        """首个命中的记录即抛错——缺口纪律要求异常报错，而非跳过或填充。"""
        if not mask.any():
            return
        i = int(np.flatnonzero(mask)[0])
        raise MarketDataError(f"{reason}：记录 {i}，日期 {records['date'][i]}")
