"""数据层：通达信本地数据的解析与访问，以及复权视图。

本层只依赖 pandas 与 numpy，**不依赖 backtrader**（ADR-0007）。

复权遵循 ADR-0003：只保存 / 传递**原始价与复权事件**，已复权的价格序列一律**用时计算**
（见 :mod:`mbt.data.adjust`）。原始价来自 :class:`TdxDataSource`，除权除息事件来自
:class:`GbbqDataSource`——两者的根目录不同，由调用方分别注入后组合。
"""

from .adjust import (
    AdjustmentEvent,
    adjustment_factors,
    backward_adjusted,
    forward_adjusted,
)
from .errors import MarketDataError
from .gbbq import GbbqDataSource, GbbqError, GbbqRecord
from .tdx import TdxDataSource

__all__ = [
    "AdjustmentEvent",
    "GbbqDataSource",
    "GbbqError",
    "GbbqRecord",
    "MarketDataError",
    "TdxDataSource",
    "adjustment_factors",
    "backward_adjusted",
    "forward_adjusted",
]
