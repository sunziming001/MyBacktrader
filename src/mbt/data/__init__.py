"""数据层：通达信本地数据的解析与访问。

本层只依赖 pandas 与 numpy，**不依赖 backtrader**（ADR-0007）。
"""

from .tdx import MarketDataError, TdxDataSource

__all__ = ["MarketDataError", "TdxDataSource"]
