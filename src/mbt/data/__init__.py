"""数据层：通达信本地数据的解析与访问，以及复权视图。

本层只依赖 pandas 与 numpy，**不依赖 backtrader**（ADR-0007）。

复权遵循 ADR-0003：只保存 / 传递**原始价与复权事件**，已复权的价格序列一律**用时计算**
（见 :mod:`mbt.data.adjust`）。

要一份能直接拿去回测的行情，走 :func:`load_market_data`——它把行情目录、权息文件与
规则表组合起来，并在返回前做完越界检查（ADR-0005）。下面三个数据源是它的零件，
需要单独取某一样时才直接用。
"""

from .adjust import (
    AdjustmentEvent,
    adjustment_factors,
    backward_adjusted,
    combined_reference_price,
    forward_adjusted,
)
from .anomaly import Anomaly, find_anomalies, require_no_anomalies
from .errors import MarketDataError
from .fundamental import (
    CwDataSource,
    FinancialRecord,
    load_financials,
    non_loss_mask,
)
from .gbbq import GbbqDataSource, GbbqError, GbbqRecord
from .instrument import instrument_type, is_stock
from .loader import (
    SkippedSymbol,
    UniverseLoad,
    load_universe_data,
    slice_markets,
    stock_symbols,
)
from .market import MarketData, load_market_data
from .master import SecurityInfo, SecurityMasterDataSource, load_listing_dates
from .panel import Panel, assemble_panel
from .tdx import TdxDataSource
from .updates import (
    GapReport,
    SymbolBoundary,
    UpdateReport,
    boundary_of,
    check_updates,
    load_boundaries,
    save_boundaries,
)

__all__ = [
    "AdjustmentEvent",
    "Anomaly",
    "CwDataSource",
    "DilutionVerdict",
    "FinancialRecord",
    "GbbqDataSource",
    "GbbqError",
    "GbbqRecord",
    "MarketData",
    "MarketDataError",
    "Panel",
    "GapReport",
    "SecurityInfo",
    "SecurityMasterDataSource",
    "SkippedSymbol",
    "SymbolBoundary",
    "TdxDataSource",
    "UniverseLoad",
    "UpdateReport",
    "adjustment_factors",
    "assemble_panel",
    "backward_adjusted",
    "boundary_of",
    "check_updates",
    "combined_reference_price",
    "find_anomalies",
    "forward_adjusted",
    "instrument_type",
    "is_stock",
    "load_boundaries",
    "load_financials",
    "load_listing_dates",
    "load_market_data",
    "load_universe_data",
    "non_loss_mask",
    "require_no_anomalies",
    "save_boundaries",
    "slice_markets",
    "stock_symbols",
]
