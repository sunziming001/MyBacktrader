"""回测层：引擎协议、撮合与产物。

引擎（backtrader）被隔离在本层之后，数据层与信号层不得依赖它（ADR-0007）。
"""

from . import _btcompat  # noqa: F401  导入即应用 Python 3.10 兼容修补
from .engine import (
    BacktestResult,
    build_tradability,
    run_backtest,
    run_portfolio_backtest,
)
from .sizing import AStockSizer, EqualWeightSizer, FixedAmountSizer

__all__ = [
    "AStockSizer",
    "BacktestResult",
    "EqualWeightSizer",
    "FixedAmountSizer",
    "build_tradability",
    "run_backtest",
    "run_portfolio_backtest",
]
