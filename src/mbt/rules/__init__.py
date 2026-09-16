"""交易制度规则表：按「生效日期 × 板块」承载制度参数（ADR-0002）。

规则表以可配置数据文件形式存在，不硬编码在代码里；查表按**成交日**取当日生效的那一条。
本层只依赖标准库与 ``tomli``，**不依赖 backtrader**（ADR-0007）。
"""

from .board import board_of
from .errors import RuleTableError
from .price import (
    PRICE_TOLERANCE,
    limit_band,
    limit_price,
    limit_prices,
    round_to_cent,
    same_price,
)
from .table import SHIPPED_RULES_PATH, RuleTable

__all__ = [
    "PRICE_TOLERANCE",
    "SHIPPED_RULES_PATH",
    "RuleTable",
    "RuleTableError",
    "board_of",
    "limit_band",
    "limit_price",
    "limit_prices",
    "round_to_cent",
    "same_price",
]
