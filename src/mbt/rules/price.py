"""涨跌停价的算法（ADR-0002）。

它既是**撮合**的判据（一字板不可成交），也是**数据质检**的判据（越界的日间跳空要么是
公司行为、要么是坏数据）。两处必须用**同一个**算法，否则「撮合时认为没触限、质检时
却报异常」这种自相矛盾迟早会发生，故它落在共享的规则层，而非 `mbt.backtest` 内部。

## 必须用十进制半进位

涨跌停价 = 前收盘价 × (1 ± 限幅)，四舍五入到**分**。内建 ``round()`` 在二进制浮点上
运算，**恰好半分**时会少一分：前收 14.45 的跌停价应是 13.01，``round`` 给 13.00。

实测（``docs/research/tdx-halt-and-limit-representation.md`` 结论四）：本机 16 只主板
股票中两种算法给出不同限价的有 1,916 天（约 4.4%）；而在两者会分歧、且当日确实触及
限价的 38 个交易日里，市场**全部**落在十进制半进位一侧，无一例外。
"""

from __future__ import annotations

from decimal import ROUND_HALF_UP, Decimal

#: 价格的最小单位：分。
_CENT = Decimal("0.01")

#: 价比的容差。价格精确到分，此处仅为吸收 ``float`` 的表示误差，
#: 不是模糊阈值——判「是否触限」不应有模糊地带。
PRICE_TOLERANCE = 1e-9


def same_price(a: float, b: float) -> bool:
    """两个价格是否相等（容忍浮点表示误差）。"""
    return abs(a - b) < PRICE_TOLERANCE


def round_to_cent(value: float) -> float:
    """按**十进制**半进位取整到分，返回 ``float``。

    价格一律精确到分，且必须走十进制——理由同下（这也是「恰好半分」类偏差的唯一出口）。
    """
    return float(Decimal(str(value)).quantize(_CENT, rounding=ROUND_HALF_UP))


def limit_price(prev_close: float, limit: float, direction: int) -> float:
    """涨跌停价：``前收盘价 × (1 + 限幅 × 方向)``，四舍五入到分。

    参数:
        prev_close: 前一交易日收盘价，即限幅的计算基数。
        limit: 当日限幅（``0.10`` 表示 10%），由规则表按板块与 ST 状态给出。
        direction: ``+1`` 取涨停价，``-1`` 取跌停价。
    """
    price = Decimal(str(prev_close)) * (Decimal(1) + Decimal(str(limit)) * direction)
    return float(price.quantize(_CENT, rounding=ROUND_HALF_UP))


def limit_band(prev_close: float, limit: float) -> tuple[float, float]:
    """当日的 ``(跌停价, 涨停价)``。合法价格必须落在这一闭区间内。"""
    return (
        limit_price(prev_close, limit, -1),
        limit_price(prev_close, limit, +1),
    )
