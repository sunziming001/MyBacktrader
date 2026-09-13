"""排序因子：可在**同一时刻横向比较大小**的数值，回答「今天谁更靠前」（``CONTEXT.md``）。

与指标的区别不在算法，而在**用途**：指标是「某一标的的数值序列」，因子要被**同一行**
横向排序。故因子必须满足两条：

1. 数值大小有方向意义，且**越大越靠前**——方向写进文档，不靠调用方猜。需要反向排序的
   因子（如「距高点越近越好」）在定义时就取成越大越好，而不是让消费方翻符号。
2. 缺失就是缺失，不填零。填零会让「无数据」在截面排序里排到一个具体位置——那是造数据。
"""

from __future__ import annotations

import pandas as pd

from mbt.data.panel import Panel
from mbt.signals._symbol_frame import check_symbol_frame
from mbt.signals.indicators import rolling_max


def drawdown_from_high(panel: Panel, n: int) -> pd.DataFrame:
    """从 n 日**最高价**回落的幅度：``1 − 当根收盘 / n 日最高价``，恒 ``≥ 0``。**越大跌得越深**。

    一行内同时要用 ``high`` 与 ``close``，故收 :class:`~mbt.data.panel.Panel`（与 ``atr``
    同理）；返回的仍是**标的宽表**，消费方向不变。

    .. warning::

        **它与 :func:`distance_to_high` 不是同一个量，别混用**：

        ====================  ======================  ==================
        函数                   基准                    方向
        ====================  ======================  ==================
        ``distance_to_high``   **收盘价**的 n 日最高值    ``≤ 0``，「越接近 0 越强」
        ``drawdown_from_high`` **最高价**的 n 日最高值    ``≥ 0``，「越大跌得越深」
        ====================  ======================  ==================

        两者互为反向，且基准不同——``high`` 的最高值与 ``close`` 的最高值不是一回事。这看起来
        像重复，但「哪根 K 线创新高」与「从最高点跌了多少」在选股里是两个不同的条件，而本项目
        的约定是**方向写进函数名与文档**，不靠调用方翻符号。

    窗口取 ``[当日 − n + 1, 当日]``（含当日），不足 n 根处为**缺失**（不给乐观答案）。
    """
    return 1.0 - panel["close"] / rolling_max(panel["high"], n)


def momentum(prices: pd.DataFrame, n: int) -> pd.DataFrame:
    """n 日动量：``当根收盘 / n 根前收盘 − 1``。**越大越强**。"""
    prices = check_symbol_frame(prices)
    return prices / prices.shift(n) - 1.0


def distance_to_high(prices: pd.DataFrame, n: int) -> pd.DataFrame:
    """距 n 日最高**收盘价**的相对距离：``当根收盘 / n 日最高收盘价 − 1``，恒 ``≤ 0``。

    数值**越接近 0 越强**。

    基准是**收盘价**的最高值，不是 K 线的最高价（``high``）——两者不是一回事，而本层的
    输入约定是一张单一字段的标的宽表，故「最高价」在此一律指收盘价的最大值（与 ``new_high``
    的口径一致）。

    定义为「接近程度」而非「回撤幅度」，是为了让所有因子**同向**（越大越好）：消费方
    排序时不必为每个因子记住方向，从而不必在每处都猜一次符号。
    """
    return check_symbol_frame(prices) / rolling_max(prices, n) - 1.0
