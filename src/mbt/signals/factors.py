"""排序因子：可在**同一时刻横向比较大小**的数值，回答「今天谁更靠前」（``CONTEXT.md``）。

与指标的区别不在算法，而在**用途**：指标是「某一标的的数值序列」，因子要被**同一行**
横向排序。故因子必须满足两条：

1. 数值大小有方向意义，且**越大越靠前**——方向写进文档，不靠调用方猜。需要反向排序的
   因子（如「距高点越近越好」）在定义时就取成越大越好，而不是让消费方翻符号。
2. 缺失就是缺失，不填零。填零会让「无数据」在截面排序里排到一个具体位置——那是造数据。
"""

from __future__ import annotations

import pandas as pd

from mbt.signals._symbol_frame import check_symbol_frame
from mbt.signals.indicators import rolling_max


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
