"""信号层：指标、过滤信号、排序因子（ADR-0001）。

三者一律算成 **日期 × 标的** 的标的宽表——回测按列取时序，选股按行取截面，**同一个函数，
两个消费方向**。全部是纯函数：无 I/O、无状态、不读全局配置。

三类的返回类型是契约的一部分，不要混用：

- **指标**返回浮点数（缺失即缺失），不含判断；
- **过滤信号**返回纯 ``bool``，缺失取 ``False``（「不合格」）；
- **排序因子**返回浮点数，可在同一时刻**横向比较大小**——数值大小有方向意义。

大多数函数取**单字段**的标的宽表；``atr`` 取**行情面板**（跨字段），但**仍然返回标的宽表**
（ADR-0009）——消费方向不变，故它照常能当排序因子用。
"""

from __future__ import annotations

from mbt.signals.factors import distance_to_high, momentum
from mbt.signals.filters import (
    above_ma,
    ma_cross_up,
    new_high,
    rising_streak,
    volume_surge,
)
from mbt.signals.indicators import atr, rolling_max, sma, volume_ratio

__all__ = [
    "above_ma",
    "atr",
    "distance_to_high",
    "ma_cross_up",
    "momentum",
    "new_high",
    "rising_streak",
    "rolling_max",
    "sma",
    "volume_ratio",
    "volume_surge",
]
