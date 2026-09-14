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

from mbt.signals.factors import (
    distance_to_high,
    drawdown_from_high,
    j_oversold,
    momentum,
    reward_risk_ratio,
    yellow_proximity,
)
from mbt.signals.filters import (
    above_ma,
    above_white,
    above_yellow,
    below_white,
    below_yellow_streak,
    high_above_white,
    j_below,
    ma_cross_up,
    new_high,
    no_contained_run,
    pullback_after_advance,
    rising_streak,
    volume_surge,
    white_above_yellow,
)
from mbt.signals.indicators import (
    KDJ,
    atr,
    ema,
    kdj,
    rolling_max,
    rolling_min,
    sma,
    volume_ratio,
    white_line,
    yellow_line,
)
from mbt.signals.swings import Swings, swings
from mbt.signals.volume import VolumeStructure, volume_contraction, volume_structure

__all__ = [
    "KDJ",
    "Swings",
    "VolumeStructure",
    "above_ma",
    "above_white",
    "above_yellow",
    "atr",
    "below_white",
    "below_yellow_streak",
    "distance_to_high",
    "drawdown_from_high",
    "ema",
    "high_above_white",
    "j_below",
    "j_oversold",
    "kdj",
    "ma_cross_up",
    "momentum",
    "new_high",
    "no_contained_run",
    "pullback_after_advance",
    "reward_risk_ratio",
    "rising_streak",
    "rolling_max",
    "rolling_min",
    "sma",
    "swings",
    "volume_contraction",
    "volume_ratio",
    "volume_structure",
    "volume_surge",
    "white_above_yellow",
    "white_line",
    "yellow_line",
    "yellow_proximity",
]
