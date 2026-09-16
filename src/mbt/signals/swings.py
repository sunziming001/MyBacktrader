"""摆动点：把价格序列切成「上涨段 / 下跌段」的**已确认**拐点（``CONTEXT.md`` 的**摆动点**）。

为什么需要它：提示里的「一波上涨」「调整期」都不是固定窗口能描述的东西——一波上涨可以走 14 根，
也可以走 40 根。把「上涨了多少」「回调了多深」「回调了几天」写成判据，前提是先知道那一段从哪
到哪，而那个边界只能由价格自身的拐点给出。

**时点正确性是本模块最要紧的事**。一个拐点只有在价格从它回撤超过阈值**之后**才能被确认，
故「峰值在 ``t`` 日」这件事要到某个 ``t' > t`` 才可知。若在 ``t`` 日就报出峰值，那就是**前视
偏差**——而且它不报错，只让回测结果偏乐观（ADR-0006）。故本模块的输出一律**只报告已确认的
拐点**，并把「距今多少根」一并给出：消费者据此知道这个拐点在被确认时已经过去了多久。

半径的取舍也在这里：``retracement`` 越小，拐点确认得越早、也越容易把噪声当成拐点；越大则
越晚确认、但越可靠。它没有默认值——这是需要按标的波动性标定的量，藏进默认值里就没法标定了。
"""

from __future__ import annotations

from typing import NamedTuple

import numpy as np
import pandas as pd

from mbt.signals._symbol_frame import check_symbol_frame

#: ``Swings`` 的四个字段，顺序与构造处一致。写成常量而不是在两处各列一遍字符串——
#: 那样加一个字段时只会改对一处（``NamedTuple`` 的字段名与这里的顺序对不上时不会报错）。
_SWING_FIELDS = ("peak_price", "peak_age", "trough_price", "trough_age")


class Swings(NamedTuple):
    """**最近一段已完成的上涨**的锚点，``peak_*`` 与 ``trough_*`` 是一对。

    两条约定要写明：

    1. ``trough_*`` 指的是**与当前峰值配对的那个起涨点**，不是「最近确认的低点」。两者在
       价格已经转头向下、且回撤又确认了一个更低的低点之后会不同——那时那个更低的低点描述的是
       **正在形成**的下一段上涨，而本结构描述的是**已完成**的那一段。判据要的是后者，故配对
       只在峰值确认的那一刻建立一次，其后不因新低点而改变。
    2. 四条输出**同时**缺失或**同时**有值。只有终点没有起点的一段上涨无从度量（「涨了多少」
       需要两端），故宁可整段不报，也不给调用方留下一个要自己猜的半截状态。序列首段自最低点
       直线上涨、起点从未被确认为低点时，就会落到这一情形。

    ``age`` 是自该拐点起经过的**根数**（``0`` 表示当根就是那个拐点），它随每根递增，
    故「回调了几天」「峰值是多久以前的」都不必再算日期。
    """

    peak_price: pd.DataFrame
    peak_age: pd.DataFrame
    trough_price: pd.DataFrame
    trough_age: pd.DataFrame


def _swing_columns(values: pd.Series, retracement: float) -> dict[str, np.ndarray]:
    """对单个标的做一次顺序扫描，产出 :class:`Swings` 的四列。

    状态机：``direction`` 为 ``0``（未定向）/ ``+1``（上涨腿）/ ``-1``（下跌腿）；``anchor``
    是当前腿的极值候选。上涨腿中价格从候选高点回撤超过 ``retracement`` 即**确认**该高点为峰值，
    下跌腿中反之确认低点。只有被确认过的点才会写进输出。

    扫描是顺序的且只读当根及之前，故「截断重算不变性」天然成立（见
    ``tests/test_signal_contract.py``）——未来的一根 K 线既不能改变过去的状态，也不能改变
    过去报出的拐点。

    缺口（``NaN``）当根：**状态不动，当根不报**——过去的价格仍然已知，故拐点结构不被一段
    停牌抹掉；但当根的价未知，故当根拒绝给出「距峰值多少根」这类判断，一律缺失。
    """
    close = values.to_numpy(dtype=float)
    n = close.size
    peak_price = np.full(n, np.nan)
    peak_age = np.full(n, np.nan)
    trough_price = np.full(n, np.nan)
    trough_age = np.full(n, np.nan)

    # 两个乘式提出来：它们只由 `retracement` 决定，而循环要跑几千万次。
    up = 1.0 + retracement
    down = 1.0 - retracement

    direction = 0
    anchor_bar = -1
    anchor_price = float("nan")
    last_trough: tuple[int, float] | None = None  # 最近被确认的低点（滚动更新）
    confirmed_peak: tuple[int, float] | None = None  # 最近被确认的峰值
    paired_trough: tuple[int, float] | None = None  # **与 confirmed_peak 配对**的起涨点

    for t in range(n):
        price = close[t]
        # `price != price` 就是 NaN 判据，但它是**纯 Python 比较**；`np.isnan(price)` 走的是
        # numpy 标量调用，实测贵 4 倍以上，而这里要判几千万次（占循环耗时的约 80%）。
        if price != price:
            continue  # 当根无价：状态不动，当根不报

        if direction == 0:
            if anchor_bar < 0:
                anchor_bar, anchor_price = t, price
            elif price > anchor_price * up:
                direction = 1
                anchor_bar, anchor_price = t, price
            elif price < anchor_price * down:
                direction = -1
                anchor_bar, anchor_price = t, price
        elif direction == 1:
            if price > anchor_price:
                anchor_bar, anchor_price = t, price  # 抬高候选峰
            elif price < anchor_price * down:
                confirmed_peak = (anchor_bar, anchor_price)
                # 配对只在**峰值确认时**建立一次。此后若价格继续下跌、又确认了一个更低的
                # 低点，那个低点属于「正在形成的下一段上涨」，不能顶掉这里已经配好的起涨点
                # ——否则已完成的这一段会被重新描述成幅度更大的一段。
                paired_trough = last_trough
                direction = -1
                anchor_bar, anchor_price = t, price
        else:  # 下跌腿
            if price < anchor_price:
                anchor_bar, anchor_price = t, price  # 压低候选谷
            elif price > anchor_price * up:
                last_trough = (anchor_bar, anchor_price)
                direction = 1
                anchor_bar, anchor_price = t, price

        # 峰值与起涨点**同时**发布：只有终点没有起点的一段上涨无从度量，
        # 故宁可整段不报，也不给调用方留下一个要自己猜的半截状态。
        if confirmed_peak is not None and paired_trough is not None:
            peak_price[t] = confirmed_peak[1]
            peak_age[t] = t - confirmed_peak[0]
            trough_price[t] = paired_trough[1]
            trough_age[t] = t - paired_trough[0]

    return {
        "peak_price": peak_price,
        "peak_age": peak_age,
        "trough_price": trough_price,
        "trough_age": trough_age,
    }


def swings(prices: pd.DataFrame, retracement: float) -> Swings:
    """找出每个标的**最近一段已完成上涨**的起涨点与峰值（**指标**，各为标的宽表）。

    参数:
        prices: 收盘价的标的宽表。拐点定义在收盘价上（不取最高/最低价）——那样判据只需一个
            字段，且「收盘价创出的高点」在实务上就是画线用的那个点。
        retracement: 确认拐点所需的回撤幅度，取 ``(0, 1)`` 的开区间（如 ``0.15``）。
            它越小确认越早、也越容易把噪声当拐点；越大越晚、越可靠。

    返回:
        :class:`Swings`（``peak_price`` / ``peak_age`` / ``trough_price`` / ``trough_age``
        各为一条标的宽表）。峰值确认之前一律缺失。

    抛:
        ValueError: ``retracement`` 不在 ``(0, 1)`` 内，或输入不是合法标的宽表。

    .. note::

        本函数是**顺序扫描**（每个标的一遍），不是窗口算子。故它比 :func:`~mbt.signals.sma`
        之类的 ``rolling`` 慢，且速度与序列长度成正比。这是拐点定义带来的固有代价：一个点
        「是不是峰值」取决于它之后回撤了多少，无法用固定窗口表达。
    """
    if not 0.0 < retracement < 1.0:
        raise ValueError(f"retracement 必须落在 (0, 1) 内，收到 {retracement!r}")

    frame = check_symbol_frame(prices)
    rows, width = frame.shape

    # **每个标的状态机只跑一次**，四个字段一次取出。此前写成
    #     {field: pd.DataFrame({name: _swing_columns(...)[field] for name in ...}) for field in 四}
    # 于是 `_swing_columns` 被调了 **4 × 标的数** 次——状态机白跑三遍（全市场就是 1.16 亿次
    # 逐根迭代，而它一遍就够）。实测（302 列 × 7,124 行）本函数由 4.03s 降到 1.2s。
    #
    # 组装也一并改了：先分配 `(行, 列)` 的 ndarray 再逐列填入，最后各包一张 DataFrame。
    # 原来按「字段 → {列名: 数组} 的字典」建表，会把 4 × 标的数 个 Series 逐列插进 DataFrame
    # ——那是 pandas 的碎片化路径，列一多就非线性变慢。
    out = {field: np.full((rows, width), np.nan) for field in _SWING_FIELDS}
    for position, name in enumerate(frame.columns):
        columns = _swing_columns(frame[name], retracement)
        for field in _SWING_FIELDS:
            out[field][:, position] = columns[field]

    return Swings(
        **{
            field: pd.DataFrame(out[field], index=frame.index, columns=frame.columns, dtype=float)
            for field in _SWING_FIELDS
        }
    )
