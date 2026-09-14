"""量能结构：按**摆动点**切出的上涨段与回调段，度量成交量在两端的变化。

为什么单独成模块：提示里的量能条件谈的是「上涨时怎么样、下跌时怎么样」，而那两段只有在段
边界定下来之后才有意义。段边界来自 :mod:`mbt.signals.swings`，故本模块的输入是**交割清单
式的** ``anchors``，而不是重新识别一遍拐点——同一次计算里，价格结构与量能结构必须吃同一份
边界，否则「回调缩量」缩的是哪一段都说不清。

三段与三个比值（``T`` 为起涨点，``P`` 为峰值，``S`` 为观测日）::

    起涨前基准 = [T − base_bars, T − 1]      上涨段 = [T, P]      回调段 = [P + 1, S]

    surge_ratio    = max(上涨段)      ÷ mean(起涨前基准)   放量倍数
    top_ratio      = mean(顶部 edge 根) ÷ mean(上涨段)      顶部相对上涨段
    pullback_ratio = mean(回调段)      ÷ max(上涨段)        缩量倍数

两个参照的取法不是随手定的，是实测出来的（九个样本，见下方各自说明）：

- ``surge_ratio`` 的分母取**起涨前基准**而不是上涨段自身——否则「放量」会变成「与自己比」。
- ``pullback_ratio`` 的分母取上涨段的**最大值**而不是均值。用均值时九个样本给出 0.51~2.30
  （中位 1.08，即「回调期与上涨期一样活跃」），用最大值时是 0.19~0.41（中位 0.32）。缩量是
  相对**那根爆量**才看得见的，相对均值看不见。
"""

from __future__ import annotations

import warnings
from typing import NamedTuple

import numpy as np
import pandas as pd

from mbt.signals.swings import Swings


class VolumeStructure(NamedTuple):
    """上涨段与回调段的量能结构，各为一条标的宽表。

    三个比值都**不是**过滤判断，只是数值——故它们照常遵循指标契约（列 = 时序、行 = 截面），
    方向各不同（``surge_ratio`` 越大越放量，``pullback_ratio`` 越小越萎缩），
    故**不要**直接当排序因子用（因子约定「越大越靠前」）。

    段边界未确认处一律缺失。
    """

    surge_ratio: pd.DataFrame
    top_ratio: pd.DataFrame
    pullback_ratio: pd.DataFrame


def _window_aggregates(
    vol: np.ndarray,
    start: int,
    end: int,
    columns: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """区间 ``[start, end]``（含两端）**只在 ``columns`` 这些列上**的 (均值, 最大值)。

    只算要用的那几列，而不是全部标的。这不是「少算一点」的近似：``np.nanmean`` /
    ``np.nanmax`` 沿 ``axis=0`` 归约，**每一列独立**，故限定列之后每一列的取值与整体算时
    **逐位相同**。而一次只为一两只标的算某个摆动对才是常态——按全部标的算，就把这两列的
    代价放大到了标的数的量级（全市场实测：这正是这一步从 5 秒涨到一个多小时的主因之一）。
    """
    window = vol[start : end + 1][:, columns]
    # 整段无有效值时**直接给 NaN**，不走 nanmean/nanmax——那两个在空切片上会发
    # RuntimeWarning。但只判「整段」不够：`np.nanmean(..., axis=0)` 只要**任一列**全缺失
    # 就会发 "Mean of empty slice" / "All-NaN slice"，而长期停牌的标的列正是这种情况。
    # 缺失是正常状态不是异常，故这里连警告一起压掉——不能靠 `errstate`（它管的是浮点
    # 状态，不是 warnings 模块）。
    if not np.isfinite(window).any():
        blank = np.full(columns.size, np.nan)
        return blank, blank.copy()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return np.nanmean(window, axis=0), np.nanmax(window, axis=0)


def _cached_aggregates(
    vol: np.ndarray,
    cache: dict[tuple[int, int], dict[int, tuple[float, float]]],
    start: int,
    end: int,
    columns: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """:func:`_window_aggregates` 的**按列缓存**版本，用于跨行会重复的窗口。

    为什么必须缓存：一个窗口的起止只由摆动点决定，而摆动点**连续多行不变**，故同一窗口会被
    反复问到。不缓存的话，每行每个摆动对都要发 4 次 numpy 调用——实测在 472 只上这让整步从
    1.6 秒变成 2.6 秒，**调用开销本身就是主项**（每次只算十来行、几列）。

    缓存粒度是**列**而不是整窗：同一窗口在不同行可能只被一部分标的问到（那些标的的摆动恰好
    走到了这一段），按整窗缓存就得整窗重算。代价是每列的记账走到 Python 字典里，比向量化慢，
    但比起「重算一整窗」仍是划算的。
    """
    slot = cache.get((start, end))
    if slot is None:
        slot = {}
        cache[(start, end)] = slot

    missing = [int(column) for column in columns if int(column) not in slot]
    if missing:
        fresh_mean, fresh_top = _window_aggregates(vol, start, end, np.array(missing))
        for position, column in enumerate(missing):
            slot[column] = (fresh_mean[position], fresh_top[position])

    size = columns.size
    return (
        np.fromiter((slot[int(c)][0] for c in columns), dtype=float, count=size),
        np.fromiter((slot[int(c)][1] for c in columns), dtype=float, count=size),
    )


def _group_columns(inverse: np.ndarray, slots: int) -> list[np.ndarray]:
    """把「每个元素属于哪个槽位」翻成「每个槽位有哪些元素」。

    返回的是**槽位内位置**（``0..len(inverse)-1``），不是标的列号——调用方拿它去索引
    ``index`` 才得到列号。这样分组与「元素代表什么」无关。

    写法上刻意避开 ``index[inverse == slot]``：那要对每个槽位扫一遍全部元素（``O(槽位×元素)``），
    而槽位数与元素数是同一量级——正是本次要消掉的那种二次开销。改成排一次序再切段，
    ``O(元素·log 元素)``。
    """
    order = np.argsort(inverse, kind="stable")  # 稳定排序：同一槽位内保持原顺序
    sorted_inverse = inverse[order]
    bounds = np.searchsorted(sorted_inverse, np.arange(slots + 1))
    return [order[bounds[slot] : bounds[slot + 1]] for slot in range(slots)]


def volume_structure(
    volumes: pd.DataFrame,
    anchors: Swings,
    *,
    edge_bars: int,
    base_bars: int,
) -> VolumeStructure:
    """度量**已完成的上涨段**与**其后的回调段**的量能结构。

    参数:
        volumes: 成交量的标的宽表，须与 ``anchors`` 出自**同一段行情**（同一个价格序列、
            同一个切片）。不同源会把边界对到别的日子上，而那种错不会报错。
        anchors: :func:`~mbt.signals.swings.swings` 的输出。它决定上涨段的起止，故
            「回调缩量」里的「回调段」也是它切出来的。
        edge_bars: 「上涨开始时」与「顶部」各取几根。两者共用一个参数是刻意的——它们要
            对称才能比较，而分成两个参数只会让人以为可以不对称。
        base_bars: 起涨前基准窗口的长度（如 ``10``）。

    返回:
        :class:`VolumeStructure`。

    缺失的三种来源，一律**缺失**而非填充：

    1. 拐点未确认（``anchors`` 在该行为缺失）——没有段就没有量能结构；
    2. 段太短——回调不足 1 根，或上涨段短于 ``edge_bars``（那时「起涨」与「顶部」会重叠）；
    3. 起涨前不足 ``base_bars`` 根——基准算不出来，放量倍数就无从谈起（序列开头即如此）。

    抛:
        ValueError: ``edge_bars`` / ``base_bars`` 不为正，或 ``volumes`` 与 ``anchors`` 的
            标的或交易日对不上（对不上会把段边界对到别的日子上，而那种错不会报错）。
    """
    if edge_bars < 1:
        raise ValueError(f"edge_bars 必须为正，收到 {edge_bars!r}")
    if base_bars < 1:
        raise ValueError(f"base_bars 必须为正，收到 {base_bars!r}")

    if not volumes.columns.equals(anchors.peak_age.columns):
        raise ValueError("volumes 与 anchors 的列（标的）必须一致，且顺序相同")
    if not volumes.index.equals(anchors.peak_age.index):
        raise ValueError("volumes 与 anchors 的索引（交易日）必须一致")

    vol = volumes.to_numpy(dtype=float)
    rows, columns = vol.shape
    peak_age = anchors.peak_age.to_numpy(dtype=float)
    trough_age = anchors.trough_age.to_numpy(dtype=float)

    #: 窗口 → {标的列: (均值, 最大值)}。只对起止由摆动点决定的窗口有意义，见
    #: :func:`_cached_aggregates`。
    cache: dict[tuple[int, int], dict[int, tuple[float, float]]] = {}
    surge = np.full((rows, columns), np.nan)
    top_ratio = np.full((rows, columns), np.nan)
    pullback_ratio = np.full((rows, columns), np.nan)

    for row in range(rows):
        peak_pos = np.where(np.isfinite(peak_age[row]), row - peak_age[row], -1.0)
        trough_pos = np.where(np.isfinite(trough_age[row]), row - trough_age[row], -1.0)
        usable = (
            (peak_pos >= 0.0)
            & (trough_pos >= 0.0)
            & (peak_pos >= trough_pos)
            & (row - peak_pos >= 1)  # 回调段非空
            & (peak_pos - trough_pos + 1 >= edge_bars)  # 「起涨」与「顶部」不重叠
            & (trough_pos - base_bars >= 0)  # 起涨前基准够长
        )
        index = np.flatnonzero(usable)
        if index.size == 0:
            continue

        trough_of = trough_pos[index].astype(int)
        peak_of = peak_pos[index].astype(int)
        # 按摆动对分组：同一对拐点落在多个标的上时，那几个标的共用一个窗口，故窗口只算一次。
        pairs, inverse = np.unique(
            np.stack([trough_of, peak_of], axis=1), axis=0, return_inverse=True
        )

        # 结果按「标的在 `index` 里的位置」排，故下面每个槽位只写回它自己那几个位置——
        # 而不是先铺一张 `(槽位数 × 标的数)` 的表再挑回来。那张表要写 `5 × 槽位 × 标的`
        # 个格子却只读回 `5 × 标的` 个，是本次消掉的另一处二次开销。
        base = np.empty(index.size)
        adv_mean = np.empty(index.size)
        adv_top = np.empty(index.size)
        edge = np.empty(index.size)
        pull = np.empty(index.size)

        # 分组只做一次（见 `_group_columns` 的说明：这就是不许写成 `inverse == slot` 的原因）。
        groups = _group_columns(inverse, pairs.shape[0])

        for slot, (trough, peak) in enumerate(pairs):
            positions = groups[slot]
            cols = index[positions]
            # 起涨前基准 / 上涨段 / 顶部：起止只由摆动点决定，跨行会重复，故走缓存。
            base[positions], _ = _cached_aggregates(
                vol, cache, int(trough) - base_bars, int(trough) - 1, cols
            )
            adv_mean[positions], adv_top[positions] = _cached_aggregates(
                vol, cache, int(trough), int(peak), cols
            )
            edge[positions], _ = _cached_aggregates(
                vol, cache, int(peak) - edge_bars + 1, int(peak), cols
            )
            # 回调段 `[peak+1, row]`：它的右端**就是当前行**，故这个窗口每天都是新的，
            # 缓存不会命中，只会把键越堆越多——只能直接算，不缓存。
            pull[positions], _ = _window_aggregates(vol, int(peak) + 1, row, cols)

        with np.errstate(invalid="ignore", divide="ignore"):
            surge[row, index] = np.where(base > 0.0, adv_top / base, np.nan)
            top_ratio[row, index] = np.where(adv_mean > 0.0, edge / adv_mean, np.nan)
            pullback_ratio[row, index] = np.where(adv_top > 0.0, pull / adv_top, np.nan)

    def frame(values: np.ndarray) -> pd.DataFrame:
        return pd.DataFrame(values, index=volumes.index, columns=volumes.columns, dtype=float)

    return VolumeStructure(
        surge_ratio=frame(surge),
        top_ratio=frame(top_ratio),
        pullback_ratio=frame(pullback_ratio),
    )


def volume_contraction(
    volumes: pd.DataFrame,
    anchors: Swings,
    *,
    edge_bars: int,
    base_bars: int,
    min_surge: float,
    max_pullback: float,
) -> pd.DataFrame:
    """「上涨放量、回调明显缩量」（**状态**）——提示 4 里站得住的那两条。

    两条判据（都取自 :func:`volume_structure`）::

        surge_ratio    >= min_surge      上涨段确实放过量
        pullback_ratio <= max_pullback   回调段相对爆量明显萎缩

    参数:
        volumes / anchors / edge_bars / base_bars: 同 :func:`volume_structure`。
        min_surge: 放量倍数下界（如 ``2.0``）。九个样本实测 1.21~15.07、中位 9.27，
            ``2.0`` 通过 8/9。
        max_pullback: 缩量倍数上界（如 ``0.5``）。九个样本实测 0.19~0.41、中位 0.32，
            ``0.5`` 通过 9/9。

    .. warning::

        **提示里的第三条（「上涨的顶部成交量没有继续放大，保持平量或缩量」）没有实现**，
        因为九个样本一致地**反过来**：爆量落在上涨段的**末尾**而非开头——量峰在上涨段内的
        相对位置是 0.72~1.00（中位 0.95，0 = 起涨点、1 = 峰值），9/9 都在后半段。
        顶部均量相对上涨段均量是 1.09~5.37 倍（中位 2.39），即顶部正是最活跃的一段。

        唯一能给出「9/9 通过」的写法是「顶部均量 ≤ 上涨段最大量」，但那**是同义反复**：
        顶部段是上涨段的子集，子集的均值不会超过全集的最大值。拿它当判据等于没判据。

        故这一条只剩两种出路：要么承认它与样本不符而放弃，要么换一个检验方式（例如比较
        顶部与前一轮顶部、或要求顶部不出现逐日递增）。两者都需要先有更多样本才能判断，
        在那之前不实现它，比实现一个恒真的判据诚实。

    缺失处一律 ``False``：段边界未确认、基准不足、缺口，都不给乐观答案（ADR-0005）。
    """
    structure = volume_structure(volumes, anchors, edge_bars=edge_bars, base_bars=base_bars)
    return (structure.surge_ratio >= min_surge) & (structure.pullback_ratio <= max_pullback)
