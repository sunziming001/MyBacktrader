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


def _pick(values: np.ndarray, inverse: np.ndarray, size: int) -> np.ndarray:
    """从「按摆动对分行、按标的列排」的中间结果里，取回每个标的自己的那一格。

    写成模块级函数而不是循环内的闭包：闭包会捕获循环变量 ``inverse``，ruff 的 B023 正是指
    这种写法——当前代码在**同一次迭代内**就调用它，故结果是对的，但这种正确依赖调用时机，
    一旦有人把它挪出循环就会静默取到最后一轮的 ``inverse``。
    """
    return values[inverse, np.arange(size)]


def _segment_aggregates(
    vol: np.ndarray,
    cache: dict[tuple[int, int], tuple[np.ndarray, np.ndarray]],
    start: int,
    end: int,
) -> tuple[np.ndarray, np.ndarray]:
    """区间 ``[start, end]``（含两端）逐标的的 (均值, 最大值)。

    按 ``(start, end)`` 去重缓存。缓存值是一个**覆盖全部标的**的向量，故同一对拐点在两个
    标的上复用时，两者都取到各自列上的聚合值——缓存不会串列。

    去重是必要的：同一对拐点会连续多行不变，故实际计算次数等于**摆动次数**，
    而不是「K 线根数 × 标的数」——后者在全市场上是亿级。
    """
    key = (start, end)
    hit = cache.get(key)
    if hit is None:
        window = vol[start : end + 1]
        # 整段无有效值时**直接给 NaN**，不走 nanmean/nanmax——那两个在空切片上会发
        # RuntimeWarning。但只判「整段」不够：`np.nanmean(..., axis=0)` 只要**任一列**全缺失
        # 就会发 "Mean of empty slice" / "All-NaN slice"，而长期停牌的标的列正是这种情况。
        # 缺失是正常状态不是异常，故这里连警告一起压掉——不能靠 `errstate`（它管的是浮点
        # 状态，不是 warnings 模块）。
        valid = np.isfinite(window)
        if valid.any():
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                hit = (np.nanmean(window, axis=0), np.nanmax(window, axis=0))
        else:
            blank = np.full(vol.shape[1], np.nan)
            hit = (blank.copy(), blank.copy())
        cache[key] = hit
    return hit


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

    cache: dict[tuple[int, int], tuple[np.ndarray, np.ndarray]] = {}
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
        # 只在**这一行**去重：不同标的多半落在不同的摆动上，而同一摆动会被多行复用（缓存）。
        pairs, inverse = np.unique(
            np.stack([trough_of, peak_of], axis=1), axis=0, return_inverse=True
        )

        base_mean = np.empty((pairs.shape[0], index.size))
        advance_mean = np.empty_like(base_mean)
        advance_top = np.empty_like(base_mean)
        edge_mean = np.empty_like(base_mean)
        pullback_mean = np.empty_like(base_mean)
        for slot, (trough, peak) in enumerate(pairs):
            m_base, _ = _segment_aggregates(vol, cache, int(trough) - base_bars, int(trough) - 1)
            m_adv, x_adv = _segment_aggregates(vol, cache, int(trough), int(peak))
            m_edge, _ = _segment_aggregates(vol, cache, int(peak) - edge_bars + 1, int(peak))
            m_pull, _ = _segment_aggregates(vol, cache, int(peak) + 1, row)
            base_mean[slot] = m_base[index]
            advance_mean[slot] = m_adv[index]
            advance_top[slot] = x_adv[index]
            edge_mean[slot] = m_edge[index]
            pullback_mean[slot] = m_pull[index]

        base = _pick(base_mean, inverse, index.size)
        adv_mean = _pick(advance_mean, inverse, index.size)
        adv_top = _pick(advance_top, inverse, index.size)
        edge = _pick(edge_mean, inverse, index.size)
        pull = _pick(pullback_mean, inverse, index.size)

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
