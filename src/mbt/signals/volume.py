"""量能结构：按**摆动点**切出的上涨段与回调段，度量成交量在两端的变化。

为什么单独成模块：提示里的量能条件谈的是「上涨时怎么样、下跌时怎么样」，而那两段只有在段
边界定下来之后才有意义。段边界来自 :mod:`mbt.signals.swings`，故本模块的输入是**交割清单
式的** ``anchors``，而不是重新识别一遍拐点——同一次计算里，价格结构与量能结构必须吃同一份
边界，否则「回调缩量」缩的是哪一段都说不清。

三段与四个比值（``T`` 为起涨点，``P`` 为峰值，``S`` 为观测日）::

    起涨前基准 = [T − base_bars, T − 1]      上涨段 = [T, P]      回调段 = [P + 1, S]

    surge_ratio           = max(上涨段)        ÷ mean(起涨前基准)   放量倍数
    top_ratio             = mean(顶部 edge 根) ÷ mean(上涨段)       顶部相对上涨段
    pullback_ratio        = mean(回调段)       ÷ max(上涨段)        缩量倍数（相对**单日爆量**）
    pullback_vs_top_ratio = mean(回调段)       ÷ mean(顶部 edge 根) 缩量倍数（相对**顶部段**）

参照物的取法分两层，两层的依据不同，别混起来：

1. **分母是「哪一段」**：``surge_ratio`` 取**起涨前基准**而不是上涨段自身——否则「放量」会
   变成「与自己比」。缩量那一条取**上涨段**（``pullback_ratio``，分母是该段里的单日最大量）。
   另一条路是取**顶部段**（``pullback_vs_top_ratio``）：它更贴原文那句「在上涨的顶部…**而在
   下跌阶段**，成交量会明显萎缩」里的「而」字，在九个**用户亲自标注**的样本上 9/9 成立。但它
   **全市场对照更差**——逐笔 8,601 → 3,145、收益率均值 +0.164% → +0.116%、均值/标准误
   3.90 → 1.68，而被它排掉的 5,969 笔反而更好（均值 +0.190%、均值/标准误 +3.74）。故
   **没有采用**它，只作为数据保留（ADR-0011）。
2. **分母取哪个统计量**：``pullback_ratio`` 取上涨段的**最大值**。这个 ``max`` 是历史遗留：
   当初拿九个样本把分母从均值换成最大值，使样本看起来像缩量——而那九个样本在均值口径下本来
   就不显示缩量（中位 1.08）。那次取舍没有任何记录，直到 1,109 笔真实成交回头检验才发现问题
   （53% 的成交按其字面含义并不缩量，ADR-0010）。

.. warning::

    本模块的**口径与判据在 2026-09-15 改过一轮**（ADR-0012），最后的落点是：

    - **新口径**的缩量（``volume_pattern`` 的 ``pullback_vs_advance``）与「顶部无量」「顶部
      那根的长上影」一起合成排序用的**形态分数**，它**不再当门**；
    - 本模块这四个**旧口径**比值里，``surge_ratio`` 与 ``pullback_ratio`` **仍然是门**
      （:func:`volume_contraction` 的两半）。③ 曾把这道门一并换成新口径的 ``surge_vs_base``、
      并去掉缩量那半条，④ 的全市场对照把那处连带改动量出来是**单独 −7.53 pt**、且它没有
      独立证据，故**已退回**旧口径；
    - 「顶部」在**新口径**里是**一根**（最高价所在那根），故新口径那两个缩量比值的分母是
      **上涨段除去顶部那根**；旧口径的「顶部」是上涨段末尾 ``edge_bars`` 根，分母是整段。

    旧口径这四个比值保留，是为了让 ADR-0010/0011 的那些对照可复算；新口径的读数另在
    :func:`volume_pattern` 里，**字段不同名**。

    ``pullback_ratio``（分母是单日最大量）**不可**读作「回调比上涨安静」：实测 1,109 笔成交里
    53% 按其字面含义并不缩量，上涨段「最大量 ÷ 均量」的中位是 2.95 倍（ADR-0010）。也**不要**
    把它与 ``pullback_vs_top_ratio`` 当同义词——两者秩相关只有约 0.48，且后者当判据时更差
    （ADR-0011）。两条独立对照都指向同一处：按字面读法，**回调期并不比上涨期安静**——换用
    「上涨段除去顶部那根的平均量」作分母后中位是 1.074（ADR-0012），与 ADR-0010 用另一条管线
    量到的 1.044 互证。

新口径在 :func:`volume_pattern` 里另起一组字段（``surge_vs_base`` / ``top_calm`` /
``pullback_vs_advance`` / ``top_shadow_atr``），**不复用**上面这四个名字：两组同名不同义的
比值在这个项目里已经被误读过一次（ADR-0010），改口径时换名字是刻意的。
"""

from __future__ import annotations

import math
import warnings
from typing import NamedTuple

import numpy as np
import pandas as pd

from mbt.data.panel import Panel
from mbt.signals.indicators import atr
from mbt.signals.swings import Swings


class VolumeStructure(NamedTuple):
    """上涨段与回调段的量能结构，各为一条标的宽表。

    四个比值都**不是**过滤判断，只是数值——故它们照常遵循指标契约（列 = 时序、行 = 截面），
    方向各不同（``surge_ratio`` 越大越放量，``pullback_ratio`` 越小越萎缩），
    故**不要**直接当排序因子用（因子约定「越大越靠前」）。

    两个「缩量」比值的差别只在**分母**，而它们不是彼此的换标度（实测秩相关约 0.48）：

    - ``pullback_ratio``：分母是上涨段的**单日最大量**。它答的是「相对那根爆量，回调是否萎缩」，
      容易被单根异常量操纵（ADR-0010）。**判据用的就是它。**
    - ``pullback_vs_top_ratio``：分母是**顶部段均量**。它答的是「回调是否比它前面那段顶部安静」，
      即原始提示里那个「而」字指向的比较；在 `[P−edge_bars+1, P]` 这个**窗口**上九个样本 9/9 成立
      （那九张截图只标注日期、未标出顶部，窗口是分析者按 `edge_bars=3` 切的），比「相对上涨段
      **均**量」的 2/9 好得多。**但全市场对照更差**（ADR-0011），故只作数据保留、不作判据。

    段边界未确认处一律缺失。
    """

    surge_ratio: pd.DataFrame
    top_ratio: pd.DataFrame
    pullback_ratio: pd.DataFrame
    pullback_vs_top_ratio: pd.DataFrame


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
    # 缺失是正常状态不是异常，故警告要被压掉——不能靠 `errstate`（它管的是浮点状态，
    # 不是 warnings 模块）。**压制不在这里做**：见 `volume_structure` 的说明（每次窗口
    # 进出一次上下文，实测占本函数耗时的约 5%，而它本该是一次性的）。
    if not np.isfinite(window).any():
        blank = np.full(columns.size, np.nan)
        return blank, blank.copy()
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

    .. note::

        **警告压制在这里做一次，不在每个窗口做一次。** ``np.nanmean`` / ``np.nanmax`` 在
        「某一列全无有效值」（长期停牌）时会发 ``RuntimeWarning``，而那是**正常状态**不是
        异常，故必须压掉。但它曾写在最热的那个窗口函数里，于是每个窗口都要进出一次
        ``warnings.catch_warnings()``——实测 36,586 次进出占本函数耗时的约 **5%**，
        而它本该是一次性的。挪到这一层之后，同一次调用里只进出一次，且它顺带覆盖了
        ``_window_aggregates`` 的任何被包装/替换版本（警告过滤器是进程级的）。
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return _volume_structure(volumes, anchors, edge_bars=edge_bars, base_bars=base_bars)


def _volume_structure(
    volumes: pd.DataFrame,
    anchors: Swings,
    *,
    edge_bars: int,
    base_bars: int,
) -> VolumeStructure:
    """:func:`volume_structure` 的实现体——警告压制由调用方负责，理由见那里的说明。"""
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
    pullback_vs_top_ratio = np.full((rows, columns), np.nan)

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
            # 分母用**顶部段均量**（`edge`）而不是单根最大值：它答的是「回调比它前面那段顶部
            # 安静吗」，即原始提示里那个「而」所指向的比较。`edge` 与 `pull` 上面都已算出，
            # 故这一步不增加任何扫描成本。
            pullback_vs_top_ratio[row, index] = np.where(edge > 0.0, pull / edge, np.nan)

    def frame(values: np.ndarray) -> pd.DataFrame:
        return pd.DataFrame(values, index=volumes.index, columns=volumes.columns, dtype=float)

    return VolumeStructure(
        surge_ratio=frame(surge),
        top_ratio=frame(top_ratio),
        pullback_ratio=frame(pullback_ratio),
        pullback_vs_top_ratio=frame(pullback_vs_top_ratio),
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

        surge_ratio            >= min_surge      上涨段确实放过量
        pullback_ratio         <= max_pullback   回调段相对上涨段**单日最大量**明显萎缩

    第二条看的是 ``pullback_ratio``（÷ 上涨段单日最大量），**不是** ``pullback_vs_top_ratio``
    （÷ 顶部段均量）——后者在九个样本上更贴原话，但**全市场对照更差**，且它排掉的那批反而更好
    （ADR-0011）。两者实测秩相关只有约 0.48，故不可互当同义词（ADR-0010）。

    参数:
        volumes / anchors / edge_bars / base_bars: 同 :func:`volume_structure`。
        min_surge: 放量倍数下界（如 ``2.0``）。九个样本实测 1.21~15.07、中位 9.27，
            ``2.0`` 通过 8/9。
        max_pullback: 缩量倍数上界——量的是「回调均量 ÷ **上涨段单日最大量**」。按字面读法它
            **不代表「回调比上涨安静」**：实测 1,109 笔成交里 53% 按字面含义并不缩量（ADR-0010）。

    .. warning::

        **本函数在 2026-09-15 一度被移出这道门**（ADR-0012 的 ③：缩量降级成分数、门只留
        「上涨放量」，且换算成新口径的 ``surge_vs_base``）。④ 的全市场对照把那处连带改动
        量出来是**单独 −7.53 pt**、且它本来就没有独立证据，故**已退回本函数**——这道门仍是
        这里的两半。

        「缩量降级成分数」那句针对的是**新口径**的 ``pullback_vs_advance``（见
        :func:`volume_pattern`），与本函数的 ``pullback_ratio`` **不是同一个统计量**，
        故两者并不冲突：分母一个是「上涨段除去顶部那根的均量」，一个是「上涨段单日最大量」。

    .. warning::

        **提示里的第三条（「上涨的顶部成交量没有继续放大，保持平量或缩量」）没有实现**，
        因为九个样本的**窗口读数**反过来：爆量落在上涨段的**末尾**而非开头——量峰在上涨段内的
        相对位置是 0.72~1.00（中位 0.95，0 = 起涨点、1 = 峰值），9/9 都在后半段。
        顶部均量相对上涨段均量是 1.09~5.37 倍（中位 2.39），即顶部正是最活跃的一段。

        （复核过的是「顶部均量 ÷ 上涨段均量」在 `[P−edge_bars+1, P]` 这个**窗口**上的读数：
        ≤ 1 的有 **0/9**、中位 2.39。ADR-0011 原把这一条写成「不是样本量的问题——标注本身
        如此」，**那是错的**：窗口是分析者按 `edge_bars=3` 切的，而那九张截图只标注日期、
        未标出顶部。故 0/9 是**那个窗口**的性质，取多宽就会跟着变。）

        唯一能给出「9/9 通过」的写法是「顶部均量 ≤ 上涨段最大量」，但那**是同义反复**：
        顶部段是上涨段的子集，子集的均值不会超过全集的最大值。拿它当判据等于没判据。

        故这一条只剩两种出路：要么承认它与窗口不符而放弃，要么换一个检验方式（例如比较
        顶部与前一轮顶部、或要求顶部不出现逐日递增）。两者都需要先有更多样本才能判断，
        在那之前不实现它，比实现一个恒真的判据诚实。

    缺失处一律 ``False``：段边界未确认、基准不足、缺口，都不给乐观答案（ADR-0005）。
    """
    structure = volume_structure(volumes, anchors, edge_bars=edge_bars, base_bars=base_bars)
    return (structure.surge_ratio >= min_surge) & (structure.pullback_ratio <= max_pullback)


class VolumePattern(NamedTuple):
    """新口径下的量能形态读数（ADR-0012）——**排序因子，不是门**。

    五条宽表同形（行 = 交易日，列 = 标的），方向各不同，故**不要**直接当排序因子用——因子
    约定「越大越靠前」，而这五条里 ``top_calm`` / ``pullback_vs_advance`` / ``top_shadow_atr``
    都是**越小越好**。合成分数的写法见 :func:`~mbt.screen.Screen` 的 ``factors`` 与
    ``normalize="rank"``（各自逐日秩归一后加权求和，秩会统一方向）。

    每个比值的方向与「分母取哪一段」::

        surge_vs_base        max(上涨段) ÷ mean([T−base, T−1])   越大越放量（旧口径统计量）
        top_calm             顶量 ÷ max(上涨段除去顶)            越**小**越「顶部无量」
        pullback_vs_advance  mean([R+1, S]) ÷ mean(上涨段除去顶) 越**小**越「调整缩量」
        top_shadow_atr       顶部的上影 ÷ ATR(顶)                越**小**越好（长上影扣分）
        top_after_peak       顶部那根在收盘峰值 P 之后几根        诊断用，不参与打分

    后三个的分母都取「**上涨段除去顶部那根**」（``[T, 顶−1] ∪ [顶+1, R]``），而不是整段。
    这不是精致化：顶部那根往往就是上涨段里量最大的那根（九个样本里量峰落在上涨段后半段，
    中位相对位置 0.95），把它留在分母里，分母就被那一根抬起来，于是分子（顶部量）与分母
    **共用了同一个数**——「顶部无量」会退化成常数附近的一个比值，量不出任何东西。

    ``surge_vs_base`` 是例外：它的分母是**起涨前基准**（与「上涨段」比会变成与自己比），
    故不含顶部那根的问题；但它仍按旧口径的统计量取**单日最大量**，只是段右端从 ``P`` 延到
    ``max(P, 顶)``。它**不是** ③ 用的那道门——门仍是 :func:`volume_contraction`（ADR-0012
    决定「放量」这道门本轮不动），这里给出来只是为了与 ② 的读数对得上。

    （③ 接线时曾把门临时换成这一项并去掉缩量那半条，④ 的全市场对照把那处连带改动量出来是
    单独 −7.53 pt，且它没有独立证据——故已退回 :func:`volume_contraction`。本字段保留，
    它仍是 ② 的读数口径，`probe_form_score.py` 要用。）

    ``top_after_peak`` 是给「收盘价峰值不是最高价」这件事留的账：实测**候选形态**里
    有 **31.2%** 的最高价落在 ``P`` 之后（``probe_b1_top_late.py``），即冲高被砸回来的
    「假突破」占比不低。上涨段的右端因此取 ``max(P, 顶部那根)``，让上涨段与回调段
    不重叠也不漏掉那一段。
    """

    surge_vs_base: pd.DataFrame
    top_calm: pd.DataFrame
    pullback_vs_advance: pd.DataFrame
    top_shadow_atr: pd.DataFrame
    top_after_peak: pd.DataFrame


def _top_of_advance(high: np.ndarray, trough_pos: np.ndarray) -> np.ndarray:
    """上涨段 ``[T, 本行]`` 内**最高价**所在那根的行号；段无效处为 ``-1``。

    「顶部」按**最高价**取一根，而不是按收盘价取（收盘口径的峰值由 :func:`~mbt.signals.
    swings.swings` 给出）。两者会分叉：冲高回落的 `high` 落在下一根阴线里，收盘口径看不见它。

    逐列扫描而不是逐格扫描：同一列上 ``T`` 大多连续多行不变，那时只需拿本行的 ``high``
    与已知的最大值比一次，不必重扫整段。``T`` 一变（新的低点被确认）才重扫——`high` 逐行
    推进，故重扫的总代价是 O(行数·列数)，不是 O(段长·行数)。实测全市场约 4 秒。
    """
    rows, columns = high.shape
    top = np.full((rows, columns), -1, dtype=np.int64)
    for column in range(columns):
        current = -1
        best = -1.0
        best_at = -1
        for row in range(rows):
            trough = trough_pos[row, column]
            if trough < 0.0:
                current, best, best_at = -1, -1.0, -1
                continue
            trough = int(trough)
            if trough != current:
                current = trough
                window = high[trough : row + 1, column]
                if np.isfinite(window).any():
                    best_at = trough + int(np.nanargmax(window))
                    best = float(high[best_at, column])
                else:
                    best, best_at = -1.0, -1
            else:
                value = float(high[row, column])
                if math.isfinite(value) and (best_at < 0 or value > best):
                    best, best_at = value, row
            top[row, column] = best_at
    return top


def _finite_stats(values: np.ndarray) -> tuple[float, float, int]:
    """一维切片**忽略缺失**后的 ``(和, 最大值, 有效个数)``。

    空集给 ``(0.0, nan, 0)``：**和给 0 而最大值给 nan** 是刻意的，不是随手写的。上涨段被顶部
    那根切成两半之后要把两半**相加再相除**，故空的那一半必须贡献 0——若那里给 nan，一次加法
    就会把整段的均值污染成 nan（这个坑踩过一次：``pullback_vs_advance`` 有 274 格因此变空）。
    而空集**没有**最大值，给 0 会让 ``top_calm`` 的分母变成 0、进而让比值变成一个假的 0。

    刻意不用 ``np.nanmean`` / ``np.nanmax``：那两个在空切片上会发 ``RuntimeWarning``，而缺失
    是正常状态不是异常；且这里要把「和 / 个数」分开拿，两个整体统计量不是它俩的接口。
    """
    finite = np.isfinite(values)
    count = int(finite.sum())
    if count == 0:
        return 0.0, math.nan, 0
    kept = values[finite]
    return float(kept.sum()), float(kept.max()), count


def volume_pattern(
    panel: Panel,
    anchors: Swings,
    *,
    base_bars: int,
    atr_n: int,
) -> VolumePattern:
    """度量新口径（ADR-0012）下的形态读数——**喂排序分数用，不当门**。

    段边界（``T`` 起涨点、``P`` 收盘口径峰值、``顶`` 最高价所在那根、``S`` 观测日）::

        起涨前基准 = [T − base_bars, T − 1]
        上涨段     = [T, max(P, 顶)]
        上涨段除顶 = [T, 顶 − 1] ∪ [顶 + 1, max(P, 顶)]
        回调段     = [max(P, 顶) + 1, S]

    参数:
        panel: 行情面板，须含 ``high`` / ``open`` / ``close`` / ``volume``。
        anchors: :func:`~mbt.signals.swings.swings` 的输出，须与 ``panel`` 出自**同一段
            行情**（同一个价格序列、同一个切片）。不同源会把边界对到别的日子上，而那种错
            不会报错。
        base_bars: 起涨前基准的长度（如 ``10``）。九个样本的实际取值是 15，但 10 已足够
            稳住基准——这一条从没成为瓶颈，故取这个值只是为了与 ADR-0010 的对照口径一致。
        atr_n: ATR 的窗口（如 ``14``），只用于把「上影」按波动率折算成可跨标的比较的量。

    返回:
        :class:`VolumePattern`。

    缺失的四种来源，一律**缺失**而非填充（产物是分数输入，缺值会被秩归一当作「最差」）：

    1. 拐点未确认（``anchors`` 在该行为缺失）——没有段就没有形态；
    2. 起涨前不足 ``base_bars`` 根——``surge_vs_base`` 无从算起（其余比值仍可算）；
    3. 上涨段只有顶部那根（``max(P, 顶) == T``）——「除去顶部那根」是空集，
       ``top_calm`` / ``pullback_vs_advance`` 都没有分母；
    4. 回调段为空（``S == max(P, 顶)``）——``pullback_vs_advance`` 没有分子。

    抛:
        ValueError: ``base_bars`` / ``atr_n`` 不为正，``panel`` 缺字段，或它与 ``anchors``
            的标的或交易日对不上。

    .. note::

        **为什么逐格算，而不像** :func:`volume_structure` **那样按摆动对分组。** 那里一个
        ``(T, P)`` 会同时落在多个标的上，分组能省掉重复窗口。这里不行：``(T, P, 顶, R)``
        这个四元组实测**每组平均只有 1.03 列**（``probe_monotonic.py``），等于没分组。

        也**不能**用「摆动点只往前走」做单调队列：实测 ``trough_pos`` / ``peak_pos`` 在相邻
        两行**都有效**处有 **44%** 是**下降**的——未确认的摆动点会随新数据**回头修正**。

        真正省下功夫的是**遍历顺序**：按**列优先**走（同一列的行连续），于是 ``T`` / ``顶`` /
        ``R`` 各自连续多行不变，三个「起止只由摆动点决定」的窗口（起涨前基准、上涨段的左右两
        半）只需记住上一次的结果，不比一次就够了。

        剩下**回调段**那一项缓存不住——它的右端就是当前行，每天都是新的。它只需要**和**与
        **个数**（不像另两处还要最大值），故改用**前缀和**：任意窗口 = 两个前缀之差。每格因此
        只是两次查表加一次减法，不再切片求和。

        **求和是精确的，不是近似。** 成交量是整数股，窗口和远小于 ``2**53``，故前缀差与直接
        切片求和在 float64 下逐位相同——实测 326 列 × 2,703 行（含 65% 缺失）五条读数**最大
        绝对差 0.000e+00**。若调用方喂进非整数成交量，两者会差在 float64 舍入以内，而那远小于
        秩归一能分辨的尺度。

        **代价模型：每「有效格」约 4~6 µs，而不是「每列多少」。** 有效格 = 摆动点已确认的格子
        （`valid`），它才是成本的驱动量。实测（换前缀和之后）：

        ===========================  ===========  ========  ==========
        面板                          有效格        用时      每有效格
        ===========================  ===========  ========  ==========
        326 列 × 2,703 行（切过）      290,944      1.24s     4.3 µs
        1003 列 × 8,726 行（全历史）  4,646,465     26.5s     5.7 µs
        ===========================  ===========  ========  ==========

        换前缀和之前的对照（同一批数据、**切过**的面板，205/405/801 列）：
        9.1 → 5.4、8.4 → 4.3、8.0 → 3.9 µs。

        **注意基准：`mbt screen` 用的是「不切片」的面板。** 它 `assemble_panel(list(loaded.markets),
        ...)`，而 `load_market_data` 返回**完整历史**（`--quality-bars` 只收窄**检查**窗口，不动
        返回的 `prices`）。故在切到 `2015-08-01` 的面板（2,703 行）上量的数**代表不了它**——
        同一批标的不切片是 7,000~8,700 行，而有效格差得更多（列与行都涨，且老股每列贡献的有效格
        远多于次新股）。本文档此前记的「全市场约 59 秒」正是在 2,701 行的**切过**面板上量的，
        别拿它当生产路径的预期。
    """
    if base_bars < 1:
        raise ValueError(f"base_bars 必须为正，收到 {base_bars!r}")
    if atr_n < 1:
        raise ValueError(f"atr_n 必须为正，收到 {atr_n!r}")

    wanted = ("high", "open", "close", "volume")
    absent = [name for name in wanted if name not in panel]
    if absent:
        raise ValueError(
            f"形态读数需要面板含 {absent}，而它没有；面板字段为 {list(panel.field_names)}"
        )

    volumes = panel["volume"]
    if not volumes.columns.equals(anchors.peak_age.columns):
        raise ValueError("panel 与 anchors 的列（标的）必须一致，且顺序相同")
    if not volumes.index.equals(anchors.peak_age.index):
        raise ValueError("panel 与 anchors 的索引（交易日）必须一致")

    high = panel["high"].to_numpy(dtype=float)
    open_ = panel["open"].to_numpy(dtype=float)
    close = panel["close"].to_numpy(dtype=float)
    vol = volumes.to_numpy(dtype=float)
    rows, columns = vol.shape

    positions = np.arange(rows, dtype=float)[:, None]
    peak_age = anchors.peak_age.to_numpy(dtype=float)
    trough_age = anchors.trough_age.to_numpy(dtype=float)
    peak_pos = np.where(np.isfinite(peak_age), positions - peak_age, -1.0)
    trough_pos = np.where(np.isfinite(trough_age), positions - trough_age, -1.0)

    top = _top_of_advance(high, trough_pos)

    # 起涨点之后不足 base_bars 根的格子仍然进循环：那时只有 surge_vs_base 缺失，其余
    # 三个比值照算。故「能不能算基准」不当成整格的门。
    valid = (trough_pos >= 0.0) & (peak_pos >= trough_pos) & (top >= 0)
    right = np.where(valid, np.maximum(peak_pos, top.astype(float)), -1.0)

    atr_frame = atr(panel, atr_n).to_numpy(dtype=float)

    surge = np.full((rows, columns), math.nan)
    calm = np.full((rows, columns), math.nan)
    pull = np.full((rows, columns), math.nan)

    # 「顶部那根」上的取值全部靠花式索引一次取齐，故不占逐格循环的时间。
    top_rows = top[valid]
    top_cols = np.nonzero(valid)[1]
    with np.errstate(invalid="ignore", divide="ignore"):
        top_high = high[top_rows, top_cols]
        body = np.maximum(open_[top_rows, top_cols], close[top_rows, top_cols])
        top_atr = atr_frame[top_rows, top_cols]
        top_shadow = np.full((rows, columns), math.nan)
        # 影取**正**值（`high − max(开, 收)`），故「越小越好」——长上影是扣分项。别写成
        # `max(开,收) − high`：那会得到一个恒非正的数，方向整个反过来。
        top_shadow[valid] = np.where(top_atr > 0.0, (top_high - body) / top_atr, math.nan)
        top_volume = np.full((rows, columns), math.nan)
        top_volume[valid] = vol[top_rows, top_cols]
    top_after = np.where(valid, top.astype(float) - peak_pos, math.nan)

    # 列优先：`nonzero(valid.T)` 得到的行号在一列内升序，列号整体升序。
    order_cols, order_rows = np.nonzero(valid.T)
    bounds = np.searchsorted(order_cols, np.arange(columns + 1))

    # 回调段那一项只需要**和**与**个数**（不像另两个窗口还要最大值），而它的窗口右端就是当前
    # 行、每天都是新的，故缓存不住。改用**前缀和**：任意窗口 `[a, b]` 的和/个数 = 两个前缀之
    # 差，于是每格是两次查表加一次减法，不再切片求和。前缀一次算好，整个矩阵一趟向量化。
    #
    # 只对**有限值**求和（`np.isfinite` 掩码），与本模块「缺失不参与统计、也不当 0」的口径一致。
    # 前缀多留一行（`[0]` 全零）是为了让 `[0, b]` 这类窗口不必单独判边界。
    finite = np.isfinite(vol)
    prefix_total = np.zeros((rows + 1, columns), dtype=float)
    prefix_total[1:] = np.where(finite, vol, 0.0).cumsum(axis=0)
    prefix_count = np.zeros((rows + 1, columns), dtype=np.int64)
    prefix_count[1:] = finite.cumsum(axis=0)

    for column in range(columns):
        first, last = int(bounds[column]), int(bounds[column + 1])
        if first == last:
            continue
        base_key = left_key = right_key = None
        base_mean = left_stats = right_stats = None
        for position in range(first, last):
            row = int(order_rows[position])
            trough_of = int(trough_pos[row, column])
            top_of = int(top[row, column])

            if trough_of != base_key:
                base_key = trough_of
                if trough_of - base_bars >= 0:
                    total, _, count = _finite_stats(vol[trough_of - base_bars : trough_of, column])
                    base_mean = total / count if count else math.nan
                else:
                    base_mean = math.nan

            if (trough_of, top_of) != left_key:
                left_key = (trough_of, top_of)
                left_stats = _finite_stats(vol[trough_of:top_of, column])

            right_of = int(right[row, column])
            if (top_of, right_of) != right_key:
                right_key = (top_of, right_of)
                right_stats = _finite_stats(vol[top_of + 1 : right_of + 1, column])

            # 上涨段除顶的均量与单日最大量：左右两半分开归约再合并，避开每格一次数组拷贝。
            # 空的那一半「和」是 0、「最大值」是 nan（见 `_finite_stats`），故均值可以直接相加，
            # 最大值必须按哪一半为空分开取。
            left_total, left_top, left_count = left_stats
            right_total, right_top, right_count = right_stats
            rest_count = left_count + right_count
            if rest_count:
                rest_mean = (left_total + right_total) / rest_count
                if left_count == 0:
                    rest_max = right_top
                elif right_count == 0:
                    rest_max = left_top
                else:
                    rest_max = left_top if left_top >= right_top else right_top
            else:
                rest_mean = rest_max = math.nan

            # 回调段 `[right_of + 1, row]`：右端是当前行，故这个窗口每天都是新的、缓存不住。
            # 用前缀和查表（见上面 `prefix_total` 的说明），不再切片求和。窗口为空（`right_of
            # == row`）时两个前缀相减都是 0，于是 `pull_count == 0`、判缺失——与旧写法一致。
            pull_total = prefix_total[row + 1, column] - prefix_total[right_of + 1, column]
            pull_count = int(prefix_count[row + 1, column] - prefix_count[right_of + 1, column])
            pull_mean = pull_total / pull_count if pull_count else math.nan

            top_value = top_volume[row, column]
            # 上涨段的单日最大量**含顶部那根**（与旧口径的 `surge_ratio` 同一个统计量，
            # 只是段右端从 `P` 变成了 `max(P, 顶)`）。顶部那根缺失时退回其余部分的最大值。
            advance_max = rest_max
            if math.isfinite(top_value) and (math.isnan(advance_max) or top_value > advance_max):
                advance_max = top_value

            if base_mean > 0.0:
                surge[row, column] = advance_max / base_mean
            if rest_max > 0.0:
                calm[row, column] = top_value / rest_max
            if rest_mean > 0.0:
                pull[row, column] = pull_mean / rest_mean

    def frame(values: np.ndarray) -> pd.DataFrame:
        return pd.DataFrame(values, index=volumes.index, columns=volumes.columns, dtype=float)

    return VolumePattern(
        surge_vs_base=frame(surge),
        top_calm=frame(calm),
        pullback_vs_advance=frame(pull),
        top_shadow_atr=frame(top_shadow),
        top_after_peak=frame(top_after),
    )
