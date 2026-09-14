"""过滤信号：对**单一标的**的是/否判断，回答「它此刻是否合格」（``CONTEXT.md``）。

两条约定在此落地：

1. 输出是**纯 ``bool``** 的标的宽表，缺失一律取 ``False``，语义是「不合格」。这不是额外代码，
   而是**数值比较的天然结果**——``nan > x`` 与 ``x > nan`` 都返回 ``False``。刻意不把缺失
   先填成某个数：填了就替数据「表了态」，而本项目宁可漏判也不凭空造出合格（ADR-0005）。
2. 判断只在**已知**上成立。窗口不足时不给乐观答案——``min_periods=n`` 让前 n-1 根为缺失，
   比较后即为 ``False``。

「状态」与「事件」两类不要混淆：``above_ma`` 回答「此刻在均线上方吗」，
``ma_cross_up`` 回答「今天刚上穿吗」，后者不是前者的子集。
"""

from __future__ import annotations

import pandas as pd

from mbt.data.panel import Panel
from mbt.signals._symbol_frame import check_symbol_frame
from mbt.signals.indicators import (
    kdj,
    rolling_max,
    sma,
    volume_ratio,
    white_line,
    yellow_line,
)
from mbt.signals.swings import swings


def new_high(prices: pd.DataFrame, n: int) -> pd.DataFrame:
    """创 n 日新高：当根收盘**严格高于**前 n 个交易日的最高收盘价。

    用 ``>`` 而非 ``>=``：恰好等于前高不是「创」新高。
    """
    return check_symbol_frame(prices) > rolling_max(prices, n).shift(1)


def above_ma(prices: pd.DataFrame, n: int) -> pd.DataFrame:
    """站上均线：当根收盘高于 n 日均线（含当根）。这是一个**状态**。"""
    return check_symbol_frame(prices) > sma(prices, n)


def ma_cross_up(prices: pd.DataFrame, n: int) -> pd.DataFrame:
    """均线上穿：当根收盘上穿 n 日均线，即昨日在下（或持平）、今日在上。这是一个**事件**。

    昨日的均线必须**已知**，否则「上穿」会退化成「第一次算得出均线」——那不是穿越，
    只是窗口刚好填满，把它当信号会在每只次新股的同一位置凭空点火。
    """
    prices = check_symbol_frame(prices)
    ma = sma(prices, n)
    return (prices > ma) & ma.shift(1).notna() & (prices.shift(1) <= ma.shift(1))


def rising_streak(prices: pd.DataFrame, n: int) -> pd.DataFrame:
    """连续上涨 n 日：最近 n 个交易日**每一次**收盘都高于前一日（故需要 n+1 根 K 线）。"""
    prices = check_symbol_frame(prices)
    return (prices.diff() > 0).rolling(n, min_periods=n).sum() == n


def volume_surge(volumes: pd.DataFrame, k: float, n: int) -> pd.DataFrame:
    """放量 k 倍：当根成交量**严格高于**前 n 日均量的 k 倍。

    它由 :func:`~mbt.signals.indicators.volume_ratio` 实现，故「基准取前 n 根」这条口径
    只有一处定义，不会与成交量比漂移。要连续的比值（而非是/否）请直接用后者。
    """
    return volume_ratio(volumes, n) > k


def white_above_yellow(prices: pd.DataFrame, n: int, windows: tuple[int, ...]) -> pd.DataFrame:
    """白线在黄线上方（**状态**）：``EMA(EMA(C, n), n) > mean(MA(C, w))``。

    两条线各在哪定义是刻意的：口径只写在
    :func:`~mbt.signals.indicators.white_line` 与 :func:`~mbt.signals.indicators.yellow_line`
    里，本函数只做比较，故「白线怎么算」不会在过滤器里出现第二份。

    参数:
        prices: 收盘价的标的宽表。
        n: 白线的双重 EMA 窗口（常见 10）。
        windows: 黄线各条均线的窗口（常见 ``(14, 28, 57, 114)``）。

    窗口不足处两条线皆为缺失，比较即 ``False``。黄线含长窗口均线，故可用起点由**它**决定
    ——这正是「不给乐观答案」的落点：一只只有 60 根 K 线的次新股不会因为白线算得出就被放行。
    """
    return white_line(prices, n) > yellow_line(prices, windows)


def j_below(panel: Panel, threshold: float, n: int, m1: int, m2: int) -> pd.DataFrame:
    """KDJ 的 J 值**严格低于**阈值（**状态**）。

    阈值与 ``n`` / ``m1`` / ``m2`` 一律由调用方传入，函数里不写死：这条规则的阈值正是需要
    按样本与区间标定的量，藏进函数名或默认值里就没法标定了。用 ``<`` 而非 ``<=``，与
    :func:`new_high`、:func:`volume_surge` 的严格比较一致——恰好落在阈值上不算。

    J 会跌破 0（定义如此，见 :class:`~mbt.signals.indicators.KDJ`），故 ``threshold=0``
    是合法且常用的一档。窗口不满、分母为 0、数据缺口的当根一律 ``False``。
    """
    return kdj(panel, n, m1, m2).j < threshold


def below_yellow_streak(prices: pd.DataFrame, n: int, windows: tuple[int, ...], days: int):
    """**连续 ``days`` 根**收盘价低于黄线（**状态**）。

    与 :func:`white_above_yellow` 是同一对线的两侧，差别在「连续」：单根跌破只是价格穿透，
    连跌 ``days`` 根才算趋势跌破。用 ``<`` 而非 ``<=``，与 :func:`new_high`、
    :func:`volume_surge` 的严格比较一致——恰好等于黄线不算跌破。

    黄线的口径只写在 :func:`~mbt.signals.indicators.yellow_line` 里（``n`` 与 ``windows``
    照旧由调用方给）；本函数只做「低于」与「连续」两件判断。

    窗口不足处黄线缺失，比较即 ``False``，故 ``days`` 根连续成立**需要** ``days`` 根可用
    的收盘价与黄线，不给乐观答案（ADR-0005）。
    """
    below = check_symbol_frame(prices) < yellow_line(prices, windows)
    return below.rolling(days, min_periods=days).sum() == days


def above_yellow(prices: pd.DataFrame, windows: tuple[int, ...]):
    """收盘价**严格高于黄线**（**状态**）——「价格还在慢线上方」。

    与 :func:`white_above_yellow` 不是一回事，两者要分开看：

    ========================  ==========================================
    ``white_above_yellow``    **白线**在黄线上方（两条线之间的关系）
    ``above_yellow``          **收盘价**在黄线上方（价格与线之间的关系）
    ========================  ==========================================

    ``white_above_yellow`` 成立并不蕴含 ``above_yellow`` 成立：价格可以从上方双双跌破，
    那时白线仍在黄线上（慢线还没跟上），而收盘已经在黄线**下方**。这两条同时要求，等于
    说「趋势向上**且**价格还没跌穿慢线」。

    它也是 :func:`below_yellow_streak` 的单根对照：那个要求连续 ``days`` 根在下方，
    这个只要求当根在上方。买入侧用它、卖出侧用那个，两侧的门槛因此不对称——要允许
    「刚跌破一点点就入场」就把 ``days`` 调大，而 ``above_yellow`` 管的是入场那一刻
    不许在黄线下方。

    用 ``>`` 而非 ``>=``：恰好收在黄线上不算在上方（与 :func:`new_high`、
    :func:`volume_surge` 的严格比较一致）。黄线窗口不足处为缺失，比较即 ``False``。
    """
    return check_symbol_frame(prices) > yellow_line(prices, windows)


def high_above_white(panel: Panel, n: int):
    """当日**最高价**高于白线（**状态**）——「盘中摸到过白线」。

    与 :func:`above_white` 的差别只在用哪个价：

    ====================  ==========================================
    ``above_white``       **收盘价**高于白线（站上了）
    ``high_above_white``  **最高价**高于白线（摸到过）
    ====================  ==========================================

    区别有实际后果：最高价摸到白线而收盘又落回下方，是「上冲被打回」；``above_white``
    看不见这一类，``high_above_white`` 看得见。B1 的「最高价破白线即卖出」用的就是后者。

    参数:
        panel: 行情面板，须含 ``close`` 与 ``high``。
        n: 白线的双重 EMA 窗口。

    用 ``>`` 而非 ``>=``：最高价恰好等于白线不算破（与 :func:`new_high` 同一口径）。
    ``high`` 缺失（停牌）处取假；白线窗口不足处为缺失，比较亦假。
    """
    close = check_symbol_frame(panel["close"])
    high = check_symbol_frame(panel["high"])
    return high > white_line(close, n)


def above_white(prices: pd.DataFrame, n: int, margin: float):
    """收盘价**高于白线 ``margin`` 比例以上**（**状态**）。

    即 ``C > 白线 × (1 + margin)``。``margin=0`` 就是「站上白线」本身；卖出规则里的
    「收盘在白线上 8%」即 ``margin=0.08``。用 ``>`` 而非 ``>=``，与 :func:`new_high`、
    :func:`volume_surge` 一致——恰好触到门槛不算。

    白线的口径只写在 :func:`~mbt.signals.indicators.white_line` 里；本函数只做比较。

    参数:
        margin: 相对白线的比例门槛，须 ``> -1``（即门槛价为正）。给了上界之外的取值时
            比较会退化成恒真，那不是「放松条件」而是「条件消失」，故报错而不是静默接受。

    窗口不足处白线缺失，比较即 ``False``，不给乐观答案（ADR-0005）。
    """
    if margin <= -1.0:
        raise ValueError(f"margin 必须 > -1（否则门槛价非正、比较恒真），收到 {margin!r}")
    return check_symbol_frame(prices) > white_line(prices, n) * (1.0 + margin)


def below_white(prices: pd.DataFrame, n: int):
    """收盘价**低于白线**（**状态**）——卖出规则里的「跌破白线」。

    刻意**不设** ``margin``（与 :func:`above_white` 不对称）：这条规则要的是「一跌破就清仓」
    这个明确动作，门槛一多，同一个价位上「算不算跌破」就会随参数漂移。要更松或更紧的
    出场，应当新增一条命名清楚的规则，而不是给这一条加旋钮。

    用 ``<`` 而非 ``<=``：恰好收在白线上不算跌破。
    """
    return check_symbol_frame(prices) < white_line(prices, n)


def pullback_after_advance(
    prices: pd.DataFrame,
    *,
    retracement: float,
    min_advance: float,
    min_drop: float,
    max_drop: float,
    min_peak_age: int,
    max_peak_age: int,
):
    """「一波上涨之后的下跌阶段」（**状态**）——提示 3 的买点所在的那个位置。

    它由**四个**各自可手算的条件组成，全部成立才算：

    ==================  ==========================================  ==============
    条件                判据                                         来自 9 样本实测
    ==================  ==========================================  ==============
    涨过一段            已完成上涨的幅度 ≥ ``min_advance``            33%~236%（中位 65%）
    正在回调            收盘相对峰值回落，落在                          −8.5%~−30.5%
                        ``[min_drop, max_drop]`` 之内                  （中位 −12.9%）
    回调不算太久        自峰值起 ``[min_peak_age, max_peak_age]`` 根     6~48 根（中位 11）
    拐点已确认          由 :func:`~mbt.signals.swings.swings` 给出       —
    ==================  ==========================================  ==============

    「上涨段」的起点与终点**不是固定窗口**，而是
    :func:`~mbt.signals.swings.swings` 确认的拐点——这正是提示里「一波上涨」与「调整期」
    的字面意思。``retracement`` 是确认拐点所需的回撤幅度，直接决定段边界，故必须显式给。

    参数:
        prices: 收盘价的标的宽表。
        retracement: 传给 :func:`~mbt.signals.swings.swings` 的拐点确认阈值。
        min_advance: 上涨幅度下界（如 ``0.30``）。
        min_drop / max_drop: 回调深度的**下界与上界，都取正数**（如 ``0.08`` / ``0.35``）。
            用正数是为了让调用处不必写两个负号——比较时内部取 ``−drop``。
        min_peak_age / max_peak_age: 自峰值起的根数区间（含两端）。

    .. note::

        **本函数只用收盘价**，与前三个条件一致：提示里的「下跌阶段」是价格结构，不是量。
        量能那一条是独立的过滤器，两条各自成立、组合起来才是完整买点——那样的好处是任一条
        改了都不必动另一条。

    缺失处一律 ``False``：拐点未确认、或窗口不足，都不给乐观答案（ADR-0005）。
    """
    anchors = swings(prices, retracement)
    close = check_symbol_frame(prices)

    advance = anchors.peak_price / anchors.trough_price - 1.0
    depth = -(close / anchors.peak_price - 1.0)  # 取正值，便于与 min_drop / max_drop 比
    age = anchors.peak_age

    return (
        (advance >= min_advance)
        & (depth >= min_drop)
        & (depth <= max_drop)
        & (age >= min_peak_age)
        & (age <= max_peak_age)
    )
