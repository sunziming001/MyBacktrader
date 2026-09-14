"""排序因子：可在**同一时刻横向比较大小**的数值，回答「今天谁更靠前」（``CONTEXT.md``）。

与指标的区别不在算法，而在**用途**：指标是「某一标的的数值序列」，因子要被**同一行**
横向排序。故因子必须满足两条：

1. 数值大小有方向意义，且**越大越靠前**——方向写进文档，不靠调用方猜。需要反向排序的
   因子（如「距高点越近越好」）在定义时就取成越大越好，而不是让消费方翻符号。
2. 缺失就是缺失，不填零。填零会让「无数据」在截面排序里排到一个具体位置——那是造数据。
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from mbt.data.panel import Panel
from mbt.signals._symbol_frame import check_symbol_frame
from mbt.signals.indicators import kdj, rolling_max, white_line, yellow_line


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


def yellow_proximity(prices: pd.DataFrame, windows: tuple[int, ...]) -> pd.DataFrame:
    """**黄线贴近度**：``黄线 ÷ |收盘 − 黄线|``——**越大离黄线越近**。

    它天然满足本层「越大越靠前」的约定，故可直接当排序因子交给
    :class:`~mbt.screen.Screen`，不必翻符号：分母趋于 0 时它趋于**无穷**，而「贴着黄线」
    正是这个因子要排在最前的状态。

    与 :func:`distance_to_high` 的关系要说明：那个因子度量「离高点多远」，这个度量「离
    **均线**多近」。买点本来就要求在回调中贴近黄线（``CONTEXT.md`` 的**黄线**），故排序用
    后者比「跌得最深」更贴合这套规则——跌得深只说明跌得多，不代表回到了支撑上。

    .. warning::

        **它度量的是「相对贴近度」，不是绝对距离。** 分子分母同量纲，故结果对价格整体缩放
        不变——等价于 ``1 ÷ (|收盘 − 黄线| ÷ 黄线)``，即相对距离的倒数。故两只股票即使与各自
        黄线的**绝对**距离相同，黄线更高的那只因子更大。这是公式本身的性质，不是实现细节；
        要按绝对距离排序应当另写一个因子（本项目不提供，因为价格量纲上的绝对距离在横截面上
        没有可比性）。

        **它取绝对值，故不区分站上还是跌破黄线。** 这也是刻意的：因子是**横截面**上的排序
        依据，而「在黄线哪一侧」是**过滤信号**该管的事（``white_above_yellow``）。把方向塞
        进因子会让同一个条件在两层各说一遍。

    两处退化要写明：

    - 收盘**恰好**等于黄线时分母为 0，结果为**正无穷**——数学上正确（它是最近的），且与
      :func:`~mbt.signals.indicators.volume_ratio` 在前 n 根均为零成交时返回无穷同一先例。
      消费方按大小排序即可，无需特判。
    - 黄线为 0（或负）时分子退化，结果不再有「贴近度」的含义。价格不会为 0，故这在真实
      行情上不出现；序列开头窗口不足时黄线是**缺失**，结果随之缺失，而 ``Screen`` 会把缺失
      的标的排除在取前 N 之外（与其余因子一致）。
    """
    line = yellow_line(prices, windows)
    return line / (check_symbol_frame(prices) - line).abs()


def reward_risk_ratio(
    prices: pd.DataFrame,
    *,
    white_n: int,
    windows: tuple[int, ...],
) -> pd.DataFrame:
    """**交易盈亏比**：``(白线 − 收盘) ÷ (收盘 − 黄线)``——**越大越划算**。

    分子是**赚头**：涨回白线还剩多少。分母是**亏头**：跌到黄线还剩多少。它把「这笔交易值不
    值得做」写成一个能在同一行内横向比较的数，故既能当排序因子，也能当过滤信号的判据
    （``> 门槛``）。

    **两个参照取自两条均线，而不是价格结构。** ``CONTEXT.md`` 的**交易盈亏比**一条要求使用
    时把参照的取法一并声明——这一版的取法是：**赚头看白线**（快线，回到它就算走完这一段）、
    **亏头看黄线**（慢线，跌破它趋势就变了）。这与策略的两条卖出规则对得上：B1 的离场正是
    「最高价破白线」与「连续 2 根收盘破黄线」，故这个比制度量的恰是「先摸到白线、还是先跌破
    黄线」。

    前一版用的是「前高 ÷ 前低」（摆动点峰值与锚定前低）。改成均线是**换参照**而非修错：
    均线的优点是两处口径都只依赖收盘价序列、可向量化，且不需要摆动点；代价是它不再反映
    这一段的**结构**（峰在哪里、前低在哪里）。

    .. note::

        **它只依赖收盘价**，故不需要行情面板、也不需要摆动点。前两版分别要 ``low`` 与
        ``anchors``，各自带来一处「必须与别处同源」的约束；这一版把那些约束一并去掉了。

    .. warning::

        **它与 :attr:`mbt.metrics.Metrics.payoff_ratio`（盈亏比）不是同一个量，不可互相印证。**

        ====================  ==========================================
        ``payoff_ratio``      **已实现**交易的「平均盈利 ÷ 平均亏损」
        ``reward_risk_ratio`` **建仓前**预估的「赚头 ÷ 亏头」
        ====================  ==========================================

        前者是复盘算出来的结果，后者是下单前的判断。后者高不蕴含前者高。

    .. warning::

        **分母 ≤ 0 时取缺失**，不给一个具体的数：收盘已跌破黄线时没有「亏头」可言，比值失去
        含义。刻意不给负数（会在横截面排序里排到最末，看着像「最差的一档」），也不给无穷。
        分子 ≤ 0（收盘已站上白线）时**照常给负值**——那是一个真实可比的比值，且会被
        ``> 门槛`` 正常挡掉，不必额外处理。

    .. warning::

        **``white_n`` 与 ``windows`` 必须与策略用的白线、黄线取同一组窗口**，否则度量的是两条
        策略并不看的线。两边默认值一致（``10`` 与 ``(14, 28, 57, 114)``），改动时要一起改。

    参数:
        prices: 收盘价的标的宽表。
        white_n: 白线的双重 EMA 窗口。
        windows: 黄线各条均线的窗口。

    抛:
        ValueError: ``windows`` 为空（没有均线就谈不上黄线）。白线窗口由 ``ema`` 自行处理，
            不额外校验。

    缺失：收盘、白线或黄线任一为缺失处即为缺失（窗口不足、停牌、尚未上市）。
    """
    close = check_symbol_frame(prices)
    line = yellow_line(close, windows)
    white = white_line(close, white_n)

    prices_array = close.to_numpy(dtype=float)
    risk = prices_array - line.to_numpy(dtype=float)
    reward = white.to_numpy(dtype=float) - prices_array
    with np.errstate(invalid="ignore", divide="ignore"):
        ratio = np.where(risk > 0.0, reward / risk, np.nan)
    return pd.DataFrame(ratio, index=close.index, columns=close.columns, dtype=float)


def j_oversold(panel: Panel, n: int, m1: int, m2: int) -> pd.DataFrame:
    """**KDJ 的 J 值超卖程度**：``−J``——**越大越超卖**。

    J 本身**越小越超卖**，而本层约定「越大越靠前」，故取负号把它翻正。它只是 J 的**单调
    变换**：排序只关心次序，故不必再做归一化（把 ``[0, 100]`` 映射成别的区间不会改变任何一次
    排序的结果，却会多出一处需要解释的常数）。

    这正是它的用处与限度：它回答「今天谁的 J 更低」，**不回答**「J 低到什么程度算超卖」——
    后者是过滤器的门槛（``j_below`` 的 ``threshold``），两者分工不同，不要互相替代。

    J 会跌破 0（定义如此，见 :func:`~mbt.signals.indicators.kdj`），故本因子的取值没有下界；
    ``−J`` 也就没有上界。缺失（窗口不足、分母为 0、缺口）处取缺失。

    参数:
        panel: 行情面板，须含 ``high`` / ``low`` / ``close``。
        n / m1 / m2: 传给 :func:`~mbt.signals.indicators.kdj` 的三个窗口。
    """
    return -kdj(panel, n, m1, m2).j
