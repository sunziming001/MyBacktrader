"""排序因子：可在**同一时刻横向比较大小**的数值，回答「今天谁更靠前」（``CONTEXT.md``）。

与指标的区别不在算法，而在**用途**：指标是「某一标的的数值序列」，因子要被**同一行**
横向排序。故因子必须满足两条：

1. 数值大小有方向意义，且**越大越靠前**——方向写进文档，不靠调用方猜。需要反向排序的
   因子（如「距高点越近越好」）在定义时就取成越大越好，而不是让消费方翻符号。
2. 缺失就是缺失，不填零。填零会让「无数据」在截面排序里排到一个具体位置——那是造数据。
"""

from __future__ import annotations

from collections import deque

import numpy as np
import pandas as pd

from mbt.data.panel import Panel
from mbt.signals._symbol_frame import check_symbol_frame
from mbt.signals.indicators import rolling_max, yellow_line
from mbt.signals.swings import Swings


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
    panel: Panel,
    anchors: Swings,
    *,
    windows: tuple[int, ...],
    stop_buffer: float,
) -> pd.DataFrame:
    """**交易盈亏比**：``(前高 − 收盘) ÷ (收盘 − 止损位)``——**越大越划算**。

    止损位取两条止损里**较高的那一条**，即策略上先被触发的那条::

        止损位 = max(黄线, 前低)
        前低   = 「峰值 → 当根」（含两端）的最低价 × (1 − stop_buffer)

    它把「这笔交易值不值得做」写成一个能在同一行内横向比较的数：分子是**赚头**（回到前高还
    剩多少），分母是**亏头**（跌到止损还有多少）。故它天然满足本层「越大越靠前」的契约，既能
    当排序因子，也能当过滤信号的判据（``> 门槛``）。

    **两个参照的取法都显式写在这里。** ``CONTEXT.md`` 的**前低**一条明令如此——它的取法直接
    决定参照价位的高低，故不许只写「前低」两个字就算了：

    - **前高**取 :func:`~mbt.signals.swings.swings` 的 ``peak_price``，即最近一段**已完成
      上涨**的峰值。刻意不用固定窗口的「N 日最高」：那样「回到前高」会退化成「回到某个窗口的
      最高」，而窗口长度恰是需要标定的那个量；摆动点给出的峰值是**价格自己切出来的**。
    - **前低**取「峰值 → **当根**」这一段的最低价。窗口**含当根**，与策略
      ``B1._anchored_prior_low`` 是同一个式子；代价见下面第二条 warning。
    - **黄线**与其余规则同一条（:func:`~mbt.signals.indicators.yellow_line`）。取两者中**较大**
      者作止损位，是因为策略的止损正是「连续跌破黄线 **或** 跌破前低 × (1 − stop_buffer)」，
      两条里先到的那条才决定这笔交易实际亏多少。

    .. warning::

        **它与 :attr:`mbt.metrics.Metrics.payoff_ratio`（盈亏比）不是同一个量，不可互相印证。**

        ====================  ==========================================
        ``payoff_ratio``      **已实现**交易的「平均盈利 ÷ 平均亏损」
        ``reward_risk_ratio`` **建仓前**预估的「赚头 ÷ 亏头」
        ====================  ==========================================

        前者是复盘算出来的结果，后者是下单前的判断。后者高不蕴含前者高。

    .. warning::

        **窗口含当根，故这里的止损位是「到当根为止的最低点」。** 实测（141 只 × 8,558 个交易日）
        它与「不含当根」的口径**落点相近但不等价**，差别集中在一类格子上：

        - 当根刚创出调整期新低时，前低随之下移到当根最低价，止损位因此**更低**、亏头更大，比值
          也就**更小**（同一格实测中位 7.1 对 12.4）——这一口径在门槛上更保守。
        - 但它也是**唯一**会放行「当根收在最低价附近、已经跌穿昨日的前低」那一类的口径：分母
          恒为正（当根最低价 ≤ 当根收盘价），故从不因为「已经跌破」而作废。放行的 29,667 格
          里有 2,300 格属于这一类，其中 **1,551 格**换成不含当根就直接给不出正亏头。

        要挡掉后一类就得把窗口改成**不含当根**，代价是同时会拒掉一部分仍在昨日前低之上的回调。
        两种口径各有代价，故由调用方选，函数里不写死。

    **两处时点漂移要写明**（它住在信号层，而止损锚定在策略层）：

    1. 本函数在**评估日**（选股／调仓的那一根）算，而策略的前低是在**建仓那一根**才锚定的，
       两者相差一根；建仓的成交价又是那一根的**开盘价**，而这里用的是当根**收盘价**。
    2. 故它是一个**决策时的预估**，不是这笔交易最终的风险敞口。选股规则按定义不管持仓
       （ADR-0001），它也只能是预估。

    退化与缺失一律取**缺失**，不给一个具体的数：

    - **分母 ≤ 0**（已经跌破黄线或前低）——没有「亏头」可言，比值失去含义。刻意不给负数
      （那会在横截面排序里排到最末，看着像「最差的一档」），也不给无穷（看着像「最好的一档」）。
    - 峰值未确认（``anchors`` 处缺失）、黄线窗口不足、窗口内无有效最低价、当根无价。

    缺口（停牌、尚未上市）**不参与取最小值**——跳过而不是当成 0，也不因为窗内有缺口就把整窗
    判为缺失（ADR-0005 的「缺口跳过」）。

    参数:
        panel: 行情面板，须含 ``close`` 与 ``low``。
        anchors: :func:`~mbt.signals.swings.swings` 的输出，须与 ``panel`` 出自**同一段行情**。
            显式收进来而不在函数内重算，是为了让「价格结构与盈亏比吃同一份段边界」成为一条
            被强制的前提——各自重算一次就可能与前一个过滤器切出不同的段，而那种错不报错。
        windows: 黄线各条均线的窗口。
        stop_buffer: 前低的缓冲比例（止损位 = 前低 × (1 − 它)），须落在 ``[0, 1)``。它应当与
            策略 ``B1`` 的同名参数取**同一个值**——两边不一致时，这个比制度量的是一条策略
            并不会执行的止损。

    抛:
        ValueError: ``stop_buffer`` 不在 ``[0, 1)`` 内，或 ``panel`` 与 ``anchors`` 的标的或
            交易日对不上（对不上会把段边界对到别的日子上，而那种错不报错）。

    .. note::

        与 :func:`~mbt.signals.swings.swings` 同量级的代价：取「峰值 → 当根」的最低值要在每个
        标的上顺序扫一遍（单调队列，摊还 O(根数)）。窗口的左端就是峰值位置，而它由 ``swings``
        保证**不后退**，这正是能用队列而不必重扫整段的原因。
    """
    if not 0.0 <= stop_buffer < 1.0:
        raise ValueError(f"stop_buffer 必须落在 [0, 1) 内，收到 {stop_buffer!r}")

    close = check_symbol_frame(panel["close"])
    low = check_symbol_frame(panel["low"])
    if not low.columns.equals(anchors.peak_age.columns):
        raise ValueError("panel 与 anchors 的列（标的）必须一致，且顺序相同")
    if not low.index.equals(anchors.peak_age.index):
        raise ValueError("panel 与 anchors 的索引（交易日）必须一致")

    lows = low.to_numpy(dtype=float)
    ages = anchors.peak_age.to_numpy(dtype=float)
    prior_low = np.full(lows.shape, np.nan)
    for column in range(lows.shape[1]):
        prior_low[:, column] = _lowest_since_peak(lows[:, column], ages[:, column])

    line = yellow_line(close, windows).to_numpy(dtype=float)
    stop = np.maximum(line, prior_low * (1.0 - stop_buffer))
    prices = close.to_numpy(dtype=float)
    risk = prices - stop
    reward = anchors.peak_price.to_numpy(dtype=float) - prices
    with np.errstate(invalid="ignore", divide="ignore"):
        ratio = np.where(risk > 0.0, reward / risk, np.nan)
    return pd.DataFrame(ratio, index=close.index, columns=close.columns, dtype=float)


def _lowest_since_peak(lows: np.ndarray, peak_age: np.ndarray) -> np.ndarray:
    """逐根取「峰值 → 当根」这一段的最低价（含两端），缺失处为缺失。

    单调队列，摊还 O(根数)：窗口的**右端每根前进一格**，**左端是峰值位置**——它只会在新的
    峰值被确认时跳向更晚的一根，由 :func:`~mbt.signals.swings.swings` 保证不后退。两端都单调
    向右，故队首即为窗口最小值，不必重扫整段。

    缺失（停牌、尚未上市）**不参与取最小值**：跳过而不是当成 0，也不因为窗内有缺口就把整窗
    判为缺失（ADR-0005 的「缺口跳过」）。整窗无有效值时给缺失。

    **峰值尚未确认的那些根，低价照样要留在队列里。** 峰值的位置往往**早于**它被确认的那一根
    （确认要等回撤到阈值），故首个可用窗口的左端落在过去；若在确认之前就把队列清空，那个窗口
    就只剩当根一根——前低会退化成「当根最低价」，盈亏比随之虚高。
    """
    bars = lows.size
    out = np.full(bars, np.nan)
    window: deque[int] = deque()

    for now in range(bars):
        value = lows[now]
        if not np.isnan(value):
            while window and lows[window[-1]] >= value:
                window.pop()
            window.append(now)

        age = peak_age[now]
        if not np.isfinite(age):
            continue  # 峰值未确认：这一格没有窗口可言（但队列留着，见文档末段）

        left = now - int(age)
        while window and window[0] < left:
            window.popleft()
        if window:
            out[now] = lows[window[0]]

    return out
