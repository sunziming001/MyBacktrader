"""指标：由**单一标的**的价量序列算出、本身不含任何判断的数值序列（``CONTEXT.md``）。

它们是过滤信号与排序因子的**构件**——「放量」与「站上均线」都组合这里的滚动窗口，而不是
各写一份。窗口一律取「当根及其之前 n-1 根」且 ``min_periods=n``：既不含未来信息，也不把
不足窗口的短缺当成数值——缺失就让它缺失（ADR-0005 的缺口纪律）。

本模块多数函数取**单字段**的标的宽表；``atr`` 与 ``kdj`` 需要**跨字段**，故它们取
:class:`~mbt.data.panel.Panel`。跨字段的信号**返回标的宽表**（ADR-0009）：消费方向与其余的
指标完全一样（列 = 时序、行 = 截面），只是输入多了两张表。
"""

from __future__ import annotations

from typing import NamedTuple

import numpy as np
import pandas as pd

from mbt.data.panel import Panel
from mbt.signals._symbol_frame import check_symbol_frame


def sma(prices: pd.DataFrame, n: int) -> pd.DataFrame:
    """n 日简单移动平均，含当根；窗口不足处为缺失值。"""
    return check_symbol_frame(prices).rolling(n, min_periods=n).mean()


def _smooth_frame(prices: pd.DataFrame, alpha: float) -> pd.DataFrame:
    """对整张标的宽表做**分段递推平滑**：段首 ``Y = X``，其后 ``Y = α·X + (1 − α)·Y_prev``。

    缺失把序列切成若干段，每段以该段**第一个可用值**播种；缺失处一律是缺失。

    **为什么按行推进，不按列。** 递推只能沿时间走，但**列之间互不相干**，故按行推进时每一行
    可以一次算完全部标的（numpy 向量），Python 层的迭代因此从「列数 × 行数」降到「行数」。
    原先按列做（``DataFrame.apply``）时，5,451 个标的各要一次 pandas 重活（``notna`` /
    ``cumsum`` / ``groupby`` / ``ewm``），实测单层 EMA 在白线（双层）上就是分钟量级。
    实测白线（467 列 × 7,206 行）：**3.64s → 0.18s（20×）**。

    **数值上不是逐位相同，但等价到末位。** ``α·x + (1−α)·y`` 与 pandas ``ewm`` 的内部写法
    在最后一两个 ULP 上不同：实测最大相对差 **4.7e-16**（白线）、**3.6e-16**（单层 EMA），
    而 B1 的 ``trend`` 门（白线 > 黄线）在 285,768 格上**零翻转**、``low_j`` 门亦零翻转
    （见 ``.scratch/smooth_dump_or_compare.py``）。这是**接受**的：递推给的是同一序列，
    差在浮点舍入，而判据是「比大小」不是「逐位比对」。

    .. note::

        不能整列调一次 ``ewm(adjust=False)``：它遇到缺失会**沿用前值**——``[1, 2, NaN, 4, 5]``
        在 ``span=2`` 下给出 ``[1, 1.667, 1.667, 3.667, 4.556]``，第 3 位不是缺失而是上一根的
        值。那等于把停牌日伪造成一个**有**指标值的交易日（ADR-0005）。故段必须分开。
    """
    frame = check_symbol_frame(prices)
    values = frame.to_numpy(dtype=float)
    rows, width = values.shape
    out = np.full((rows, width), np.nan)
    # `prev` 为 NaN 表示「当前处在段首」（段首或刚跨过一个缺口）。
    prev = np.full(width, np.nan)
    for t in range(rows):
        row = values[t]
        seeded = np.where(np.isnan(prev), row, alpha * row + (1.0 - alpha) * prev)
        prev = np.where(np.isnan(row), np.nan, seeded)
        out[t] = prev
    return pd.DataFrame(out, index=frame.index, columns=frame.columns, dtype=float)


def ema(prices: pd.DataFrame, n: int) -> pd.DataFrame:
    """n 日指数移动平均，**通达信口径**：``EMA_t = (2·C_t + (n−1)·EMA_{t−1}) ÷ (n+1)``。

    与 :func:`sma` 的口径差别是**刻意的，不要对齐**：

    - ``sma`` 是尾随窗口均值，窗口不满处为**缺失**（``min_periods=n``）；
    - ``ema`` 是递推平滑，**没有「窗口填满」这一刻**。通达信与行情软件都以**首根**播种
      （``EMA₁ = C₁``）并一路递推；要对齐图上的线就必须照此口径。若也按 ``min_periods=n``
      处理，前 n−1 根会缺失，且首个非缺失值也与图上不同。

    代价要写明：它在前若干根的取值受**序列起点**影响——那不是数据的性质，是本函数的口径。
    故凡与长窗口均线（如 114 日均线）同时出现的信号，其可用起点由**那个长窗口**决定，
    本函数不额外设门槛。

    缺失处一律是缺失：没有收盘价就没有 EMA。缺口把序列切成若干段，每段以该段**第一个可用值**
    播种——与 :func:`atr` 是同一条缺口纪律，而**不是**「沿用前值续算」：把跨越停牌的递推连起来
    等于假装停牌期间也有收盘价（ADR-0005）。行情软件上没有缺口行，故只在**无缺口**的序列上
    两种口径一致——也就是说「字面复刻图上的线」这件事，在有空缺的标的身上并不成立。
    """
    return _smooth_frame(prices, 2.0 / (n + 1))


def rolling_max(prices: pd.DataFrame, n: int) -> pd.DataFrame:
    """n 日最高价，含当根；窗口不足处为缺失值。"""
    return check_symbol_frame(prices).rolling(n, min_periods=n).max()


def rolling_min(prices: pd.DataFrame, n: int) -> pd.DataFrame:
    """n 日最低价，含当根；窗口不足处为缺失值。

    与 :func:`rolling_max` 对称。喂 ``low`` 字段即得「近 n 根的**前低**」——止损位要用的量
    （见 ``CONTEXT.md`` 的**前低**）；喂收盘价得的是收盘价的低点，两者不是一回事。
    """
    return check_symbol_frame(prices).rolling(n, min_periods=n).min()


def white_line(prices: pd.DataFrame, n: int) -> pd.DataFrame:
    """**白线**：``EMA(EMA(C, n), n)``——双重指数平滑，行情软件图上的一条快线。

    名字取自图上线的颜色，不是算法性质；同名术语见 ``CONTEXT.md``。``n`` 由调用方给定
    （常见取 10）而不设默认值：默认值会让「用了哪个口径」从调用处消失。

    外层 EMA 吃到内层在缺口处留下的缺失，故缺口会把两层的递推都切开——与 :func:`ema`
    是同一条缺口纪律。
    """
    return ema(ema(prices, n), n)


def yellow_line(prices: pd.DataFrame, windows: tuple[int, ...]) -> pd.DataFrame:
    """**黄线**：若干条 ``MA(C, w)`` 的**等权均值**，行情软件图上的一条慢线。

    名字取自图上线的颜色，不是算法性质；同名术语见 ``CONTEXT.md``。``windows`` 由调用方
    给定（常见取 ``(14, 28, 57, 114)``）而不设默认值，理由同 :func:`white_line`。

    均值用**简单平均**而非加权：图上那四条线是等权的。任一条 MA 在当根缺失（窗口不满或
    数据有缺口），结果即缺失——NaN 会自然沿加法传播，不必额外判空。
    """
    if not windows:
        raise ValueError("windows 不能为空：至少需要一条均线才谈得上均值")
    return sum(sma(prices, w) for w in windows) / len(windows)


class KDJ(NamedTuple):
    """KDJ 的三条线，各是一条标的宽表。

    三条线**各自**都是合法的指标（列 = 时序、行 = 截面），故这个结构不破坏「指标返回
    标的宽表」的契约（ADR-0009）——它只是把一次计算里的三个结果一起交回，省得调用方
    为了拿 ``j`` 而把 ``k``、``d`` 算两遍。
    """

    k: pd.DataFrame
    d: pd.DataFrame
    j: pd.DataFrame


def kdj(panel: Panel, n: int, m1: int, m2: int) -> KDJ:
    """KDJ 指标，**通达信口径**：

    .. code-block:: text

        RSV_t = (C_t − LLV(L, n)) ÷ (HHV(H, n) − LLV(L, n)) × 100
        K_t   = SMA(RSV, m1, 1)      # 即 α = 1/m1 的递推平滑
        D_t   = SMA(K,   m2, 1)
        J_t   = 3·K_t − 2·D_t

    其中 ``SMA(X, N, M) = (M·X + (N−M)·Y_prev) ÷ N`` 是通达信的递推均值，与本模块的
    :func:`ema` 是同一族（``α = M/N``）——**不是**移动平均。用简单均线算出的 K 与图上不同，
    这条差别已被测试钉住。

    参数:
        panel: 须含 ``high`` / ``low`` / ``close`` 三个字段，且三者同日对齐
            （对齐由 :class:`~mbt.data.panel.Panel` 保证）。
        n / m1 / m2: 三个窗口，由调用方给出；KDJ 的常见取值为 ``9 / 3 / 3``，但这里不设
            默认值——默认值会让「用了哪个口径」从调用处消失。

    返回:
        :class:`KDJ`（``k`` / ``d`` / ``j`` 各为一条标的宽表）。``j`` 不属于 ``[0, 100]``，
        会跌破 0、也会超过 100，这是定义如此而非瑕疵。

    三类缺失一律**缺失**，不填充：

    - **窗口不满**（``min_periods=n``）：不足 n 根就无从谈最高/最低，故 RSV 缺失。与图上
      的差别要写明：行情软件的 ``LLV``/``HHV`` 在不足 n 根时按**已有的**几根算，故它从首根
      就有值；本项目的缺口纪律不给这种乐观答案。这段差异只影响序列开头 n−1 根，且递推会
      在约 m1×5 根内把差别衰减掉。
    - **分母为 0**：窗口内最高价等于最低价（一字板、长期停牌复牌）时 RSV 无定义。此处判
      缺失而**不是**沿用前值——与图上「平盘时 KDJ 保持前值」的观感不同，取的是「不猜」。
    - **数据缺口**：    缺失把序列切成若干段，每段以第一个可用值播种，缺口处保持缺失
    （与 :func:`_smooth_frame` 同一纪律）。
    """
    high = check_symbol_frame(panel["high"])
    low = check_symbol_frame(panel["low"])
    close = check_symbol_frame(panel["close"])

    highest = high.rolling(n, min_periods=n).max()
    lowest = low.rolling(n, min_periods=n).min()
    span = highest - lowest
    # 分母为 0 时 RSV 无定义：先算出比值、再把分母非正的位置判为缺失，
    # 免得 0/0 的 NaN 与「窗口不满」的 NaN 混为一谈。
    rsv = (close - lowest) / span * 100.0
    rsv = rsv.where(span > 0)

    k = _smooth_frame(rsv, 1.0 / m1)
    d = _smooth_frame(k, 1.0 / m2)
    return KDJ(k=k, d=d, j=3.0 * k - 2.0 * d)


def volume_ratio(volumes: pd.DataFrame, n: int) -> pd.DataFrame:
    """成交量比：``当根成交量 ÷ 前 n 根均量``。

    基准取**前 n 根**而非含当根：含当根会把当根的放量本身算进基准，于是放量越猛基准越高、
    比值越小——正好与「放量」的语义相反（``volume_surge`` 判定放量即复用它）。

    .. warning::

        **它不是交易所口径的「量比」。** 官方量比是「当日开盘后**每分钟平均**成交量 ÷ 过去
        **5 个**交易日每分钟平均成交量」，是个**盘中**指标，用日线数据**无法忠实复现**。
        本项目刻意不实现量比，也刻意不用那个词——用一个有法定含义的词去指另一件事，正是
        本项目反复警惕的失真。
    """
    volumes = check_symbol_frame(volumes)
    prior_mean = volumes.rolling(n, min_periods=n).mean().shift(1)
    return volumes / prior_mean


def _wilder_smooth(true_range: pd.Series, n: int) -> pd.Series:
    """对一个标的的真实波幅做**分段** Wilder 平滑（``atr`` 的零件）。

    缺失把序列切成若干段，每段各自播种：一段需**连续** n 个真实波幅，其简单均值放在该段
    第 n 根，此前缺失；其后走 Wilder 递推。不足 n 个真实波幅的段整段缺失。

    这里**逐段**调用 ``ewm`` 而不是整列调用一次，是因为整列调用会连缺失处的毛病一起带来：
    ``ewm(adjust=False)`` 遇到缺失会**沿用前值**（把缺失位置填成上一次的状态），且会把缺口
    之后的第一个原始真实波幅当成新的起点——那等于报出一个未经平滑的值。逐段处理两个问题
    都不存在，且段内的递推仍由 ``ewm`` 在 C 层完成。
    """
    valid = true_range.notna()
    if not valid.any():
        return true_range

    run_id = (~valid).cumsum()
    out = pd.Series(np.nan, index=true_range.index, dtype=float)
    for _, run in true_range[valid].groupby(run_id[valid]):
        if len(run) < n:
            continue
        seeded = run.copy()
        seeded.iloc[: n - 1] = np.nan
        seeded.iloc[n - 1] = run.iloc[:n].mean()
        out.loc[run.index] = seeded.ewm(alpha=1.0 / n, adjust=False).mean()
    return out


def atr(panel: Panel, n: int) -> pd.DataFrame:
    """平均真实波幅（ATR），**Wilder 平滑**口径。

    真实波幅（TR）逐根为下三者的最大：

    .. code-block:: text

        high − low
        |high − 前收|
        |low  − 前收|

    真实波幅的参照收盘价取**最近一个可得**的收盘价。前一根停牌时它没有收盘价，而无论
    「凭空当 0」还是「让整根作废」都会丢掉复牌日的跳空——那恰恰是真实波幅要度量的东西。
    这**不是**被禁止的填充：价格序列本身不变、输出在缺失处也仍然缺失，这里只是给真实波幅
    挑一个参照。序列开头没有参照收盘价，退回 ``high − low``。

    ATR 以**连续 n 个真实波幅的简单均值**播种于该段的第 n 根，其后走 Wilder 递推：

    .. code-block:: text

        ATR_t = (ATR_{t-1} × (n − 1) + TR_t) ÷ n

    **缺失把序列切成若干段，每段各自播种。** 一段不足 n 个真实波幅即整段缺失；缺口之后
    需重新积累 n 个连续值才重新播种。缺失处一律是缺失——**不沿用**前值，因为那会把停牌日
    伪造成一个有波幅的交易日（ADR-0005 的缺口纪律）。

    口径必须写明而不是默认，因为「ATR」不加限定会被读成简单均线版，而两者数值不同：
    在 n=3 的样本上手算，Wilder 口径后两值为 1.3889 / 1.7593，简单均线版为 1.5 / 1.8333。

    参数:
        panel: 须含 ``high`` / ``low`` / ``close`` 三个字段，且三者同日对齐
            （对齐由 :class:`~mbt.data.panel.Panel` 保证）。
        n: 平滑窗口。

    返回:
        标的宽表。形态与 :func:`sma` 相同，故它可直接当排序因子用。
    """
    high = check_symbol_frame(panel["high"])
    low = check_symbol_frame(panel["low"])
    close = check_symbol_frame(panel["close"])

    reference_close = close.ffill().shift(1)
    true_range = np.maximum(
        np.maximum(high - low, (high - reference_close).abs()),
        (low - reference_close).abs(),
    )
    # 序列开头没有参照收盘价：三者的最大值全为缺失，故退回 high − low。
    true_range = (high - low).where(reference_close.isna(), true_range)

    return true_range.apply(_wilder_smooth, n=n)
