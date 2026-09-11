"""指标：由**单一标的**的价量序列算出、本身不含任何判断的数值序列（``CONTEXT.md``）。

它们是过滤信号与排序因子的**构件**——「放量」与「站上均线」都组合这里的滚动窗口，而不是
各写一份。窗口一律取「当根及其之前 n-1 根」且 ``min_periods=n``：既不含未来信息，也不把
不足窗口的短缺当成数值——缺失就让它缺失（ADR-0005 的缺口纪律）。

本模块多数函数取**单字段**的标的宽表；``atr`` 是第一个需要**跨字段**的，故它取
:class:`~mbt.data.panel.Panel`。跨字段的信号**返回标的宽表**（ADR-0009）：消费方向与其余的
指标完全一样（列 = 时序、行 = 截面），只是输入多了两张表。
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from mbt.data.panel import Panel
from mbt.signals._symbol_frame import check_symbol_frame


def sma(prices: pd.DataFrame, n: int) -> pd.DataFrame:
    """n 日简单移动平均，含当根；窗口不足处为缺失值。"""
    return check_symbol_frame(prices).rolling(n, min_periods=n).mean()


def rolling_max(prices: pd.DataFrame, n: int) -> pd.DataFrame:
    """n 日最高价，含当根；窗口不足处为缺失值。"""
    return check_symbol_frame(prices).rolling(n, min_periods=n).max()


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
