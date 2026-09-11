"""过滤信号：对**单一标的**的是/否判断，回答「它此刻是否合格」（``CONTEXT.md``）。

两条约定在此落地：

1. 输出是**纯 ``bool``** 的宽表，缺失一律取 ``False``，语义是「不合格」。这不是额外代码，
   而是**数值比较的天然结果**——``nan > x`` 与 ``x > nan`` 都返回 ``False``。刻意不把缺失
   先填成某个数：填了就替数据「表了态」，而本项目宁可漏判也不凭空造出合格（ADR-0005）。
2. 判断只在**已知**上成立。窗口不足时不给乐观答案——``min_periods=n`` 让前 n-1 根为缺失，
   比较后即为 ``False``。

「状态」与「事件」两类不要混淆：``above_ma`` 回答「此刻在均线上方吗」，
``ma_cross_up`` 回答「今天刚上穿吗」，后者不是前者的子集。
"""

from __future__ import annotations

import pandas as pd

from mbt.signals._wide import check_wide
from mbt.signals.indicators import rolling_max, sma


def _prior_average(values: pd.DataFrame, n: int) -> pd.DataFrame:
    """前 n 根的均值，**不含当根**。``shift(1)`` 就是这一层的时点保证所在。"""
    return values.rolling(n, min_periods=n).mean().shift(1)


def new_high(prices: pd.DataFrame, n: int) -> pd.DataFrame:
    """创 n 日新高：当根收盘**严格高于**前 n 个交易日的最高收盘价。

    用 ``>`` 而非 ``>=``：恰好等于前高不是「创」新高。
    """
    return check_wide(prices) > rolling_max(prices, n).shift(1)


def above_ma(prices: pd.DataFrame, n: int) -> pd.DataFrame:
    """站上均线：当根收盘高于 n 日均线（含当根）。这是一个**状态**。"""
    return check_wide(prices) > sma(prices, n)


def ma_cross_up(prices: pd.DataFrame, n: int) -> pd.DataFrame:
    """均线上穿：当根收盘上穿 n 日均线，即昨日在下（或持平）、今日在上。这是一个**事件**。

    昨日的均线必须**已知**，否则「上穿」会退化成「第一次算得出均线」——那不是穿越，
    只是窗口刚好填满，把它当信号会在每只次新股的同一位置凭空点火。
    """
    prices = check_wide(prices)
    ma = sma(prices, n)
    return (prices > ma) & ma.shift(1).notna() & (prices.shift(1) <= ma.shift(1))


def rising_streak(prices: pd.DataFrame, n: int) -> pd.DataFrame:
    """连续上涨 n 日：最近 n 个交易日**每一次**收盘都高于前一日（故需要 n+1 根 K 线）。"""
    prices = check_wide(prices)
    return (prices.diff() > 0).rolling(n, min_periods=n).sum() == n


def volume_surge(volumes: pd.DataFrame, k: float, n: int) -> pd.DataFrame:
    """放量 k 倍：当根成交量**严格高于**前 n 日均量的 k 倍。

    基准取**前 n 根**（见 ``_prior_average``）而非含当根：含当根会把当根的放量本身算进
    基准，于是放量越猛基准越高、信号越难触发——正好与「放量」的语义相反。
    """
    return check_wide(volumes) > _prior_average(volumes, n) * k
