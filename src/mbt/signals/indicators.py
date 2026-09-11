"""指标：由**单一标的**的价量序列算出、本身不含任何判断的数值序列（``CONTEXT.md``）。

它们是过滤信号与排序因子的**构件**——「放量」与「站上均线」都组合这里的滚动窗口，而不是
各写一份。窗口一律取「当根及其之前 n-1 根」且 ``min_periods=n``：既不含未来信息，也不把
不足窗口的短缺当成数值——缺失就让它缺失（ADR-0005 的缺口纪律）。
"""

from __future__ import annotations

import pandas as pd

from mbt.signals._wide import check_wide


def sma(prices: pd.DataFrame, n: int) -> pd.DataFrame:
    """n 日简单移动平均，含当根；窗口不足处为缺失值。"""
    return check_wide(prices).rolling(n, min_periods=n).mean()


def rolling_max(prices: pd.DataFrame, n: int) -> pd.DataFrame:
    """n 日最高价，含当根；窗口不足处为缺失值。"""
    return check_wide(prices).rolling(n, min_periods=n).max()
