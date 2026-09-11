"""测试共用的小工具（非 fixture）。

指标层与产物层都要构造「净值曲线」与「成交明细」，两处各写一份就会漂移——尤其成交明细的
列名是 ``mbt.backtest.TRADE_COLUMNS`` 的契约，改一处忘一处会让测试与实现各说各话。
"""

from __future__ import annotations

import pandas as pd


def curve(values, start="2024-01-02"):
    """由资产序列构造净值曲线（交易日为索引、升序）。"""
    return pd.Series(values, index=pd.bdate_range(start, periods=len(values)), dtype=float)


def trades_table(rows, start="2024-01-02", periods=10):
    """由 ``(日期序号, size, price, commission)`` 构造成交明细。

    列与命名都与 ``mbt.backtest.TRADE_COLUMNS`` 一致；``value`` 由 ``size × price`` 得出
    （与撮合层记录的口径相同）。
    """
    index = pd.bdate_range(start, periods=periods)
    return pd.DataFrame(
        [
            {
                "date": index[day],
                "size": size,
                "price": price,
                "value": size * price,
                "commission": commission,
            }
            for day, size, price, commission in rows
        ]
    )
