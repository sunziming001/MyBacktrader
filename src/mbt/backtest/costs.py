"""A 股交易费用的计算与注入。

费用必须在**撮合时**进入现金流，而不是事后从净值里扣：否则 broker 看到的可用资金偏多，
仓位数量会随之偏大，回测结果与实际不符。

难点在于取费率需要**成交日**，而 backtrader 的 ``getcommission(size, price)`` 不带日期。
解法：broker 在计算费用的那一刻被调用 ``getcommissioninfo(data)``，此时
``data.datetime.date(0)`` 正是成交日，于是把日期注入费用对象即可。
这比事后修正净值更贴近引擎的真实撮合路径，且策略无法绕过。
"""

from __future__ import annotations

import backtrader as bt


class AStockCommissionInfo(bt.CommissionInfo):
    """按成交日查表计算 A 股交易费用。

    费用由三部分构成：

    - **手续费**：按成交金额乘费率，并受单笔最低金额约束。费率由用户给定（券商约定）。
    - **印花税**：仅卖出方缴纳，费率按成交日查表。
    - **过户费**：费率与收取方向均按板块与成交日查表。

    成交日与标的由 :class:`AStockBroker` 在费用计算前注入。
    """

    params = (
        ("rules", None),
        ("commission_min", 0.0),
    )

    def __init__(self):
        super().__init__()
        self.trade_date = None
        self.symbol = None
        self._board = None

    def for_execution(self, symbol, on) -> None:
        """由 broker 在费用计算前调用，注入本次成交的标的与日期。"""
        self.symbol = symbol
        self.trade_date = on
        self._board = self.p.rules.board_of(symbol) if symbol else None

    def _getcommission(self, size, price, pseudoexec):
        gross = abs(size) * price
        date = self.trade_date
        rules = self.p.rules

        fee = max(self.p.commission * gross, self.p.commission_min)

        if size < 0:  # 卖出
            fee += rules.stamp_duty_rate(date) * gross

        if self._board is not None:
            sides = rules.transfer_fee_sides(self._board, date)
            charged = (
                sides == "both" or (sides == "buy" and size > 0) or (sides == "sell" and size < 0)
            )
            if charged:
                fee += rules.transfer_fee_rate(self._board, date) * gross

        return fee


class AStockBroker(bt.brokers.BackBroker):
    """在费用计算那一刻把成交日与标的注入费用对象。

    这是本层唯一需要触碰引擎内部的地方：``getcommissioninfo`` 恰好在每次费用计算前
    被调用，且调用时的当前 K 线就是成交所在的那根。
    """

    def getcommissioninfo(self, data):
        info = super().getcommissioninfo(data)

        inject = getattr(info, "for_execution", None)
        if inject is not None:
            inject(getattr(data, "_mbt_symbol", None), data.datetime.date(0))

        return info
