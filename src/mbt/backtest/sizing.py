"""仓位分配（票据 #6）。

AC 要求「多标的共享资金池，有最大持仓数与仓位分配」，但**没有定义分配规则**。故这里给出
一个精确定义、能手算的默认实现，并把机制做成可插拔——「按排序因子加权」之类要等 #7 的
选股规则定了形状再说，现在猜就是空想。

## 默认口径：把剩余资金摊给剩余仓位

设最大持仓数为 ``M``、当前已持有 ``H`` 只标的（不含正在下单的这一只），本单可用资金为
``C``，则本次买入的目标金额为::

    C ÷ max(1, M − H)

于是依次下单时，第一笔用掉约 ``C/M``，第二笔用掉剩余资金的 ``1/(M−1)``，最后一笔把剩余
全部投出。它的好处是**每一步都能手算**，且不会因为价格变动而需要回头修正既有仓位。

## 尚未建模

- **A 股「买入必须是 100 股整数倍」**（一手 = 100 股）。本模块不做整手取整，故产出的股数
  可能不是 100 的倍数。这是一处**已知的乐观偏差**（真实下单会被拒或改单），已在 README 的
  风险清单里登记。
- 单笔上限、按波动率加权、行业分散等更复杂的分配规则。
"""

from __future__ import annotations

import backtrader as bt


class EqualWeightSizer(bt.Sizer):
    """把剩余资金摊给剩余仓位的 sizer（见模块说明）。

    参数:
        max_positions: 最大持仓数。``None`` 表示不限——此时每次买入用掉**全部**可用资金，
            在单标的回测里等价于「满仓一次买入」。
    """

    params = (("max_positions", None),)

    def _getsizing(self, comminfo, cash, data, isbuy):
        if not isbuy:
            # 卖出交由策略决定数量（通常是全部持仓），sizer 不插手。
            return self.broker.getposition(data).size

        held = sum(1 for position in self.broker.positions.values() if position.size)
        if self.p.max_positions is None:
            # 不限持仓数时「剩余仓位」无从谈起，故本次买入用尽可用资金。
            # 这会让后续买入因资金不足被拒——那些拒单会记进 ``BacktestResult.rejected``，
            # 所以不会悄无声息地发生。
            slots = 1
        else:
            slots = self.p.max_positions - held
            if slots <= 0:
                return 0

        price = data.close[0]
        if price <= 0:
            return 0

        return int(cash / slots / price)
