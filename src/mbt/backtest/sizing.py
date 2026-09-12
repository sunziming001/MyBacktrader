"""持仓分配：决定每个标的该分到多少资金（票据 #6）。

AC 要求「多标的共享资金池，有最大持仓数与仓位分配」，但**没有定义分配规则**。故这里给出
一个精确定义、能手算的默认实现，并把机制做成可插拔——「按排序因子加权」之类要等 #7 的
选股规则定了形状再说，现在猜就是空想。

这里的词按 `CONTEXT.md` 的分工用：**持仓**是「组合拿着哪些标的」，**仓位**是「其中每一只
占多大比例」。

## 默认口径：等权

设最大持仓（标的数）为 ``M``、当前已持有 ``H`` 只标的（不含正在下单的这一只），本单可用
资金为 ``C``，则本次买入的目标金额为::

    C ÷ max(1, M − H)

也就等于给每个持仓名额一个等额的**目标仓位**，依次把可用资金摊出去：第一笔用掉约 ``C/M``，
第二笔用掉剩余资金的 ``1/(M−1)``，最后一笔把剩余全部投出。它的好处是**每一步都能手算**，
且不会因为价格变动而需要回头修正既有的持仓。

## 尚未建模

- **A 股「买入必须是 100 股整数倍」**（一手 = 100 股）。本模块不做整手取整，故产出的股数
  可能不是 100 的倍数。这是一处**已知的乐观偏差**（真实下单会被拒或改单），已在 README 的
  风险清单里登记。
- 单笔上限、按波动率加权、行业分散等更复杂的分配规则。
"""

from __future__ import annotations

import backtrader as bt


class EqualWeightSizer(bt.Sizer):
    """等权 sizer：把可用资金摊给剩余的持仓名额（见模块说明）。

    每个持仓名额的目标**仓位**相同（``1/M``），故依次买入时自然把可用资金摊开。

    参数:
        max_positions: 最大持仓**标的数**。``None`` 表示不限——此时每个名额的目标仓位无从
            谈起，故本次买入用尽可用资金。那会让后续买入因资金不足被拒，而那些拒单会记进
            ``BacktestResult.rejected``，不会悄无声息地发生。
    """

    params = (("max_positions", None),)

    def _getsizing(self, comminfo, cash, data, isbuy):
        if not isbuy:
            # 卖出交由策略决定数量（通常是全部持仓），sizer 不插手。
            return self.broker.getposition(data).size

        held = sum(1 for position in self.broker.positions.values() if position.size)
        if self.p.max_positions is None:
            slots = 1
        else:
            slots = self.p.max_positions - held
            if slots <= 0:
                return 0

        price = data.close[0]
        if price <= 0:
            return 0

        return self._affordable(comminfo, cash / slots, price)

    @staticmethod
    def _affordable(comminfo, budget, price) -> int:
        """在 ``budget`` 之内买得起的最大股数——**含交易费用**。

        为什么不能只算 ``budget / price``：撮合层要求「成交金额 + 费用 ≤ 现金」，而费用是
        **制度必收**的（过户费与佣金无关，``commission_mode="net"`` 时还有经手费与证管费）。
        于是 `budget / price` 几乎正好用光预算时，加上费用就超——**每一笔买单都以 ``Margin``
        被拒**。实测单标的（``max_positions`` 省略或为 1）时零成交，而退出码仍是 0：不报错，
        只是回测空转。**这是本项目最防的那类失败。**

        做法是**二分**：``size * price + fee(size)`` 对 ``size`` 严格单调递增（``size*price``
        递增，``fee`` 单调不减），故在 ``[0, int(budget/price)]`` 上二分即可求出满足约束的
        最大整数，精确且无需知道费率。

        刻意**不**乘一个 0.999 之类的安全系数：那既盖不住 ``commission=0.3%`` 这类配置，也会
        在费率很小时白白少买，而两者都不会报错。

        为什么要用 ``comminfo`` 而不用自己的费率常量：backtrader 的 ``Sizer.getsizing`` 会先调
        ``broker.getcommissioninfo(data)``（本项目的 ``AStockBroker`` 在那里把**成交日与板块**
        注入费用对象），**再**调 ``_getsizing``。所以这里的 ``comminfo`` 已能算出当日当板块的
        真实费用——照抄一份费率表出来只会与之漂移（Shotgun Surgery）。
        """
        low, high = 0, int(budget / price)
        while low < high:
            mid = (low + high + 1) // 2
            if mid * price + comminfo.getcommission(mid, price) <= budget:
                low = mid
            else:
                high = mid - 1
        return low
