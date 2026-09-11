"""A 股交易费用的计算与注入。

费用必须在**撮合时**进入现金流，而不是事后从净值里扣：否则 broker 看到的可用资金偏多，
仓位数量会随之偏大，回测结果与实际不符。

难点在于取费率需要**成交日**，而 backtrader 的 ``getcommission(size, price)`` 不带日期。
解法：broker 在计算费用的那一刻被调用 ``getcommissioninfo(data)``，此时
``data.datetime.date(0)`` 正是成交日，于是把日期注入费用对象即可。
这比事后修正净值更贴近引擎的真实撮合路径，且策略无法绕过。

本模块同时承载撮合侧的硬约束（T+1、涨跌停不可成交、停牌跳过）。它们与费用同属
「撮合按当日制度执行」这一件事，故共用 :class:`AStockBroker` 这一个接入点。
"""

from __future__ import annotations

from decimal import ROUND_HALF_UP, Decimal

import backtrader as bt

#: 价格的最小单位：分。
_CENT = Decimal("0.01")

#: 价比的容差。价格精确到分，此处仅为吸收 ``float`` 的表示误差，
#: 不是模糊阈值——判「是否触限」不应有模糊地带。
_PRICE_TOLERANCE = 1e-9


def _same_price(a: float, b: float) -> bool:
    """两个价格是否相等（容忍浮点表示误差）。"""
    return abs(a - b) < _PRICE_TOLERANCE


def _limit_price(prev_close: float, limit: float, direction: int) -> float:
    """当日涨跌停价：前收盘价 × (1 + 限幅 × 方向)，四舍五入到分。

    必须用**十进制**半进位，不能用内建 ``round()``：``round`` 走二进制浮点，
    在恰好半分处会少一分——前收 14.45 的跌停价应是 13.01，``round`` 给 13.00。

    代价不只是数字难看：差一分就足以**漏判一字板**，从而放过一笔现实中不可能
    的成交，正是 ADR-0002 要防的虚假信心。实测（见
    ``docs/research/tdx-halt-and-limit-representation.md``）：本机 16 只主板
    股票中两种算法给出不同限价的有 1,916 天（约 4.4%）；而在两者会分歧、且
    当日确实触及限价的 38 个交易日里，市场**全部**落在十进制半进位一侧，无一例外。
    """
    price = Decimal(str(prev_close)) * (Decimal(1) + Decimal(str(limit)) * direction)
    return float(price.quantize(_CENT, rounding=ROUND_HALF_UP))


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
    """撮合侧的 A 股硬约束，以及把成交日与标的注入费用对象。

    三条约束共用 ``_try_exec`` 这一个接入点，但**处置方式刻意不同**：

    - **涨跌停不可成交、当日无成交**：本日**不具备成交条件**，故保留挂单，
      引擎会自动把它挪到下一根 K 线再试（即用户选定的「挂单保留至下一可交易日」）。
    - **T+1**：本单在当日**不合法**，故 ``reject()`` 终结，不拖到次日。

    区分依据是引擎的挂单循环：``_try_exec`` 返回后，订单若仍 ``alive()`` 就会被
    重新放回 pending 队列；被拒则不再存活。因此「不成交」只需什么都不做。

    ``getcommissioninfo`` 是本层唯一另一处接入点：它在每次费用计算前被调用，且
    调用时的当前 K 线就是成交所在的那根，于是成交日得以注入费用对象。
    """

    params = (("rules", None),)

    def __init__(self):
        super().__init__()
        #: data → (成交日, 该日买入的股数)。用于 T+1 判定。
        self._bought_today: dict = {}
        #: order.ref → 已记入的累计成交股数，避免部分成交被重复计数。
        self._counted: dict = {}

    # --- 费用：注入成交日与标的 ---

    def getcommissioninfo(self, data):
        info = super().getcommissioninfo(data)

        inject = getattr(info, "for_execution", None)
        if inject is not None:
            inject(getattr(data, "_mbt_symbol", None), data.datetime.date(0))

        return info

    # --- 撮合约束 ---

    def _try_exec(self, order):
        if order.issell() and not self._t1_allows(order):
            # T+1：当日买入的股份当日不可卖。交易所就是拒单，故不得留到次日。
            order.reject()
            self.notify(order)
            return

        if not self._tradeable_today(order):
            return  # 保留挂单：仍 alive 的订单会被引擎放回队列，下一根再试

        return super()._try_exec(order)

    def _execute(self, order, ago=None, price=None, cash=None, position=None, dtcoc=None):
        result = super()._execute(
            order, ago=ago, price=price, cash=cash, position=position, dtcoc=dtcoc
        )

        # ago is None 表示伪执行（试探资金够不够）。它在 order.execute 之前就返回，
        # 不改订单状态，因此这里不会把未成交的单误记成已买入。
        if ago is not None:
            self._remember_bought_today(order)

        return result

    def _remember_bought_today(self, order):
        """记下本笔买入的股数，供 T+1 判定。

        按累计成交量取增量，故部分成交分批到来时不会重复计数。
        """
        if not order.isbuy():
            return

        executed = abs(order.executed.size)
        delta = executed - self._counted.get(order.ref, 0)
        if delta <= 0:
            return
        self._counted[order.ref] = executed

        data = order.data
        on = data.datetime.date(0)
        prev_date, prev_qty = self._bought_today.get(data, (on, 0))
        qty = prev_qty if prev_date == on else 0
        self._bought_today[data] = (on, qty + delta)

    def _t1_allows(self, order):
        """可卖量 = 持仓 − 当日买入量；卖出量超过可卖量即违规。

        这里把「当日买入」当作一整块扣减，没有逐笔追踪卖出的是哪一批股票。日线上
        一根 K 线只能下一次单、成交在下一根，日内多次往返实际不会出现，故无需
        lot 级追踪。代价是极端场景下偏保守——宁可少卖，也不放出不存在的成交。
        """
        data = order.data
        on = data.datetime.date(0)
        bought_on, qty = self._bought_today.get(data, (on, 0))
        bought_today = qty if bought_on == on else 0

        sellable = self.positions[data].size - bought_today
        return abs(order.created.size) <= sellable

    def _tradeable_today(self, order):
        """当日是否具备成交条件。缺规则表时只判「有无成交量」。"""
        data = order.data

        if data.volume[0] == 0:
            # 当日无成交。股票停牌是缺记录，但基金/债券会留下零量的平价 K 线。
            return False

        return not self._limit_locked(order)

    def _limit_locked(self, order):
        """本根 K 线是否是**挡住该订单方向**的一字板。

        判据必须同时满足两条：

        1. ``o == h == l == c``——全天只有一个价；
        2. 该价**恰在当日涨跌停价上**——由前收盘价与当日限幅算出。

        只凭形态（第 1 条）会把没有涨跌停的品种也算成锁死，故必须补上第 2 条。
        方向也必须分清：涨停一字**买不进但卖得出**，跌停一字**卖不出但买得到**
        ——后者是常被写错的：跌停时卖方排队，买方反而立即成交。
        """
        data = order.data
        if not (data.open[0] == data.high[0] == data.low[0] == data.close[0]):
            return False
        if len(data) < 2:  # 没有前收盘价就算不出涨跌停价
            return False

        symbol = getattr(data, "_mbt_symbol", None)
        if symbol is None or self.p.rules is None:
            return False  # 无制度信息则不判锁死——此处不猜

        close = data.close[0]
        prev_close = data.close[-1]
        limit = self.p.rules.limit_for(symbol, data.datetime.date(0))

        if order.isbuy():
            return _same_price(close, _limit_price(prev_close, limit, 1))
        return _same_price(close, _limit_price(prev_close, limit, -1))
