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

import backtrader as bt
import pandas as pd

from mbt.rules import limit_price, same_price


class AStockCommissionInfo(bt.CommissionInfo):
    """按成交日查表计算 A 股交易费用。

    费用由四部分构成：

    - **手续费**：按成交金额乘费率，并受单笔最低金额约束。费率由用户给定（券商约定）。
    - **印花税**：仅卖出方缴纳，费率按成交日查表。
    - **过户费**：费率与收取方向均按板块与成交日查表。
    - **经手费 + 证管费**：按板块与成交日查表，双向。**仅当** ``commission_mode="net"``
      时叠加——券商报「全佣」时这两项已含在费率里，再叠加就是重复计费。

    成交日与标的由 :class:`AStockBroker` 在费用计算前注入。
    """

    params = (
        ("rules", None),
        ("commission_min", 0.0),
        ("commission_mode", "all_in"),
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
        # 非股票标的（指数、基金、可转债）没有板块，这里**报错而不兜底**：把股票那套
        # 费率硬套上去会让成本悄悄算错（ADR-0002），而「不猜」是本项目在制度参数上的
        # 一贯立场（见 `mbt.rules.board_of` 与 `_limit_locked`）。
        #
        # 后果是「对一个非股票标的下单」会在**计算仓位**时即失败——那正是应有的响亮
        # 反应：这类标的本就不得按股票制度交易，股票池也会把它排除。
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

            # 经手费与证管费总是发生，但「全佣」报价已把它们计入费率里，
            # 故只在「净佣」口径下叠加，免得把同一笔钱收两遍。
            if self.p.commission_mode == "net":
                fee += rules.handling_fee_rate(self._board, date) * gross
                fee += rules.regulatory_fee_rate(self._board, date) * gross

        return fee


class AStockBroker(bt.brokers.BackBroker):
    """撮合侧的 A 股硬约束，以及把成交日与标的注入费用对象。

    四条约束共用 ``_try_exec`` 这一个接入点，但**处置方式刻意不同**：

    - **涨跌停不可成交、当日无成交**：本日**不具备成交条件**，故保留挂单，
      引擎会自动把它挪到下一根 K 线再试（即用户选定的「挂单保留至下一可交易日」）。
    - **T+1**：本单在当日**不合法**，故 ``reject()`` 终结，不拖到次日。
    - **股票池外买入**：资格问题不会因等待而消失，故 ``reject()`` 终结并留痕。
      **卖出不受限**——持仓可能因调仓或数据变化掉出池子，禁卖会把仓位卡死。
    - **长期无 K 线**：标的停牌超过 ``order_expiry_ticks`` 个交易日即 ``reject()``。
      「保留至下一可交易日」在单标的下是「下一天」，停牌数月时就变成「另一笔交易」。

    区分依据是引擎的挂单循环：``_try_exec`` 返回后，订单若仍 ``alive()`` 就会被
    重新放回 pending 队列；被拒则不再存活。因此「不成交」只需什么都不做。

    ``getcommissioninfo`` 是本层唯一另一处接入点：它在每次费用计算前被调用，且
    调用时的当前 K 线就是成交所在的那根，于是成交日得以注入费用对象。

    ## 为什么需要 ``tradability`` 掩码

    单标的路径不需要它——停牌日**根本没有那一行**，引擎见不到它，``volume[0] == 0``
    就够了。但**多标的场景下 backtrader 在停牌日返回的是上一根的陈旧 K 线**：
    ``volume`` 看着完全正常，于是订单会按**陈旧价成交**。实测（8 根 K 线、B 缺 2 天）：
    订单在 B 停牌当天按 20.0 成交了。那是「凭空造出一笔没有成交的成交」，正是 ADR-0002
    要防的虚假信心，且它是**组合场景独有的**。

    掩码是逐日的「当日具备成交条件」布尔表（有 K 线且成交量 > 0），故它能答出
    ``data`` 自己答不了的问题：**今天这根到底是新的还是陈旧的**。
    """

    params = (
        ("rules", None),
        #: boolean 标的宽表：当日具备成交条件（有 K 线且量 > 0）。多标的场景必填。
        ("tradability", None),
        #: boolean 标的宽表：股票池。给出时，池外标的的**买入**被拒、卖出不受限。
        ("universe", None),
        #: 标的连续多少个交易日无 K 线后，挂单失效。默认 5。
        ("order_expiry_ticks", 5),
        #: 最大持仓**标的数**。``None`` 表示不限。
        ("max_positions", None),
    )

    def __init__(self):
        super().__init__()
        #: data → (成交日, 该日买入的股数)。用于 T+1 判定。
        self._bought_today: dict = {}
        #: order.ref → 已记入的累计成交股数，避免部分成交被重复计数。
        self._counted: dict = {}
        #: order.ref → (最后计数的 tick 日期, 连续无 K 线天数)，用于挂单失效。
        self._stale: dict = {}
        #: 本 tick 的主时钟日期，由 :meth:`next` 缓存（见 :meth:`_today`）。
        self._clock = None

    def next(self):
        # 主时钟每个 tick 只算一次：它要遍历全部标的，放进 _try_exec 会变成每个订单一次。
        self._clock = self._compute_clock()
        super().next()

    @property
    def tradability_mask(self):
        """当日**是否具备成交条件**的布尔标的宽表（有 K 线且量 > 0）。

        停牌日为 ``False``，而此时 ``data.close[0]`` 返回的是**陈旧价**。策略读价格做决策前
        必须先查这张表，否则会基于几个月前的价格下单——那不会报错，只会让结论失真
        （ADR-0006）。
        """
        return self.p.tradability

    @property
    def universe_mask(self):
        """**股票池**的布尔标的宽表。撮合层已强制「池外不可买」，这张表供策略自查
        （例如察觉自己的持仓已掉出池子）。"""
        return self.p.universe

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
            return self._reject(order)

        if order.isbuy() and self._mask_says(self.p.universe, order) is False:
            # 股票池外买入：资格问题不会因等待而消失，故拒单而不是保留挂单。
            return self._reject(order)

        if order.isbuy() and self._at_position_limit(order):
            # 已持满最大持仓数时，对**新标的**的买入被拒；对已有持仓加仓不受限，
            # 因为那不会增加持仓的标的数。
            return self._reject(order)

        if not self._tradeable_today(order):
            return  # 保留挂单：仍 alive 的订单会被引擎放回队列，下一根再试

        return super()._try_exec(order)

    def _at_position_limit(self, order) -> bool:
        """本笔买入是否会让持仓的**标的数**超过上限。"""
        limit = self.p.max_positions
        if limit is None:
            return False
        if self.positions[order.data].size:
            return False  # 加仓，不增加标的数
        held = sum(1 for position in self.positions.values() if position.size)
        return held >= limit

    def _reject(self, order):
        order.reject()
        self.notify(order)
        # 订单已终结，它的计数没有留存价值；不清理会让这张表随回测长度无界增长。
        self._stale.pop(order.ref, None)

    def _clear_stale(self, order):
        """标的重新有了 K 线——停牌的那一串中断了，计数随之归零。"""
        self._stale.pop(order.ref, None)

    def _today(self):
        """引擎的**主时钟**日期，即当日真实的交易日。

        不能用 ``data.datetime.date(0)``：标的停牌时它停在**上一根** K 线的日期，据此
        查掩码会把停牌日误判为可交易。主时钟取各标的当前日期里的**最大者**——引擎把每个
        标的推进到不超过时钟的位置，故有 K 线的标的其日期正是时钟，而停牌的那个落后，
        最大值因此恰是时钟本身。
        """
        if self._clock is None:
            self._clock = self._compute_clock()
        return self._clock

    def _compute_clock(self):
        latest = None
        for data in getattr(self.cerebro, "datas", ()):
            if len(data) == 0:
                continue  # 该标的尚未开始（首根 K 线晚于当前 tick）
            current = data.datetime.date(0)
            if latest is None or current > latest:
                latest = current
        return latest

    def _mask_says(self, frame, order):
        """查标的宽表掩码。返回 ``None`` 表示「无掩码，无从判断」。"""
        symbol = getattr(order.data, "_mbt_symbol", None)
        on = self._today()
        if frame is None or symbol is None or on is None:
            return None
        if symbol not in frame.columns:
            return False
        stamp = pd.Timestamp(on)
        if stamp not in frame.index:
            return False
        return bool(frame.at[stamp, symbol])

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
        """当日是否具备成交条件，含「这根 K 线是不是今天的」。"""
        data = order.data

        if self.p.tradability is not None:
            if self._mask_says(self.p.tradability, order) is not True:
                # 当日无 K 线（停牌）或为零量。挂单可以再等，但等太久就作废——否则一笔单
                # 会在停牌数月后的复牌日突然成交，那已不是原来那笔交易。
                self._expire_if_stale(order)
                return False
            self._clear_stale(order)
        elif len(self.cerebro.datas) > 1:
            # 多标的却没给掩码：此时无从分辨 K 线是新的还是陈旧的，而那会造出按陈旧价
            # 成交的订单。宁可报错也不静默放过。
            raise RuntimeError(
                "多标的回测必须提供 tradability 掩码：停牌日 backtrader 会返回上一根的"
                "陈旧 K 线，没有掩码就会按陈旧价成交（凭空造出未发生的成交）。"
                "请用 run_portfolio_backtest，它会自动构建掩码。"
            )
        elif data.volume[0] == 0:
            # 当日无成交。股票停牌是缺记录，但基金/债券会留下零量的平价 K 线。
            return False

        return not self._limit_locked(order)

    def _expire_if_stale(self, order) -> bool:
        """标的连续无 K 线达 ``order_expiry_ticks`` 天时拒单。返回是否已拒。

        计数按**交易日**而非调用次数：``_try_exec`` 每个 tick 只会被调一次，但用日期
        去重可以让计数不依赖引擎的调用节奏。
        """
        today = self._today()
        last_date, days = self._stale.get(order.ref, (None, 0))
        if today is not None and last_date == today:
            return days >= self.p.order_expiry_ticks

        days += 1
        self._stale[order.ref] = (today, days)

        # 「超过 N 个交易日」是**严格大于**：第 N 个无 K 线的交易日仍算在容忍期内。
        if days > self.p.order_expiry_ticks:
            self._reject(order)
            return True
        return False

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
            return same_price(close, limit_price(prev_close, limit, 1))
        return same_price(close, limit_price(prev_close, limit, -1))
