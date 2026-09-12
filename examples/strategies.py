"""示例策略：**不是库的一部分**，放在这里只为回答「``--strategy`` 那一串该怎么写」。

```
mbt backtest --strategy examples.strategies:BuyAndHold ...          # 从仓库根跑
mbt backtest --strategy examples.strategies:CloseCrossesAboveMA ... # 均线上穿
mbt backtest --strategy mypkg.strategies:MyStrategy ...             # 你自己的包
```

写自己的策略时：

- 类继承 ``backtrader.Strategy``（ADR-0001：策略用 Python 表达，不造声明式中间层）；
- 参数用 ``--param name=value`` 传入，会作为关键字参数转给策略类；
- **先把下面 :class:`EngineClock` 抄走**——多标的回测里取「今天」比看上去难（见它的说明）；
- 读价格前**必须**查 ``self.broker.tradability_mask``：停牌日是**陈旧价**，而**尚未上市**的标的
  给的是它**最后一根**的价格（即未来价）——两者都不报错，只让结论失真（ADR-0006）。

引擎会从**第一根** K 线起就调用 ``next()``（票据 #31），故晚上市的标的不会阻塞其余标的，
``len(self)`` 就是「引擎跑到第几根」。但若你自己实现了 ``prenext``，引擎就不再插手——那时请
自己把 ``next`` 接过去，否则又会退回「等全体就绪」。

三个掩码（都由撮合层强制，策略读它们是为了**自己做决定**，不是替代约束）：

===================  ========================================  ====================
掩码                 回答什么                                  为 ``None`` 时
===================  ========================================  ====================
``tradability_mask``  当天这根 K 线是不是**今天**的（有量）    不会
``universe_mask``     当天是否在**股票池**内                    单标的入口下为 ``None``
``selection_mask``    当天是否被**选股规则**选中                没给选股规则时为 ``None``
===================  ========================================  ====================
"""

from __future__ import annotations

import backtrader as bt
import pandas as pd


class EngineClock:
    """取**引擎主时钟**：各标的当前日期里的**最大者**。这就是「今天」。

    做法就是把 ``self.datas`` 扫一遍取最大值，理由是：有 K 线的标的其日期正是时钟，停牌的
    或尚未开始的落后，故最大值恰是时钟本身。

    **不要用 ``self.data0.datetime.date(0)``。** 若第一个标的的历史比回测区间短（次新股、
    北交所早期、或 ``--limit`` 随手选中的任意一只），它的日期会**一直停在末根**，于是掩码
    查到的是**另一天**——不报错，只是结果错。引擎内部记账时踩过同一个坑（见
    ``mbt.backtest.engine._EngineClock``，本类与它同源）。

    .. note::

        这个类其实是**库该提供的东西**——每个多标的策略都需要它，抄在示例里意味着每人都得
        重新推导一次（而第一个版本就写错了）。把它提升进 ``mbt`` 是对的方向，但那是一次
        库 API 变更，故此处先按示例给出。
    """

    def today(self) -> pd.Timestamp:
        latest = None
        for data in self.datas:
            if len(data) == 0:
                continue  # 该标的尚未开始（首根晚于当前 tick）
            current = data.datetime.date(0)
            if latest is None or current > latest:
                latest = current
        return pd.Timestamp(latest)


class BuyAndHold(bt.Strategy, EngineClock):
    """第一次有机会就买满，其后不动。

    参数:
        size: 每次下单的股数。默认 100（一手）。传了它就**绕过**引擎的等权 sizer。
    """

    params = (("size", 100),)

    def next(self):
        today = self.today()
        mask = self.broker.tradability_mask
        for data in self.datas:
            if self.getposition(data).size:
                continue
            if mask is not None and not mask.at[today, data._name]:
                continue
            self.buy(data=data, size=self.p.size)


class CloseCrossesAboveMA(bt.Strategy, EngineClock):
    """收盘价上穿均线就买入，下穿或**出池**就卖出。

    **入场条件与信号层的 :func:`mbt.signals.ma_cross_up` 同口径**——都是「收盘上穿
    ``SMA(n)``」，是**单**均线，不是双均线交叉。要改条件时先看信号层有没有现成的定义：
    ADR-0001 的立意就是「同一条件不必写两遍」，写两遍必然漂移。

    参数:
        n: 均线窗口。默认 20。
    """

    params = (("n", 20),)

    def __init__(self):
        # 每个标的各一条均线，按 `_name` 索引——`self.datas` 的顺序**不等于**掩码的列顺序，
        # 而掩码是按标的符号取值的。
        self.ma = {data._name: bt.ind.SMA(data.close, period=self.p.n) for data in self.datas}

    def next(self):
        today = self.today()
        broker = self.broker
        n = self.p.n

        for data in self.datas:
            name = data._name
            holding = self.getposition(data).size

            if holding:
                # 1) 出池 → 清仓。这正是「股票池」与「今天没被选中」分成两张掩码的理由：
                #    出池要清仓，只是没被选中则未必。
                if broker.universe_mask is not None and not broker.universe_mask.at[today, name]:
                    self.close(data=data)
                    continue

                # 2) 下穿 → 清仓。注意信号层目前**只有** `ma_cross_up`，没有对应的下穿
                #    定义——故这一个条件暂时是「写了两遍」的那一半。
                if len(data) >= n + 1:
                    ma = self.ma[name]
                    if data.close[0] < ma[0] and data.close[-1] >= ma[-1]:
                        self.close(data=data)
                continue

            # --- 以下只在下单前判断，故必然无持仓 ---

            # 3) 停牌日这根 K 线是陈旧的，`close[0]` 不可信——先问掩码。
            if not broker.tradability_mask.at[today, name]:
                continue

            # 4) 选股规则（未给时为 None，此时不加这层限制）。
            if broker.selection_mask is not None and not broker.selection_mask.at[today, name]:
                continue

            # 5) 上穿。要「今日均线」与「昨日均线」**都已知**——`len(data)` 已含当根，
            #    故需 n+1 根。少了这道判断，「上穿」会退化成「第一次算得出均线」，在每只
            #    次新股的同一位置凭空点火（信号层为同一理由加了 `ma.shift(1).notna()`）。
            if len(data) < n + 1:
                continue
            ma = self.ma[name]
            if data.close[0] > ma[0] and data.close[-1] <= ma[-1]:
                # 不传 size → 交给引擎的等权 sizer（`EqualWeightSizer`）。
                # 传了 size 会让 sizer 失效，`max_positions` 之下的等权分配就没了。
                self.buy(data=data)
