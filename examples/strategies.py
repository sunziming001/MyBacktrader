"""示例策略：买入并持有。

**它不是库的一部分**，放在这里只为回答一个问题：``--strategy`` 那一串该怎么写。

```
mbt backtest --strategy examples.strategies:BuyAndHold ...       # 从仓库根跑
mbt backtest --strategy mypkg.strategies:BuyAndHold ...          # 你自己的包
```

写自己的策略时：

- 类继承 ``backtrader.Strategy``（ADR-0001：策略用 Python 表达，不造声明式中间层）；
- 位置参数通过 ``--param name=value`` 传入，会作为关键字参数转给策略类；
- 需要跨标的操作时读 ``self.datas``；**读价格前先查 ``self.broker.tradability_mask``**——
  停牌日的 ``close[0]`` 是**陈旧价**，不查会基于几个月前的价格下单（见引擎文档）。
"""

from __future__ import annotations

import backtrader as bt
import pandas as pd


class BuyAndHold(bt.Strategy):
    """第一次有机会就买满，其后不动。

    参数:
        size: 每次下单的股数。默认 100（一手）。
    """

    params = (("size", 100),)

    def next(self):
        for data in self.datas:
            if self.getposition(data).size:
                continue
            # 停牌日的 K 线是陈旧的，`close[0]` 因此不可信——先问掩码。
            #
            # 注意要转成 Timestamp：掩码的索引是 DatetimeIndex，直接拿 `datetime.date`
            # 去 `.at[]` 取不到，会抛 KeyError。
            today = pd.Timestamp(self.data0.datetime.date(0))
            mask = self.broker.tradability_mask
            if mask is not None and not mask.at[today, data._name]:
                continue
            self.buy(data=data, size=self.p.size)
