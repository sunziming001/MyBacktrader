"""库里的策略：**选股规则管买哪些，策略管何时买、何时卖**（ADR-0001）。

本模块放的是那些「每个多标的策略都要用」的东西，以及本项目顺手带上的几条现成策略。它们进库
而不是留在 ``examples/``，理由是**测试要能直接 import 它们**，且 ``mbt backtest --strategy``
是指名它们的地方——放在示例目录里，两件事都别扭（``examples/`` 不是包的一部分）。

写自己的策略时：

- 类继承 ``backtrader.Strategy``，并**带上** :class:`EngineClock`——多标的回测里取「今天」
  比看上去难，见它的说明；
- 参数用 ``--param name=value`` 传入，会作为关键字参数转给策略类；
- 读价格前**必须**查 ``self.broker.tradability_mask``：停牌日是**陈旧价**，而**尚未上市**的
  标的给的是它**最后一根**的价格（即未来价）——两者都不报错，只让结论失真（ADR-0006）。
"""

from __future__ import annotations

import backtrader as bt
import numpy as np
import pandas as pd

#: **选股分数**（``mbt.screen.SCREEN_SCORE_FIELD``，``CONTEXT.md``）在信号里的字段名。
#:
#: 策略读它来定「先买谁」，**不要照着重算**——复算会漂移，而漂移了不会报错。
from mbt.screen import GREEN_BRICK_FIELD, SCREEN_SCORE_FIELD  # noqa: F401  对外再导出

#: 砖型策略读的那个信号：当天是不是**绿砖**（0.0 / 1.0）。名字由产出方定义（见
#: :data:`mbt.screen.GREEN_BRICK_FIELD`），这里只是让策略那一侧读起来短一点。
_GREEN_BRICK = GREEN_BRICK_FIELD


class EngineClock:
    """取**引擎主时钟**：各标的当前日期里的**最大者**。这就是「今天」。

    **不要用 ``self.data0.datetime.date(0)``。** 若第一个标的的历史比回测区间短（次新股、
    北交所早期、或 ``--limit`` 随手选中的任意一只），它的日期会**一直停在末根**，于是掩码
    查到的是**另一天**——不报错，只是结果错。引擎内部记账时踩过同一个坑（见
    ``mbt.backtest.engine._EngineClock``，本类与它同源）。

    **但这个值不必自己扫**：backtrader 每 tick 已经把它写进了**策略自己的** ``datetime`` 线
    ——runonce 走 ``Strategy._oncepost(dt)``（``dt`` 是 cerebro 取的「各标的下一根日期的最小
    者」，而它已把所有 ``advance_peek() <= dt`` 的标的推进过，推完之后那个最小者恰好就是各
    标的当前日期的最大者），runnext 走 ``Strategy._clk_update``（直接写
    ``max(d.datetime[0] for d in self.datas if len(d))``）。两条路径都在调用 ``next()`` 之前
    把值写好，故读 ``self.datetime.date(0)`` 与扫一遍 ``self.datas`` 是同一个结果。

    差别只在代价上：扫一遍是每 tick × 每标的，全市场是 4752 × 2701 = 1280 万次
    ``len()`` + ``date()``。**注意是 ``self.datetime``（策略自己的时钟线），不是
    ``self.data0.datetime``**——后者正是上面那个坑。
    """

    def today(self) -> pd.Timestamp:
        return pd.Timestamp(self.datetime.date(0))


def signal_value(signals, field, today, name) -> float:
    """从 ``broker.signals`` 取一个标量信号；缺失、越界、或没给信号时一律返回 ``NaN``。

    **一律返回 NaN 是刻意的**：``NaN`` 的比较全部为假，故「不知道」天然表现为「不动作」。
    若这里改成抛异常或返回 0，调用方就得自己分辨「没数据」与「数据是 0」，而漏掉一处就会
    在无数据的日子凭空下单或凭空卖出。
    """
    if signals is None:
        return float("nan")
    frame = signals.get(field) if hasattr(signals, "get") else None
    if frame is None or name not in frame.columns:
        return float("nan")
    if today not in frame.index:
        return float("nan")
    return float(frame.at[today, name])


class Brick(bt.Strategy, EngineClock):
    """**砖型**：前一天绿砖、当天红砖且红砖更大时买入；持有期间当天是**绿砖**就卖出。

    **买入条件不在这里**：它是 :func:`mbt.screen.brick_screen` 那三条过滤器的组合，由引擎经
    ``broker.selection_mask`` 交进来（ADR-0001：同一条件不必写两遍）。本类只管**买多少、
    先买谁、何时卖**——那三件都依赖持仓与资金，是**路径依赖**的，而选股规则按定义不管持仓。

    成交时点沿用引擎的既有口径：订单在**本根收盘**下定、在**次一根开盘**成交。故

    - 本根收盘算出名单 → 次根开盘买入；
    - 本根收盘读到绿砖 → 次根开盘卖出。

    参数:
        green_field: 读「当天是不是绿砖」的那个信号字段名。默认与
            :func:`mbt.screen.brick_signals` 产出的那个一致。

    **按名次填空着的名额**：候选按**选股分数**（``CONTEXT.md``）**从高到低**依次提交买单，故
    名额或资金不够时先到的（即分数最高的）占住它们。那个分数由引擎从规则那边**原样**交过来
    （见 :data:`~mbt.screen.SCREEN_SCORE_FIELD`），本类**不重算**。

    **已持有的不加仓**：与 :class:`~examples.strategies.B1`、``CloseCrossesAboveMA`` 同一处置。

    **卖出的判据是状态，不是事件**：「今天是不是绿砖」而不是「今天是不是刚从红转绿」。两者在
    持有期里分岔很多（绿砖常连出几根），而需求说的是前者。同一标的**已有挂单时不再挂**，
    故跌停卖不出去时那笔单会继续等；而它若因停牌太久被作废，``notify_order`` 会把标记清掉，
    下一次读到绿砖时**重新挂**——否则那笔持仓会被永久卡住。

    .. warning::

        **没有止损、也没有持有封顶**。这是需求方的设定（那张票的验收里写着「无止损、无时间
        封顶」），不是漏实现：砖型只有一条卖出规则。代价是**一只买入后再也不转绿的标的会被
        无限期持有**，回撤不设上界——读这份回测结果时要把这一条算进去。

    前低、白线、黄线那些**一概不读**：砖型这条线自带它的买卖点，加别的判据就不是它了。
    """

    params = (("green_field", _GREEN_BRICK),)

    def __init__(self):
        #: 已有卖单在飞的标的——避免同一根 K 线上重复挂单（见类文档）。
        self._exiting: set[str] = set()
        #: 「今天该遍历谁」的索引，第一次 ``next()`` 时才建（那时 ``self.datas`` 才齐）。
        self._data_by_name: dict[str, object] = {}
        self._candidates_by_day: dict[object, tuple[str, ...]] = {}
        self._indexed = False

    def next(self):
        today = self.today()
        if not self._indexed:
            self._build_indices()
        # 先卖后买。顺序**固定**是为了让同一份输入永远给出同一份产物；它不影响现金——
        # 挂单中的卖单不释放名额也不释放现金（见 `mbt.backtest.sizing`）。
        self._exit_on_green(today)
        self._enter_candidates(today)

    # --- 索引 -----------------------------------------------------------------

    def _build_indices(self) -> None:
        """建好「标的 → data」与「交易日 → 当日候选（按名次排好）」两张表。

        按行扫一遍是**一次性**开销，而每根现查一次 ``.at`` 才是要避免的那种按标的数放大的
        开销。排序也在这里做完——它逐日只排当天的候选（个位数），不必每根重排。
        """
        self._data_by_name = {data._name: data for data in self.datas}
        self._candidates_by_day = self._index_candidates()
        self._indexed = True

    def _index_candidates(self) -> dict[object, tuple[str, ...]]:
        """把选股掩码与选股分数转成「交易日 → 当日候选，**从优到劣**」。

        先要求选股表**覆盖**可交易表的每一天：查不到当天时 ``.get`` 会退化成「当日无人入选」，
        而那会**静默地一根都不建仓**——与「当天确实没有候选」长得一模一样。故这里把不一致
        当场变成异常（与 ``B1`` 同一处判据）。
        """
        selection = self.broker.selection_mask
        if selection is None:
            return {}

        tradability = self.broker.tradability_mask
        missing = tradability.index.difference(selection.index)
        if len(missing):
            raise ValueError(
                f"选股表缺少 {len(missing)} 个交易日（首缺 {missing[0]}）："
                "那几天会静默地一根都不建仓。选股表与可交易表必须按同一个交易日并集建。"
            )

        scores = None
        if self.broker.signals is not None:
            scores = self.broker.signals.get(SCREEN_SCORE_FIELD)

        names = selection.columns.to_numpy()
        flags = selection.to_numpy(dtype=bool)
        indexed: dict[object, tuple[str, ...]] = {}
        for row, day in enumerate(selection.index):
            picked = list(names[np.flatnonzero(flags[row])])
            if scores is not None and day in scores.index:
                row_scores = scores.loc[day]
                # 缺失的名次记**最差**（排到最后），而不是排除它们——是否合格已由选股掩码
                # 回答，这里只回答「先买谁」。
                picked.sort(
                    key=lambda name: (
                        -(row_scores[name] if row_scores[name] == row_scores[name] else -np.inf),
                        name,
                    )
                )
            else:
                picked.sort()
            indexed[day] = tuple(picked)
        return indexed

    # --- 买入 -----------------------------------------------------------------

    def _enter_candidates(self, today) -> None:
        """按名次依次提交买单；名额或资金不够时由 sizer 与撮合层处置，本类不必数名额。"""
        for name in self._candidates_by_day.get(today, ()):
            data = self._data_by_name[name]
            if self.getposition(data).size:
                continue  # 已持有的不加仓
            if not self.broker.tradability_mask.at[today, name]:
                continue  # 停牌或尚未上市：这根 K 线是陈旧价/未来价
            # 不传 size → 交给 sizer（砖型的每日脚本用 FixedAmountSizer：每笔固定金额）。
            self.buy(data=data)

    # --- 卖出 -----------------------------------------------------------------

    def _exit_on_green(self, today) -> None:
        """持有期间当天是绿砖 → 清仓（成交落在次根开盘）。

        **不做可交易性判断**：卖出不受股票池与选股掩码限制，而「今天没有 K 线」时那个读数
        本身就是缺失（砖型图算不出来）——``green_brick`` 把缺失记成 **False**（判据是比大小，
        缺失处两者都不成立），故信号是 0.0 而不是 NaN，``0.0 > 0.5`` 为假、自然不动作。用不着
        再看一遍掩码。
        """
        for name, data in self._data_by_name.items():
            if name in self._exiting:
                continue  # 已有卖单在飞：跌停卖不出去时那笔单要继续等
            if not self.getposition(data).size:
                self._exiting.discard(name)
                continue
            if signal_value(self.broker.signals, self.p.green_field, today, name) > 0.5:
                self._exiting.add(name)
                self.close(data=data)

    def notify_order(self, order) -> None:
        """卖单**终结**后清掉标记，好让下一次绿砖能重新挂单。

        「终结」包括成交、被拒、被撤、以及**挂单失效**（停牌太久）。最后那一种是关键：不清
        标记的话，一笔因停牌而作废的卖单会把那只标的**永久**卡住——它既不会再挂单，也永远
        不会卖。
        """
        if order.alive() or not order.issell():
            return
        data = order.data
        if data is not None:
            self._exiting.discard(data._name)
