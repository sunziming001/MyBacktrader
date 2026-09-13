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

from mbt.screen import Screen


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


class UndervaluedGrowth(bt.Strategy, EngineClock):
    """**低估成长**：估值便宜 + 深度回撤时买入，估值恢复、增长放缓或转亏时卖出。

    **买入条件不在这里**：它是 :func:`mbt.screen.undervalued_growth_screen`，由引擎经
    ``broker.selection_mask`` 交进来。七个条件：

    1. 动态PE 百分位 < 8%（相对自身历史足够便宜）；
    2. 动态PE > 0（剔除亏损）；
    3. 0 < PEG < 0.75（增长为正、且价格相对增长仍算便宜）；
    4. **一年内跌幅 > 42%**（自近一年的最高价回落足够深）；
    5. 总市值 > 100 亿（剔除小盘股）；
    6. ROE > 10%（盈利质量门槛）；
    7. **净资产为正**（剔除净资产为负的困境公司）。

    第 7 条在默认参数下**是冗余的**：净资产 < 0 且 ROE > 10 意味着净利TTM < 0（ROE 的分子
    分母同时为负时商为正），于是 PE = 市值 ÷ 净利TTM 也为负，已被第 2 条拦掉。全市场核验里
    「净资产 ≤ 0」的 29,914 格中有 98% 的 ROE > 10，却**没有一个** PE > 0，反例为 0 格；
    加过滤前后候选集都是 8,602 格。它只在 ``pe_above < 0``（放行亏损公司）时才真正起作用，
    保留它是为了让「净资产为正」是一个**被声明的前提**，见 :data:`mbt.screen.MIN_EQUITY`。

    排序因子是 **ROE**，故候选取「折价 + 深跌 + 高质量」里质量最高的那几只。

    这样 ``mbt screen`` 与 ``mbt backtest`` 用的是**同一个对象**，条件不必写两遍（ADR-0001）。

    本类只管**卖出**——那三条都依赖「已经持有」与当日估值，是**路径依赖**的，而筛选规则按
    定义不管持仓（见 :class:`~mbt.screen.Screen`）。

    参数:
        percentile_exit: 动态PE 百分位高于它就卖出。默认 0.70。
        peg_exit: PEG 高于它就卖出。默认 1.1。

    三条卖出条件（任一满足）:

    1. 动态PE **≤ 0** —— 公司转亏，估值失去意义。与「买入要求 PE > 0」对称；
    2. 百分位 > ``percentile_exit`` —— 已不再便宜；
    3. PEG > ``peg_exit`` —— 价格相对增长已偏贵。

    PEG 的增长率是**归母净利当期累计同比**，PE 用的是**总股本 × 收盘价 ÷ 归母净利TTM**——
    与通达信的市盈率公式逐项对齐（见 :mod:`mbt.data.valuation`）。这一项对结果影响极大：
    实测 ``sz002692`` 在 2025-09-02，通达信口径给出 PE 百分位 **91.82%**、PEG **−36.57**
    （不该买），而项目早先自创的「年化 EPS + 滞后同比」口径给出 **2.55%** 与 **+0.42**——
    **会把它买进来**。这类假阳性正是本策略拒绝自创口径的原因。

    .. warning::

        **「排除 ST」在本项目里是一条空条件**：本地数据没有股票名称，出厂规则表刻意不登记
        任何 ``st_period``，故 ST 股不会被排除。实测 ``sz002731``（2025-05-07 建仓）就是 ST，
        而且引擎还按 **10%** 的限幅在给它撮合（真实是 5%）——那意味着它的一字板会被当成可成交。
        详见 ADR-0002 与 :mod:`mbt.rules`。

    .. warning::

        **这两个阈值没有经过论证，且实测无法用本地样本调优。** 做过一次出场门槛扫描
        （0.70 / 0.50 / 0.30 / 0.20，同区间、同其它参数，**只换标的抽样**，各 300 只）：

        ============  ==========  ==========
        出场门槛       抽样 A       抽样 B
        ============  ==========  ==========
        0.70            +4.18%      +5.65%
        0.30           +11.33%      +2.34%
        0.20           +12.18%      +5.71%
        ============  ==========  ==========

        抽样 A 上「降低门槛把年化从 4% 提到 12%」；而抽样 B 上 0.30 反而**最差**，0.70 与
        0.20 几乎相同——**排序翻转**。更关键的是量级：同一参数在两份抽样间的差异
        （4.18% vs 5.65%）与不同参数间的差异**同量级**，故参数效应与抽样噪声分不开。

        11 年只有 40~67 笔平仓，样本量本身就不足以支撑调参。唯一在两边都稳定的是**回撤**
        （42%~53%，与门槛几乎无关）；出场门槛能机械地缩短持仓（中位 359→257 天、254→184
        天），但收益方向不稳。

        故这两个阈值是**可用的起点**，不是**调优过的参数**。要用它们下判断，先把样本量问题
        解决掉（全市场 + 走查验证），否则调出来的只是噪声——而那正是本项目反复防的
        「虚假信心」。

    .. warning::

        **分年份走查的结论更重：这份回测结果由一个年份撑起来。**

        逐年收益（300 只抽样、2015-08 起）::

            2015  +11.28%      2021   +3.52%
            2016  -13.39%      2022  -17.71%
            2017   -0.57%      2023  +11.22%
            2018  -30.97%      2024  +30.87%
            2019  +85.79%      2025   +7.40%
            2020  +27.96%      2026  -32.15%（年内至今）
            12 年里 5 年亏损

        - **2019 一年占「各年正收益之和」的 48%**；
        - 剔掉 2019：其余 11 年累计 **−16.52%**（含它是 +55.10%）；
        - 换起点：2018 起 +61.77%、2020 起 +29.51%，而 **2022 起 −11.39% 且跑输基准
          3.10 个百分点**。

        故「+55%、年化 4.18%」**不能读作策略优势**——它主要来自 2019 那一波，而 2022 年以后
        是负的。要把这份数字当作证据，至少需要：全市场（而非 300 只抽样）、剔除单一牛年后
        仍成立、以及 2022 年以后也成立。三条都不满足。

        .. note::

            上面那张逐年表与「剔除 2019」的数字取自**早先的自创口径**（年化 EPS + 滞后同比）。
            改成通达信口径后（见 :mod:`mbt.data.valuation`）重跑 2018 起：年化 **+12.68%**、
            2019 **+110.36%**、剔除 2019 后 **+29.14%**（**转正**）、2022 起跑 **+62.67%**。
            「靠某一年」的结构**没变**（2019 仍占各年正收益之和的一半），但比早先那版站得住
            一些。
    """

    params = (("percentile_exit", 0.70), ("peg_exit", 1.1))

    def next(self):
        today = self.today()
        signals = self.broker.signals

        for data in self.datas:
            name = data._name
            if self.getposition(data).size:
                if self._should_exit(signals, today, name):
                    self.close(data=data)
                continue

            # 停牌或尚未上市时这根 K 线是**陈旧价/未来价**，先问掩码。
            if not self.broker.tradability_mask.at[today, name]:
                continue
            # 买入条件来自选股规则算好的候选集，本类不再重复一遍。
            selection = self.broker.selection_mask
            if selection is None or not selection.at[today, name]:
                continue
            self.buy(data=data)

    def _should_exit(self, signals, today, name) -> bool:
        """三条卖出条件任一满足即真。缺信号时**三条件全假**（见 :func:`signal_value`）。"""
        pe = signal_value(signals, "pe", today, name)
        percentile = signal_value(signals, "pe_percentile", today, name)
        peg = signal_value(signals, "peg", today, name)
        return bool(pe <= 0 or percentile > self.p.percentile_exit or peg > self.p.peg_exit)


class B1(bt.Strategy, EngineClock):
    """**回调买点 + 三条卖出规则**：白线/黄线之上、J 值偏低、回调缩量时买入。

    **买入条件不在这里**：它是 :func:`b1_signals` 那几个过滤器组的
    :class:`~mbt.screen.Screen`，由引擎经 ``broker.selection_mask`` 交进来。本类只管
    **持有与卖出**——三条卖出规则全都依赖「已经持有」与「建仓时的价格」，是**路径依赖**的，
    而选股规则按定义不管持仓。

    三条卖出规则（任一满足即动作）:

    ==========  ==========================================  ================
    规则        判据                                        动作
    ==========  ==========================================  ================
    止损        ``stop_streak``：连续 ``stop_days`` 根收盘    清仓
                低于黄线，**或**收盘跌破建仓时锚定的前低
    分批止盈    ``trim``：收盘 ≥ 白线 × (1 + ``trim_margin``)  卖出一半
    趋势离场    ``below_white``：曾收在白线上（armed），      清仓
                其后收盘跌破白线
    ==========  ==========================================  ================

    **两条状态一律在「首次见到这个持仓」的那一根建立**，而不是在下单那一根：

    - ``armed``（是否曾收在白线上）与 ``trimmed``（是否已减过仓）在建仓后从 **False**
      起算——建仓那一根本身**不**武装趋势离场。理由是多一个选择就要多一处解释：把建仓日
      也算进去，会让「买在白线下方、次日反抽到白线上方、第三日又跌破」这种最常见的形态
      在第三天就被清掉，而买点本来就允许收盘贴近白线（实测九个样本里七成的买入价在白线
      **下方**）。故趋势离场至少要等一次明确的站上。
    - 前低止损位在建仓后**只取一次**。若每根都重算，「自峰值以来的最低价」会随价格下跌
      一起下移，判据永远不成立——那不是止损，是跟随。这也是它必须留在策略层的原因：
      信号层没有「持仓」这个概念。

    参数:
        white_n: 白线的双重 EMA 窗口。默认 10。
        yellow_windows: 黄线各条均线的窗口。默认 ``(14, 28, 57, 114)``。
        stop_days: 「连续跌破黄线」的根数。默认 2。
        stop_buffer: 前低止损的缓冲比例（跌破前低 × (1 − 它) 才触发）。默认 0.01。
        trim_margin: 分批止盈相对白线的比例门槛。默认 0.08。

    **前低取「峰值之后」的那一段**（调整期的前低），不是「峰值之前」的波段起点。两者
    位置差得很远：实测九个样本上，取波段起点给出的止损距中位 **−36.9%**（最松 −70.5%），
    那样的止损形同虚设；取调整期前低是 **−2.7%**（−1.1% ~ −20.0%）。同一批样本上两者在
    60 根内都未被触发，故这条止损**无法**用这批样本验证——它们全是买在局部底部的成功案例。

    .. warning::

        实测九个样本上，趋势离场会**明显早于**趋势走完就出场（平均持有 23 根，而原口径的
        「连续跌破黄线」是 40~127 根）。最刺眼的一例是 ``sz002074``：持有 7 根、+1.5% 出场，
        而持有到第 60 根是 +52.3%——它买在白线**下方 4.9%**，反抽站上白线之后第一次回落就
        被清掉。单次一个收盘价低于白线就清仓，对「在白线附近震荡」的形态非常敏感。

        这**不**说明这套卖出规则不好——这九个样本全是上涨的，任何「早点卖」的规则都必然
        少赚。要判断它有没有价值，只能看**失败样本**（跌幅不深、长期横盘的那些），而那批
        样本目前一个都没有。
    """

    params = (
        ("white_n", 10),
        ("yellow_windows", (14, 28, 57, 114)),
        ("stop_days", 2),
        ("stop_buffer", 0.01),
        ("trim_margin", 0.08),
    )

    def __init__(self):
        # 持仓期的路径依赖状态：**按标的**各持一份，且只在有持仓时有效。
        self._armed: set[str] = set()
        self._trimmed: set[str] = set()
        self._stop_price: dict[str, float] = {}

    def next(self):
        today = self.today()
        signals = self.broker.signals

        for data in self.datas:
            name = data._name
            position = self.getposition(data)

            if not position.size:
                # 空仓即清状态：下一次建仓必须从「未武装、未减仓、前低重取」开始，
                # 否则上一笔的武装状态会漏到下一笔上，趋势离场会凭空提前触发。
                self._armed.discard(name)
                self._trimmed.discard(name)
                self._stop_price.pop(name, None)
                self._enter_if_selected(data, name, today)
                continue

            if name not in self._stop_price:
                # 首次见到这个持仓——上一根下的单成交了，此刻才是「建仓时」。
                self._armed.discard(name)
                self._trimmed.discard(name)
                self._stop_price[name] = self._anchored_prior_low(data, signals, today, name)

            self._manage_exit(data, name, position, signals, today)

    # --- 买入 -----------------------------------------------------------------

    def _enter_if_selected(self, data, name, today) -> None:
        """买入条件来自选股规则算好的候选集，本类不再重复一遍（ADR-0001）。"""
        if not self.broker.tradability_mask.at[today, name]:
            return  # 停牌或尚未上市：这根 K 线是陈旧价/未来价
        selection = self.broker.selection_mask
        if selection is None or not selection.at[today, name]:
            return
        # 不传 size → 交给引擎的等权 sizer（`EqualWeightSizer`）。
        self.buy(data=data)

    # --- 卖出 -----------------------------------------------------------------

    def _manage_exit(self, data, name, position, signals, today) -> None:
        close = data.close[0]

        # 1) 止损优先：同一根上若既触发止损又触发减仓，先清仓与「减半再清」等价，
        #    而先判止损不会留下半仓。
        stop_level = self._stop_price.get(name)
        stop_hit = signal_value(signals, "stop_streak", today, name) > 0.5
        if stop_hit or (stop_level == stop_level and close < stop_level):
            self.close(data=data)
            return

        # 2) 趋势离场：曾收在白线上，其后跌破白线。
        above = signal_value(signals, "above_white", today, name)
        below = signal_value(signals, "below_white", today, name)
        if above > 0.5:
            self._armed.add(name)
        if name in self._armed and below > 0.5:
            self.close(data=data)
            return

        # 3) 分批止盈：只做一次；买点允许收盘贴近白线，故它常常在趋势离场之前就触发。
        if name not in self._trimmed and signal_value(signals, "trim", today, name) > 0.5:
            half = int(position.size // 2)
            if half > 0:
                # **未按「一手 = 100 股」整手取整**（README 已记的乐观偏差）；卖出一半后
                # 余下的股数同样可能不是整手。
                self._trimmed.add(name)
                self.close(data=data, size=half)

    def _anchored_prior_low(self, data, signals, today, name) -> float:
        """建仓时锚定「调整期的前低」= 峰值之后到建仓日之间的最低价。

        峰位由 ``peak_age`` 给出（自峰值起经过的根数），故这一段是**含当根**的 ``age + 1``
        根。``peak_age`` 缺失（拐点未确认）时不设这条止损——缺失取「不设」而不是取 0，
        与信号层「缺失即不合格」同一精神：不猜。
        """
        age = signal_value(signals, "peak_age", today, name)
        if age != age:  # NaN
            return float("nan")
        bars = int(age) + 1
        if bars < 1 or bars > len(data):
            return float("nan")
        lowest = min(data.low[-k] for k in range(bars))
        return float(lowest) * (1.0 - self.p.stop_buffer)


def b1_signals(
    markets,
    *,
    align_to=None,
    white_n=10,
    yellow_windows=(14, 28, 57, 114),
    stop_days=2,
    trim_margin=0.08,
    retracement=0.08,
) -> dict[str, pd.DataFrame]:
    """算出卖点与止损要读的信号（**标的宽表**），供 ``run_portfolio_backtest(signals=...)``。

    在**完整历史**上算，再截到回测区间——这正是「先算后截」的口径
    （:func:`mbt.data.panel.clip_fields` 的说明）。若先截再算，摆动点与黄线都会因为起点
    不同而给出不同的值，于是同一段行情在两个区间里结论不同。

    参数:
        markets: 一组已质检行情。**必须是完整历史**（``load_market_data`` 的返回值），
            不是切过窗口的那些——否则「先算后截」就反了。
        align_to: 一组行情；给了就把结果的**标的集合与交易日范围**对齐到它。
            **回测时必须给**，且必须传**切过窗口的**那一组行情：引擎要求信号与行情面板
            逐日同列对齐，而完整历史的信号比切过的行情长、也可能多出被跳过的标的，
            不对齐就会以「index / columns 不一致」报错。

            刻意接受一个「行情组」而不是 ``(start, end)`` 两个值：那两个值若与实际切片
            不一致，结果仍是错的，而**行情的边界就在行情里**——从它推出来不会漂移。

    返回的五个字段都是**浮点**（布尔信号转成 0.0/1.0）：它们的消费方是策略侧
    :func:`signal_value`，读的是标量，而面板字段保持数值型可以省掉一处类型分支。

    ==================  ====================================================
    字段                含义
    ==================  ====================================================
    ``above_white``     收盘严格高于白线（0/1）
    ``below_white``     收盘严格低于白线（0/1）
    ``trim``            收盘 ≥ 白线 × (1 + ``trim_margin``)（0/1）
    ``stop_streak``     连续 ``stop_days`` 根收盘低于黄线（0/1）
    ``peak_age``        自最近一段已完成上涨的峰值起经过的根数（缺失=拐点未确认）
    ==================  ====================================================
    """
    from mbt.data.panel import clip_fields
    from mbt.signals import above_white, below_white, below_yellow_streak, swings

    closes = {market.symbol: market.prices["close"] for market in markets}
    close_frame = pd.DataFrame(closes)

    def as_float(frame):
        return frame.astype(float)

    anchors = swings(close_frame, retracement=retracement)
    fields = {
        "above_white": as_float(above_white(close_frame, white_n, 0.0)),
        "below_white": as_float(below_white(close_frame, white_n)),
        "trim": as_float(above_white(close_frame, white_n, trim_margin)),
        "stop_streak": as_float(
            below_yellow_streak(close_frame, white_n, tuple(yellow_windows), stop_days)
        ),
        "peak_age": anchors.peak_age.astype(float),
    }
    if align_to is None:
        return fields

    bounds = [market.prices.index for market in align_to]
    lower = min(index.min() for index in bounds)
    upper = max(index.max() for index in bounds)
    return clip_fields(fields, align_to, start=lower, end=upper)


def b1_screen(
    *,
    white_n=10,
    yellow_windows=(14, 28, 57, 114),
    kdj_n=9,
    kdj_m1=3,
    kdj_m2=3,
    j_max=20.0,
    retracement=0.08,
    min_advance=0.10,
    min_drop=0.08,
    max_drop=0.35,
    min_peak_age=5,
    max_peak_age=50,
    volume_edge_bars=3,
    volume_base_bars=10,
    min_surge=2.0,
    max_pullback=0.5,
    top_n=None,
) -> Screen:
    """**B1 策略**的选股规则：趋势 + 位置 + J 值 + 量能（四条过滤），**黄线贴近度**排序。

    它是 ``Screen``，故同一条件既能用于 ``mbt screen`` 选股，也能作为回测的入场闸门——
    不必写两遍（ADR-0001）。

    ==================  ================================================
    过滤器              判据
    ==================  ================================================
    趋势                白线在黄线上（提示 2）
    价格在慢线上        收盘 > 黄线（B1 追加）
    位置                「一波上涨之后的下跌阶段」（提示 3）
    J 值                KDJ 的 J 偏低（提示 3）
    量能                上涨放量 + 回调缩量（提示 4）
    ==================  ================================================

    「价格在慢线上」这一条与「趋势」不是同义反复：前者比较**价格**与黄线，后者比较
    **白线**与黄线。价格可以从上方双双跌破，那时白线仍在黄线上（慢线还没跟上）——实测
    ``sh688321`` 在 2025-06-20 就是这样。追加它是为了解决一件具体的事：止损要求「连续 2 根
    收盘低于黄线」，若入场时价格已在黄线**下方**，那条止损几乎立刻触发（全市场回测里
    78% 的出场都是它，持有期中位仅 1 天）。

    排序因子是 :func:`~mbt.signals.factors.yellow_proximity`（``黄线 ÷ |收盘 − 黄线|``），
    **越大离黄线越近**，故买点取「回调到支撑上」的那几只，而不是「跌得最深」的那几只。
    因子本身不带方向（取了绝对值），「在黄线哪一侧」由过滤条件管。

    参数:
        top_n: 只取前 N 名。``None``（默认）表示**不截断**——那正是「本金不限」的语义：
            每个合格标的都买一份。给了它就只买离黄线最近的那 N 只。

    各阈值的取法见 ``CONTEXT.md`` 与 ``.scratch`` 下的实测记录；默认值是九个样本上的
    **可执行起点**，不是调优过的参数。特别地：

    - ``retracement``：实测只有 **0.08** 让九个样本全部呈现「涨过一段 + 正在回调」
      （0.04~0.06 与 0.10 都是 7/9，0.15 是 5/9）。
    - ``j_max``：九个样本的信号日 J 落在 −13.4 ~ 19.8（中位 −4.8），故 20 覆盖 9/9。
    - ``min_surge=2.0`` / ``max_pullback=0.5``：量能两条，9 个样本上命中 8/9
      （只有 ``sz301076`` 不通过，因为摆动点给它切出的是一段 4 根、14% 的短促上冲）。

    .. warning::

        「位置」条件**几乎不筛人**：全历史实测，它在 35%~52% 的交易日成立。真正的选择性
        来自 J 值与量能，故不要指望它单独构成一个稀疏的候选集。
    """
    from mbt.signals import (
        above_yellow,
        j_below,
        pullback_after_advance,
        swings,
        volume_contraction,
        white_above_yellow,
        yellow_proximity,
    )

    windows = tuple(yellow_windows)

    def trend(panel):
        return white_above_yellow(panel["close"], white_n, windows)

    def not_below_the_line(panel):
        return above_yellow(panel["close"], windows)

    def position(panel):
        return pullback_after_advance(
            panel["close"],
            retracement=retracement,
            min_advance=min_advance,
            min_drop=min_drop,
            max_drop=max_drop,
            min_peak_age=min_peak_age,
            max_peak_age=max_peak_age,
        )

    def low_j(panel):
        return j_below(panel, j_max, kdj_n, kdj_m1, kdj_m2)

    def volume(panel):
        return volume_contraction(
            panel["volume"],
            swings(panel["close"], retracement=retracement),
            edge_bars=volume_edge_bars,
            base_bars=volume_base_bars,
            min_surge=min_surge,
            max_pullback=max_pullback,
        )

    def proximity(panel):
        return yellow_proximity(panel["close"], windows)

    return Screen(
        filters=(trend, not_below_the_line, position, low_j, volume),
        factor=proximity,
        top_n=top_n,
    )


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
