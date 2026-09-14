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
import numpy as np
import pandas as pd

from mbt.screen import Screen


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

    .. note::

        这个类其实是**库该提供的东西**——每个多标的策略都需要它，抄在示例里意味着每人都得
        重新推导一次（而第一个版本就写错了）。把它提升进 ``mbt`` 是对的方向，但那是一次
        库 API 变更，故此处先按示例给出。
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
    最高价破白线 ``high_above_white``：当日最高价 > 白线       清仓
    两日未站上   建仓后第 ``confirm_white_days`` 根收盘仍在    清仓
                白线**下方**
    ==========  ==========================================  ================

    **成交时点**：订单在本根收盘下定、在**下一根开盘**成交，且 T+1 使当日买入不可当日卖出。
    故上面三条的判据都读**当根**的收盘价／最高价，而成交价是**下一根的开盘价**。规则若被读成
    「以当日收盘价卖出」，那是本引擎做不到的一项——本项目的撮合一律在次根开盘，没有「按当根
    收盘成交」的口径（`mbt.backtest.costs` 的 `_execute` 只接受引擎给的成交价，而引擎按
    `coc=False` 运行）。这处偏差的方向不定，故不假装它不存在。

    **买入侧的掩码按「下单那一根」判**（`AStockBroker._mask_says(at_creation=True)`），
    故「T 入选、T+1 已不入选」的订单**照样成交**。这一点在 B1 上不是细节：J 阈值收紧到 8 之后
    入选格很稀疏，实测有 86.3% 的入选格只连续成立一天——若按成交那根复查，这批信号会全部
    被拒（拒单理由写作「未被选股规则选中」），入场条件就变成了「入选且次日仍入选」。

    **「最高价破白线」在建仓当根就成立时**，会自然落到「第二日开盘卖出」：本根收盘下卖单，
    它要在下一根开盘成交；而 T+1 又不允许在本根卖出当日买入的股份。两处约束指向同一个落点，
    故不需要为这种情形单写一条分支。

    **``armed`` 这个状态随「最高价破白线」这条规则一并去掉了。** 旧版的趋势离场要求「先
    站上白线、其后跌破」，故必须记「是否武装过」；新版是「摸到白线就卖」，不需要记忆。
    同理，分批止盈（卖一半）也被去掉，故不再有 ``trimmed`` 与「只减一次」的保护。

    参数:
        white_n: 白线的双重 EMA 窗口。默认 10。
        yellow_windows: 黄线各条均线的窗口。默认 ``(14, 28, 57, 114)``。
        stop_days: 「连续跌破黄线」的根数。默认 2。
        stop_buffer: 前低止损的缓冲比例（跌破前低 × (1 − 它) 才触发）。默认 0.01。
        confirm_white_days: 建仓后第几根仍未站上白线就离场，数的是**交易日**。默认 2，
            即 T+2。根数只在**当日可交易**时才递增，故停牌不会把它提前或推后。

    **前低取「峰值之后」的那一段**（调整期的前低），不是「峰值之前」的波段起点。两者
    位置差得很远：实测九个样本上，取波段起点给出的止损距中位 **−36.9%**（最松 −70.5%），
    那样的止损形同虚设；取调整期前低是 **−2.7%**（−1.1% ~ −20.0%）。同一批样本上两者在
    60 根内都未被触发，故这条止损**无法**用这批样本验证——它们全是买在局部底部的成功案例。

    **前低在建仓后只取一次**。若每根都重算，「自峰值以来的最低价」会随价格下跌一起下移，
    判据永远不成立——那不是止损，是跟随。这也是它必须留在策略层的原因：信号层没有「持仓」
    这个概念。

    .. warning::

        新版「最高价破白线就卖」比旧的趋势离场**更早**离场：旧的要先收在白线上、再等跌破，
        新的只要盘中摸到白线。故持有期会进一步缩短，而「摸到白线」在回调后的第一次反弹里
        很常见。这一条是不是把趋势掐得太短，只能由全市场结果回答——本文件不再预判。

        旧版的实测可作为量级参考：九个上涨样本上平均持有 23 根；同一批样本上一旦改成
        「摸到就卖」，持有期只会更短。
    """

    params = (
        ("white_n", 10),
        ("yellow_windows", (14, 28, 57, 114)),
        ("stop_days", 2),
        ("stop_buffer", 0.01),
        ("confirm_white_days", 2),
    )

    def __init__(self):
        # 持仓期的路径依赖状态：**按标的**各持一份，且只在有持仓时有效。
        self._held_bars: dict[str, int] = {}
        self._stop_price: dict[str, float] = {}
        # 下面几份是「本根该遍历谁」的索引，第一次 ``next()`` 时才建（那时 ``self.datas`` 才齐）。
        self._data_by_name: dict[str, object] = {}
        self._order_of_name: dict[str, int] = {}
        self._selected_by_day: dict[object, tuple[str, ...]] = {}
        self._indexed = False

    def next(self):
        today = self.today()
        signals = self.broker.signals

        for data, name in self._candidates(today):
            position = self.getposition(data)

            if not position.size:
                # 空仓即清状态：下一次建仓必须从「第 0 根、前低重取」开始，
                # 否则上一笔的持有根数会漏到下一笔上，T+2 那条会提前触发。
                self._held_bars.pop(name, None)
                self._stop_price.pop(name, None)
                self._enter_if_selected(data, name, today)
                continue

            if name not in self._held_bars:
                # 首次见到这个持仓——上一根下的单在本根开盘成交了，故**本根就是 T 日**。
                # 之后每过一根可交易日递增一次（见 `_manage_exit` 之后的计数）。
                self._held_bars[name] = 0
                self._stop_price[name] = self._anchored_prior_low(data, signals, today, name)

            self._manage_exit(data, name, position, signals, today)

            # 计数放在判据之后：`_manage_exit` 用到的「本根是第几根」必须是递增前的值。
            # 只在**当日可交易**时递增，故 T+2 数的是两个交易日，停牌不会把它推后。
            if self.broker.tradability_mask.at[today, name]:
                self._held_bars[name] += 1

    # --- 遍历范围 -------------------------------------------------------------

    def _candidates(self, today):
        """本根**可能有动作**的标的——把每根的遍历量从「全部标的」降到「少数几个」。

        只有两个来源，各自对应循环体里唯一的两类动作：

        - **有持仓** → 止损 / 趋势离场 / 分批止盈（:meth:`_manage_exit`）；
        - **当日候选** → 唯一可能**新建仓**的来源。

        其余标的为什么可以跳过：循环体对它们所做的全是空操作——三个 ``discard``/``pop``
        对不在集合里的键无效，``_enter_if_selected`` 查到「未被选中」立刻返回。全市场下这
        不是小账：4752 只 × 2701 根 = 1280 万次遍历，而每根真正有可能动作的标的只有个位数。
        实测（232 只标的）这一段占「逐根」阶段的六成。

        两处看着像遗漏、实测却**不必要**的候选来源，写在这里免得以后被「补」回去：

        - **刚清仓的标的**（``_stop_price`` 里还留着上一笔的锚定值）。它确实要被清理，但
          不必单独列一类：要在它身上再建仓只能靠 ``_enter_if_selected``，而那要求它当天在
          候选里——于是必然先走一遍空仓分支，清理就发生在那里。若它一直不再入选，那份陈旧
          状态也永远读不到（读它需要持仓，而持仓又回到前一句）。故「陈旧状态被读到」在路径
          上不可达。
        - **有挂单的标的**。挂单必然出自「空仓分支」里下的单，而那个分支已经清掉了状态；
          之后它唯一还能做的就是**再买一次**，那同样要求当天被选中。而成交不会漏：成交后
          持仓落在 ``broker.positions`` 上，每根都被上面第一个来源扫到。

        （两条都试过单独列一类，实测产出一模一样——多出来的遍历只换来一份读不到的状态。）

        产出按 ``self.datas`` 的原顺序：顺序本身不影响结果（每个标的各自独立，现金只在
        **成交**时变动，而下单不占用现金），但固定顺序让同一份输入永远给出同一份产物。
        """
        if not self._indexed:
            self._build_indices()

        seen: dict[str, object] = {}

        for data, position in self.broker.positions.items():
            if position.size:
                seen[data._name] = data

        for name in self._selected_by_day.get(today, ()):
            data = self._data_by_name.get(name)
            if data is not None:
                seen[name] = data

        return [(seen[name], name) for name in sorted(seen, key=self._order_of_name.__getitem__)]

    def _build_indices(self) -> None:
        """建好「标的 → data」「标的 → 原顺序」「交易日 → 当日候选」三张表。

        选股表按行扫一遍是**一次性**开销（2701 行），而每根现查一次 ``.at`` 才是要避免的
        那种按标的数放大的开销。
        """
        self._data_by_name = {data._name: data for data in self.datas}
        self._order_of_name = {data._name: i for i, data in enumerate(self.datas)}
        self._selected_by_day = self._index_selected_days()
        self._indexed = True

    def _index_selected_days(self) -> dict[object, tuple[str, ...]]:
        """把选股掩码转成「交易日 → 当日选中的标的」。

        先要求选股表**覆盖**可交易表的每一天：查不到当天时 ``.get`` 会退化成「当日无人入选」，
        而那会**静默地一根都不建仓**——与「当天确实没有候选」长得一模一样。故这里把不一致
        当场变成异常。
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

        names = selection.columns.to_numpy()
        flags = selection.to_numpy(dtype=bool)
        return {
            day: tuple(names[np.flatnonzero(flags[row])]) for row, day in enumerate(selection.index)
        }

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
        """三条卖出规则，任一满足即清仓。判据都读**当根**，成交落在**下一根开盘**。

        顺序：止损在最前——同一根上若既触发止损又触发离场，两者的动作相同（都是清仓），
        但先判止损能保证「止损优先」这件事在换规则时不被顺手改掉。``position`` 参数保留
        是为了与旧签名一致（曾经用它算「卖一半」的股数）。
        """
        _ = position
        close = data.close[0]

        # 1) 止损（口径未变）：连续 N 根收盘低于黄线，或跌破建仓时锚定的前低。
        stop_level = self._stop_price.get(name)
        stop_hit = signal_value(signals, "stop_streak", today, name) > 0.5
        if stop_hit or (stop_level == stop_level and close < stop_level):
            self.close(data=data)
            return

        # 2) 最高价破白线 → 清仓。建仓当根若已成立，卖单落在下一根开盘，正是「第二日开盘」。
        if signal_value(signals, "high_above_white", today, name) > 0.5:
            self.close(data=data)
            return

        # 3) T+confirm_white_days 收盘仍在白线下方 → 清仓。
        #    只在**当日可交易**时判，否则停牌那根会用陈旧收盘价把这条提前点着。
        if (
            self._held_bars.get(name, 0) >= self.p.confirm_white_days
            and (self.broker.tradability_mask.at[today, name])
        ):
            if signal_value(signals, "below_white", today, name) > 0.5:
                self.close(data=data)

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

    返回的四个字段都是**浮点**（布尔信号转成 0.0/1.0）：它们的消费方是策略侧
    :func:`signal_value`，读的是标量，而面板字段保持数值型可以省掉一处类型分支。

    ==================  ====================================================
    字段                含义
    ==================  ====================================================
    ``below_white``     收盘严格低于白线（0/1）——「T+2 仍在白线下」那条用
    ``high_above_white`` 当日最高价严格高于白线（0/1）——「最高价破白线」那条用
    ``stop_streak``     连续 ``stop_days`` 根收盘低于黄线（0/1）
    ``peak_age``        自最近一段已完成上涨的峰值起经过的根数（缺失=拐点未确认）
    ==================  ====================================================

    ``above_white`` 与 ``trim`` 两个字段随旧卖出规则一并去掉了：前者用于「武装」趋势离场，
    后者用于分批止盈，两条规则都已不存在。留着它们是死字段——而信号面板里多一个没人读的
    列，下一次改卖出规则时会先让人以为它还在用。
    """
    from mbt.data.panel import clip_fields
    from mbt.signals import (
        below_white,
        below_yellow_streak,
        high_above_white,
        swings,
    )

    closes = {market.symbol: market.prices["close"] for market in markets}
    highs = {market.symbol: market.prices["high"] for market in markets}
    close_frame = pd.DataFrame(closes)
    window_frame = {
        "close": close_frame,
        "high": pd.DataFrame(highs),
    }

    def as_float(frame):
        return frame.astype(float)

    anchors = swings(close_frame, retracement=retracement)
    fields = {
        "below_white": as_float(below_white(close_frame, white_n)),
        "high_above_white": as_float(high_above_white(window_frame, white_n)),
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


def _valuation_field(panel, name: str):
    """取一个**来自财务数据**的面板字段；缺了就说清该怎么办。

    估值不在行情里，故它只能由调用方经 ``signals=`` 并进面板。缺字段时 ``Panel`` 本身会抛
    ``KeyError``，但那句话只说「没有这个字段」，不够回答「那我该做什么」——而这是本规则最
    容易踩的一脚（行情全都对得上，唯独估值没接上）。
    """
    if name not in panel:
        raise ValueError(
            f"B1 的选股规则需要面板含 {name!r}，而它来自财务数据、不在行情里。"
            "调用方要先把 `mbt.data.valuation.valuation_for(...)` 的字段并进传给引擎的 "
            "`signals=`（见 `mbt.data.panel.with_signals`），或用 `mbt screen --cw-root` 取估值。"
        )
    return panel[name]


def b1_screen(
    *,
    white_n=10,
    yellow_windows=(14, 28, 57, 114),
    kdj_n=9,
    kdj_m1=3,
    kdj_m2=3,
    j_max=8.0,
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
    pe_above=0.0,
    percentile_below=0.20,
    top_n=None,
) -> Screen:
    """**B1 策略**的选股规则：趋势 + 价格在慢线上 + 位置 + J 值 + 量能 + 估值，
    **按 J 值的超卖程度**排序。

    它是 ``Screen``，故同一条件既能用于 ``mbt screen`` 选股，也能作为回测的入场闸门——
    不必写两遍（ADR-0001）。

    ==================  ================================================
    过滤器              判据
    ==================  ================================================
    趋势                白线在黄线上
    价格在慢线上        收盘 > 黄线
    位置                「一波上涨之后的下跌阶段」
    J 值                ``J < j_max``（默认 8）
    量能                上涨放量 + 回调缩量
    PE 为正             ``PE > pe_above``（默认 0，剔除亏损）
    PE 百分位低         ``PE 百分位 < percentile_below``（默认 0.20）
    ==================  ================================================

    排序因子是 :func:`~mbt.signals.factors.j_oversold`（``−J``，越大越超卖）。它回答
    「今天谁的 J 更低」，而 J 的**门槛**由上一条过滤器管（``j_max``）——两者分工不同：
    因子排序、过滤器取舍。

    .. note::

        **这里不再有「交易盈亏比 > 门槛」那一条。** 它曾是第七条（赚头看白线、亏头看黄线，
        ``min_reward_risk`` 默认 4.0，见 :func:`~mbt.signals.factors.reward_risk_ratio`），
        现已移除。那个函数本身**仍留在信号层**（公开、有测试），只是不再被这条规则使用——
        与 :func:`~mbt.signals.factors.yellow_proximity` 的处置相同。

    .. warning::

        **最后两条读的是财务数据，不在行情里。** 调用方必须把
        :func:`~mbt.data.valuation.valuation_for` 的 ``pe`` / ``pe_percentile`` 并进
        传给引擎的 ``signals=``（它会经 :func:`~mbt.data.panel.with_signals` 成为面板字段）。
        面板若缺这两个字段，本规则会**报错并说明该怎么办**，而不是静默少一条过滤。

        它们与 ``UndervaluedGrowth`` 的对应条件同源同口径（都是
        ``mbt.data.valuation``），故 `mbt screen` 那边需要 ``--cw-root``。

    .. note::

        **「PE 为正」不是「PE 百分位低」的冗余。** 百分位是 ``[0, 1]`` 的 min-max 归一化
        位置，而**非正 PE 照常参与** ``LLV``/``HHV``（见 :func:`~mbt.data.valuation.pe_percentile`
        的说明）。故一只亏损股完全可能落在低位（它自己的 PE 在窗口里最低），必须由这一条
        单独剔除。

    .. note::

        **盈亏比换了参照**，从「前高 ÷ 前低」改为 **``(白线 − 收盘) ÷ (收盘 − 黄线)``**：
        赚头看白线、亏头看黄线，与两条卖出规则（最高价破白线、连续破黄线）对得上。
        故它现在只依赖收盘价，不再需要摆动点。同一改动把它的参数从
        ``(panel, anchors, windows, stop_buffer)`` 简化为 ``(prices, white_n, windows)``。

    .. warning::

        **排序因子换成 J 之后，「盈亏比」就只剩过滤这一个作用了。** 而 ``top_n=None``
        （默认，「本金不限」）意味着每个合格标的都买一份——那时排序因子对买入**没有影响**，
        只在给了 ``top_n`` 时才决定「取哪 N 只」。这是两条独立的旋钮，不要指望改排序会
        改变不限额下的结果。

    .. warning::

        **``j_max`` 从 20 收到 8。** 九个样本的信号日 J 落在 −13.4 ~ 19.8（中位 −4.8），
        故 20 覆盖 9/9，而 8 只覆盖一部分（J 在 8~19.8 之间的样本会被挡掉）。收紧它是
        需求方的决定，不是标定出来的；实测影响见下面的注记。

    .. note::

        :func:`~mbt.signals.swings.swings` 在这里被算了**两次**（「位置」与「量能」各一次，
        外加两次过滤器调用各自的份）。盈亏比不再需要摆动点，故从四次降到两次。
    """
    from mbt.data.valuation import PE, PE_PERCENTILE
    from mbt.signals import (
        above_yellow,
        j_below,
        j_oversold,
        pullback_after_advance,
        swings,
        volume_contraction,
        white_above_yellow,
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

    def profitable(panel):
        return _valuation_field(panel, PE) > pe_above

    def cheap(panel):
        return _valuation_field(panel, PE_PERCENTILE) < percentile_below

    def oversold(panel):
        return j_oversold(panel, kdj_n, kdj_m1, kdj_m2)

    return Screen(
        filters=(
            trend,
            not_below_the_line,
            position,
            low_j,
            volume,
            profitable,
            cheap,
        ),
        factor=oversold,
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
