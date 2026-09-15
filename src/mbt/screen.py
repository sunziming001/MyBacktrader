"""选股规则：过滤器 + 排序 + 取前 N（票据 #7，AC 1–3、5）。

`CONTEXT.md` 里**选股规则**是「过滤器 + 排序 + 取前 N 的组合，产出一个候选标的集。它回答
『今天买哪些』，不回答『买多少』」。本模块就是那句定义。

## 三个部件都是「面板 → 标的宽表」的纯函数

取**行情面板**（ADR-0009）而不取单字段的标的宽表，是因为必须同时容纳两类部件：多数过滤
信号只吃一个字段（``new_high``），而 ``atr`` 一行内要用 high/low/close。面板是唯一能统一
两者的输入形状，也让回测侧（手上有 ``MarketData``）与独立选股侧走同一条路径。

代价是单字段信号要写成 ``lambda panel: new_high(panel["close"], n)``——略啰嗦，但显式，比再
引一层「按字段分派」的机制便宜。

## 整体是因果的，故可**一次算完**

选股只在**一行之内**比较（排序取前 N 不跨行），而信号本身只回看。故「整张帧一次性算出」
与「逐日以当日为评估日各算一次」结果必然相同——这条性质有测试钉住（截断重算不变性，
见 ``tests/test_screen.py``）。因此引擎与调用方都可以算一次、按日期取行，不必逐 tick 重跑。

## 候选集是 boolean 标的宽表

与**股票池**同形（ADR-0001 的「行 = 截面」）。回测侧只要把它当闸门，与股票池是同一套机制；
CLI 取某一行即成清单。故**一种形状，两个消费方向**。
"""

from __future__ import annotations

import datetime as dt
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import pandas as pd

from mbt.data.panel import Panel
from mbt.signals import drawdown_from_high

#: 组面板时提供给选股规则的字段。刻意**不含** ``amount``：它与信号无关，没有理由被带进
#: 计算（ADR-0009 的「字段显式声明」）。这是一份**固定声明**的集合，不是从数据里推断的。
SCREEN_FIELDS = ("open", "high", "low", "close", "volume")

#: 「低估成长」策略用的跌幅信号字段名。
#:
#: 由 :func:`drawdown_fields` 产出，与 :func:`undervalued_growth_screen` 同在本模块——两者的
#: 耦合是**局部**的，故名字写在一处即可，不必当参数传来传去。
DRAWDOWN_FIELD = "drawdown_1y"

#: 「一年」按**交易日**计，取 252（与 :data:`mbt.metrics.TRADING_DAYS_PER_YEAR` 的年化口径
#: 一致）。刻意不用 365 个日历日：行情只在交易日有值，用日历日会让窗口长度随假期漂移。
DRAWDOWN_LOOKBACK = 252

#: 跌幅门槛，来自需求方的设定（「跌幅应该大于 42%」）。
DRAWDOWN_THRESHOLD = 0.42

#: 总市值门槛（元），来自需求方的设定（「市值应该大于 100 亿」）。100 亿 = 1e10 元。
MIN_MARKET_CAP = 1e10

#: ROE 门槛，**百分数**（``10.0`` 表示 10%），来自需求方的设定。
#: 口径见 :data:`mbt.data.valuation.ROE`。
MIN_ROE = 10.0

#: 净资产（归母股东权益）门槛，单位**元**。``0.0`` 表示「净资产必须为正」。
#:
#: .. note::
#:
#:     **在默认的 ``pe_above = 0.0`` 下，这条过滤恒为冗余**——它被「动态PE > 0」蕴含：
#:
#:         净资产 < 0 且 ROE > 10 ⟹ 净利TTM < 0 ⟹ PE < 0 ⟹ 已被排除
#:
#:     （ROE 的分子分母同时为负时商才为正，见 :data:`mbt.data.valuation.EQUITY`。）
#:     全市场核验：反例 **0 格**（8,361,752 个有效格里「净资产 ≤ 0」有 29,914 个，其中
#:     98% 的 ROE > 10，却没有一个 PE > 0）；加过滤前后候选集都是 **8,602 格**，无差异。
#:
#:     它真正的用处在 ``pe_above < 0``：那时亏损公司被放行，净资产为负的那些会带着巨大的
#:     正 ROE 挤进排序（排序因子正是 ROE），这条下界就变成必需的。把它写进条件，是为了让
#:     「净资产为正」是一个**被声明的前提**，而不是 PE 符号的偶然推论。
#:     冗余关系有测试钉住：``test_the_equity_filter_only_bites_when_negative_pe_is_allowed``。
MIN_EQUITY = 0.0


def drawdown_fields(markets, *, lookback: int = DRAWDOWN_LOOKBACK) -> dict[str, pd.DataFrame]:
    """由**完整**行情算出「近一年跌幅」信号，供 :func:`undervalued_growth_screen` 取用。

    返回 ``{字段名: 标的宽表}``，可直接与估值信号合并后交给 ``Screen``（见
    :func:`mbt.data.panel.with_signals`）。

    .. warning::

        **必须传完整行情，不能传已被 ``--start`` 截断的行情。** 跌幅要回看 ``lookback`` 个
        交易日；若拿截断后的行情去算，回测开头那一段的窗口不足 → 全为缺失 → 过滤条件
        （缺失取假）会把**开头整年的候选全部排除**，而且**不报错**。这与估值百分位的坑是
        同一个（见 :func:`mbt.data.valuation.valuation_for`），故同样走「先算后截」：

        1. 用**完整**行情算跌幅；
        2. 用 :func:`mbt.data.panel.clip_fields` 截到回测区间。

    参数:
        markets: 一组 :class:`~mbt.data.market.MarketData`（**完整历史**）。
        lookback: 回看的交易日数，默认 :data:`DRAWDOWN_LOOKBACK`。
    """
    from mbt.data.panel import assemble_panel

    if not markets:
        raise ValueError("算跌幅至少要有一个标的")
    panel = assemble_panel(list(markets), ("high", "close"))
    return {DRAWDOWN_FIELD: drawdown_from_high(panel, lookback)}


#: 过滤器与排序因子的共同签名：**行情面板进、标的宽表出**。
#:
#: 刻意不叫 ``Signal``——`CONTEXT.md` 里「信号」是**过滤信号**与**排序因子**各自 `_Avoid_`
#: 列出的简称，用它当类型名会让两种部件在代码里混为一谈。
FrameTransform = Callable[[Panel], pd.DataFrame]


#: 归一方式。``"rank"``：逐日把每一项在**可交易池**内转成百分位（越大越靠前的分位）。
#:
#: 只有这一种，因为合成分数要解决的就一件事——把**尺度不同、方向已统一**的几项变成可相加的
#: 东西。百分位是最不挑分布的那种做法（不假设正态、不被单根极值主导）。
NORMALIZATIONS = ("rank",)


@dataclass(frozen=True)
class Screen:
    """一条选股规则：过滤器 + 排序 + 取前 N。

    三个部件**都可独立配置**（AC 1）：

    - ``filters``：若干个过滤信号，以 **AND** 组合。空元组表示「不过滤」。
    - ``factor`` / ``factors``：排序因子，按信号层契约**越大越靠前**。``None`` / 空元组表示不排序。
    - ``top_n``：只取前 N 名。``None`` 表示不截断。

    **多因子（``factors`` + ``weights`` + ``normalize``）**：给多个因子时，本类先把它们
    **合成一个分数**再排序，故 ``ScreenResult.scores`` 始终是**一张**宽表，下游
    （``_keep_top``、``candidates()``）无需知道它由几项拼成。

    - ``weights``：各项的权重，**必须为正**。``None`` 表示等权。刻意不允许负权重——那等于
      给了一个反转方向的开关，而「要反向就在因子层写一个新因子」是本类的既定立场（见下）。
    - ``normalize``：``"rank"`` 时，每一项先**逐日**在**可交易池**内转成百分位，再按权重
      取加权平均。**多因子必须给归一**——几项尺度不同的数直接相加，加出来的东西没有含义，
      而那不会报错。
    - 归一按**可交易池**（``universe_mask``）算，不是按「当天通过过滤的那几格」。池外的高分
      不该把池内的分位往下压；这也是归一放在**本类**而不是写成闭包因子的原因之一：
      只有本类在那一刻手上有股票池掩码。

    **缺失**：某项缺失时该项的百分位记 0（最差），故一个缺项的标的**不会**被排除，只是排在
    后面；三项**全缺**才留下缺失（于是被排除，与下面的既定纪律一致）。

    **不提供反转方向的开关**：因子一律「越大越靠前」，这是信号层的既定契约（``distance_to_high``
    特意定义成「接近程度」而非「回撤幅度」，就是为了让所有因子同向）。给一个反转开关会让那条
    约定形同虚设，于是每个因子又得各自交代方向。要反向就在因子层写一个新因子——本类只提供
    权重，且权重为正，就是这条立场的执行。

    **不管持仓**：规则回答「今天买哪些」，不回答「买多少」，也不知道你已经持有什么——
    「重复的要不要跳过」归策略与引擎（``max_positions``、sizer）。持仓是路径依赖的运行时
    状态，而规则按定义是纯函数。
    """

    filters: tuple[FrameTransform, ...] = ()
    factor: FrameTransform | None = None
    factors: tuple[FrameTransform, ...] = ()
    weights: tuple[float, ...] | None = None
    normalize: str | None = None
    top_n: int | None = None

    def apply(
        self,
        panel: Panel,
        *,
        as_of: dt.date | dt.datetime | str | None = None,
        universe_mask: pd.DataFrame | None = None,
        progress=None,
    ) -> ScreenResult:
        """算出整张候选集。

        参数:
            panel: 行情面板，须含 :data:`SCREEN_FIELDS` 里的字段。
            as_of: 评估日。给了就只算到该日为止（**该日之后的记录一律不使用**，ADR-0006）；
                省略则算到数据末端。该日必须是交易日，否则报错——「12 月 31 日的选股结果」
                实际由别的某一天算出，是静默失真。
            universe_mask: **股票池**掩码（boolean 标的宽表）。给了就与选股结果取交集，
                故股票池规则在选股时同样生效（AC 4）。规则**自己不建池**：建池要碰行情、
                准入规则与规则表，那会把这层从纯函数变成碰数据的函数，而 ``build_universe``
                只有一个实现，重复一次必然漂移。

                给了 ``as_of`` 时掩码会被**同样截断**到该日——否则两者的形状必然对不上
                （面板截了、掩码没截），而这会以「形状不一致」报错的形式暴露出来。
            progress: 进度上报的接收端（:class:`~mbt.progress.ProgressReporter`）。``None``
                （默认）时不输出。给出时**逐个过滤器**上报一次——每个过滤器都要独立扫一遍
                整张面板，全市场尺度下单个过滤器就是分钟量级，而在此之前它没有任何输出
                （见 :mod:`mbt.progress`）。

                名字取部件的 ``__name__``：过滤器常写成局部函数（如 ``worth_the_risk``），
                有名字；写成 ``lambda`` 的会得到 ``<lambda>``，一样能读。

        返回:
            :class:`ScreenResult`。

        抛:
            ValueError: 组合不成立（给了 ``top_n`` 却没给 ``factor``、或 ``top_n < 1``）、
                ``as_of`` 不是交易日、某个部件返回的形状与面板不一致、或部件返回的类型不对。
                这些都在**这里**校验而非构造时，因为 dataclass 是公开可直接构造的，而它的
                合法性取决于运行时拿到的面板。
        """
        if self.top_n is not None and self.factor is None and not self.factors:
            raise ValueError(
                "给了 top_n 就必须有排序因子：没有排序就无从谈「前 N 名」。"
                "要按代码顺序取前 N 请显式给一个因子，不要指望默认顺序。"
            )
        if self.top_n is not None and self.top_n < 1:
            raise ValueError(f"top_n 至少为 1，收到 {self.top_n}")

        parts = self._score_parts()

        working = _truncate(panel, as_of)
        index, columns = _shape_of(working)

        if universe_mask is not None and as_of is not None:
            # 掩码要跟着面板一起截，否则形状必然对不上——面板截了、掩码没截。
            universe_mask = universe_mask.loc[: _timestamp(as_of, "评估日")]
        # **先归一掩码**，因为归一要用它当排名池（见类文档）。挪到这里而不是留在最后：
        # 同一个掩码被用两次（排名池、与选股结果取交），两处必须是同一份对齐过的对象。
        universe = (
            _align(universe_mask, index, columns, "股票池掩码")
            if universe_mask is not None
            else None
        )

        selected = pd.DataFrame(True, index=index, columns=columns)
        for i, one_filter in enumerate(self.filters):
            if progress is not None:
                progress.stage(
                    f"过滤器 {i + 1}/{len(self.filters)}：{_part_name(one_filter)}",
                    note="每个过滤器各扫一遍整张面板",
                )
            out = _validate(one_filter(working), index, columns, f"第 {i + 1} 个过滤器")
            if out.dtypes.map(lambda dtype: dtype.kind != "b").any():
                raise ValueError(
                    f"第 {i + 1} 个过滤器返回的不是布尔值——过滤信号回答「合格与否」，"
                    f"是/否才是它的语义。若你要的是连续数值，那是**排序因子**。"
                )
            selected &= out

        scores = pd.DataFrame()
        if parts:
            raws = []
            for position, (one_factor, _) in enumerate(parts, start=1):
                if progress is not None:
                    suffix = "" if len(parts) == 1 else f" {position}/{len(parts)}"
                    progress.stage(f"排序因子{suffix}：{_part_name(one_factor)}")
                out = _validate(
                    one_factor(working), index, columns, f"第 {position} 个排序因子"
                )
                if out.dtypes.map(lambda dtype: dtype.kind != "f").any():
                    raise ValueError(
                        "排序因子返回的不是浮点数——因子要能横向比较大小，布尔答不了"
                        "「谁更靠前」。"
                    )
                raws.append(out)

            scores = _combine(raws, self.weights, self.normalize, universe)
            # 因子缺失即排除：排序未知的标的不参与取前 N，也不该因为「不知道」而被当成合格。
            # 与过滤器「缺失取 False」同一精神。（合成分数在**全部**项都缺失时才留缺失，
            # 只缺一项的按最差分位参与排名，见类文档。）
            selected &= scores.notna()

        if universe is not None:
            selected &= universe

        if self.top_n is not None:
            selected = _keep_top(selected, scores, self.top_n)

        return ScreenResult(selected=selected, scores=scores)

    def _score_parts(self) -> tuple[tuple[FrameTransform, float], ...]:
        """把 ``factor`` / ``factors`` 两种写法归一成「(因子, 权重)」序列，并校验组合。"""
        if self.factor is not None and self.factors:
            raise ValueError(
                "factor 与 factors 只能给一个：前者是单项的简写，后者是多因子。"
                "两个都给时该以谁为准没有正确答案，故不猜。"
            )

        if self.factor is not None:
            parts = ((self.factor, 1.0),)
        else:
            parts = tuple((one, 1.0) for one in self.factors)

        if not parts:
            if self.weights or self.normalize:
                raise ValueError(
                    "给了 weights / normalize 却没有给因子：没有可加权的对象。"
                )
            return ()

        if self.weights is None:
            parts = tuple((one, 1.0 / len(parts)) for one, _ in parts)
        else:
            if len(self.weights) != len(parts):
                raise ValueError(
                    f"weights 有 {len(self.weights)} 个而因子有 {len(parts)} 个——"
                    f"对不上时每一项该拿哪个权重没有正确答案，故不猜。"
                )
            for weight in self.weights:
                if not isinstance(weight, int | float) or weight != weight:
                    raise ValueError(f"权重必须是数字，收到 {weight!r}")
                if weight <= 0:
                    raise ValueError(
                        f"权重必须为正，收到 {weight!r}——负权重等于一个反转方向的开关，"
                        f"而「要反向就在因子层写一个新因子」是本类的既定立场。"
                    )
            total = float(sum(self.weights))
            parts = tuple(
                (one, float(weight) / total)
                for (one, _), weight in zip(parts, self.weights, strict=True)
            )

        if self.normalize is not None and self.normalize not in NORMALIZATIONS:
            raise ValueError(
                f"不认识的归一方式 {self.normalize!r}，只有 {NORMALIZATIONS}——"
                f"静默忽略它会让「几项直接相加」这种没有含义的算法悄悄通过。"
            )
        if len(parts) > 1 and self.normalize is None:
            raise ValueError(
                "多个因子必须给 normalize：几项尺度不同的数直接相加，加出来的东西没有含义，"
                "而那不会报错。"
            )
        return parts


@dataclass(frozen=True)
class ScreenResult:
    """选股结果。

    属性:
        selected: boolean 标的宽表，True 即当日选中。它是**候选集**，也是回测的入场闸门。
        scores: 排序因子的数值，形状同上；无因子时是空帧。缺失处以缺失值表示
            （被排除的标的其数值仍在，但 ``selected`` 已为 False）。
    """

    selected: pd.DataFrame
    scores: pd.DataFrame

    def candidates(self, on: dt.date | dt.datetime | str) -> list[str]:
        """指定评估日选中的标的，**从优到劣**排列。

        顺序即排名，故它同时回答「买哪些」与「谁更靠前」。没有因子时按代码升序。

        抛:
            ValueError: 该日不是交易日（数据里没有这一行）。**不向前取最近的交易日**——
            那会让「这一天的结果」实际是别的某一天算出来的，而且是静默的。注意它与
            「那天选出了空集」是两件事：后者是合法产物。
        """
        stamp = _timestamp(on, "评估日")
        row = _row(self.selected, stamp, "选股结果")
        picked = [symbol for symbol in row.index if bool(row[symbol])]

        if self.scores.empty:
            return sorted(picked)

        scores = _row(self.scores, stamp, "因子值")
        ranked = sorted(picked, key=lambda symbol: (-scores[symbol], symbol))
        return ranked


def momentum_screen(window: int = 20, top_n: int = 10) -> Screen:
    """内置的默认选股规则：按 ``window`` 日动量排序取前 ``top_n``。

    **它是库的一部分，不是 CLI 里的业务逻辑**——AC 明令「CLI 是库入口的薄映射，不含独立
    业务逻辑」。CLI 只**指名**这条规则，规则的语义、默认值与测试都在这里。

    为什么库要给一条默认规则：AC 要求「一条命令跑选股：指定评估日」，而没有任何规则的选股
    没有东西可跑。这条规则足够具体、一跑就出结果，且用的是信号层既有的指标（ADR-0001：
    选股规则消费信号，不自造一套）。

    .. note::

        它只是一个**可用的起点**，不是一条有依据的策略——动量的窗口与取前几名都没经过
        论证。真正的选股规则应当由你自己组装并在回测里检验。
    """
    from mbt.signals import momentum

    return Screen(factor=lambda p: momentum(p["close"], window), top_n=top_n)


def undervalued_growth_screen(
    *,
    percentile_below: float = 0.08,
    pe_above: float = 0.0,
    peg_below: float = 0.75,
    peg_above: float = 0.0,
    drawdown_above: float = DRAWDOWN_THRESHOLD,
    min_market_cap: float = MIN_MARKET_CAP,
    min_roe: float = MIN_ROE,
    min_equity: float = MIN_EQUITY,
    top_n: int = 5,
) -> Screen:
    """**低估成长**策略的买入条件（票据 #51、#52、#58、#62）。

    这是「同一条件不必写两遍」的落点（ADR-0001）：回测用它当入场闸门（引擎的 ``screen``），
    选股用它出候选清单（``mbt screen``），两边是**同一个对象**。

    七个过滤条件：

    - ``动态PE 百分位 < percentile_below`` —— 相对自身历史足够便宜；
    - ``动态PE > pe_above`` —— 剔除亏损（负 PE 无估值含义）；
    - ``peg_above < PEG < peg_below`` —— 增长要为正、且价格相对增长仍算便宜；
    - ``一年内跌幅 > drawdown_above`` —— 从近一年的**最高价**回落足够深；
    - ``总市值 > min_market_cap`` —— 剔除小盘股（默认 100 亿元）；
    - ``ROE > min_roe`` —— 盈利质量门槛（默认 10%，**百分数**）；
    - ``净资产 > min_equity`` —— **净资产为正**（默认 0 元，严格大于）。

    .. note::

        **最后一条在默认参数下是冗余的**，它被「动态PE > 0」蕴含：

            净资产 < 0 且 ROE > 10 ⟹ 净利TTM < 0 ⟹ PE < 0 ⟹ 已被 ``profitable`` 排除

        （ROE 的分子分母同时为负时商为正，故净资产为负的公司会带着巨大的正 ROE 通过 ROE
        门槛——但那种情形下它的净利必为负，PE 因而为负。）全市场核验：反例格数为 0，
        加过滤前后候选集无差异。

        它真正的用处在 ``pe_above < 0``：那时亏损公司被放行，这条下界就变成必需的。
        写进条件是让「净资产为正」成为**被声明的前提**，而不是 PE 符号的偶然推论。
        详见 :data:`MIN_EQUITY`。

    排序因子是 **ROE**：**盈利质量越高越靠前**（``Screen`` 的契约是「越大越靠前」，而 ROE
    天然满足）。

    .. note::

        **排序因子从「跌幅」换成 ROE 是一个立场变化（票据 #58）。** 原来按跌幅排序选的是
        「跌得最狠的那几只」——实测把标的池扩到全市场后，那等于在选小市值/退市边缘的股票
        （净值 −50.9%）。现在改成「先按便宜 + 深跌 + 排除小盘筛出候选，再按**盈利质量**优先」：
        「在折价里挑质量最好的」，而不是「在折价里挑跌得最惨的」。

    这些条件由估值信号（前三条 + 市值 + ROE + 净资产）与 :func:`drawdown_fields`（跌幅）提供，
    调用方要把两组信号都并进行情面板，见各函数的说明。
    """
    from mbt.data.valuation import (
        EQUITY,
        MARKET_CAP,
        PE,
        PE_PERCENTILE,
        PEG,
        ROE,
    )

    def cheap(panel):
        return panel[PE_PERCENTILE] < percentile_below

    def profitable(panel):
        return panel[PE] > pe_above

    def growth_worth_paying(panel):
        return (panel[PEG] > peg_above) & (panel[PEG] < peg_below)

    def deeply_fallen(panel):
        return panel[DRAWDOWN_FIELD] > drawdown_above

    def big_enough(panel):
        return panel[MARKET_CAP] > min_market_cap

    def good_quality(panel):
        return panel[ROE] > min_roe

    def solvent(panel):
        return panel[EQUITY] > min_equity

    def quality(panel):
        return panel[ROE]

    return Screen(
        filters=(
            cheap,
            profitable,
            growth_worth_paying,
            deeply_fallen,
            big_enough,
            good_quality,
            solvent,
        ),
        factor=quality,
        top_n=top_n,
    )


def valuation_screen(
    *,
    percentile_below: float = 0.08,
    pe_above: float = 0.0,
    peg_below: float = 0.75,
    peg_above: float = 0.0,
    top_n: int = 5,
) -> Screen:
    """**旧版**估值筛选（无跌幅条件、按百分位排序）。保留给对照实验用。

    它与 :func:`undervalued_growth_screen` 的区别只有两处：少了跌幅过滤、排序因子是
    ``1 − 百分位`` 而非跌幅。留着它是为了让「加跌幅条件、换排序因子」这件事**可被对照**——
    否则改动前后的差异就无从归因。
    """
    from mbt.data.valuation import PE, PE_PERCENTILE, PEG

    def cheap(panel):
        return panel[PE_PERCENTILE] < percentile_below

    def profitable(panel):
        return panel[PE] > pe_above

    def growth_worth_paying(panel):
        return (panel[PEG] > peg_above) & (panel[PEG] < peg_below)

    def cheapness(panel):
        return 1.0 - panel[PE_PERCENTILE]

    return Screen(
        filters=(cheap, profitable, growth_worth_paying),
        factor=cheapness,
        top_n=top_n,
    )


def _combine(
    raws: list[pd.DataFrame],
    weights: tuple[float, ...] | None,
    normalize: str | None,
    universe: pd.DataFrame | None,
) -> pd.DataFrame:
    """把若干个因子合成**一个**分数。

    单项且未给归一 → 原样返回（故旧的 ``factor=`` 路径行为逐位不变）。

    Multiple → 逐项先归一再加权平均（权重已由 :meth:`Screen._score_parts` 归一成和为 1）。
    **归一按行（日）算，且只在 ``universe`` 为真的列之间排名**：
    池外的高分不该把池内的分位往下压。没有 ``universe`` 时（调用方没给股票池）才退化成
    「整张面板一起排」——那是唯一可用的池，但它与「可交易池」不是一回事，故在类文档里写明。
    """
    if len(raws) == 1 and normalize is None:
        return raws[0]

    parts = [_rank_percentile(one, universe) for one in raws]

    if weights is None:
        share = 1.0 / len(parts)
        shares = [share] * len(parts)
    else:
        total = float(sum(weights))
        shares = [float(weight) / total for weight in weights]

    combined = parts[0] * shares[0]
    for one, share in zip(parts[1:], shares[1:], strict=True):
        combined = combined + one * share

    # 全部项都缺 → 留缺失（于是被排除）。只缺一项的，那一项记 0（最差），照常参与排名。
    everything_missing = raws[0].isna()
    for one in raws[1:]:
        everything_missing &= one.isna()
    return combined.where(~everything_missing)


def _rank_percentile(frame: pd.DataFrame, universe: pd.DataFrame | None) -> pd.DataFrame:
    """逐日在池内把因子转成百分位（越大越靠前，落在 (0, 1]）；缺失记 0。

    用**升序** ``rank`` 再除以当天的有效个数：升序名次越大 ⇒ 因子值越大 ⇒ 分位越接近 1，
    于是「越大越靠前」这条契约在归一之后仍然成立（归一不能顺手把方向翻掉）。
    """
    pool = frame if universe is None else frame.where(universe)
    ranks = pool.rank(axis=1, ascending=True)
    counts = pool.notna().sum(axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        percentiles = ranks.div(counts.replace(0, np.nan), axis=0)
    # 缺失（含当天池内一个有效值都没有）记 0 = 最差。故意不用「中性值」：中性会让「不知道」
    # 与「看得清但一般」等价，而缺失是不知道，按 ADR-0005 的纪律不该给乐观答案。
    return percentiles.fillna(0.0)


def _truncate(panel: Panel, as_of) -> Panel:
    """把面板截到评估日（含当日）——**时点正确性的落点**（AC 5，ADR-0006）。"""
    if as_of is None:
        return panel

    stamp = _timestamp(as_of, "评估日")
    reference = _shape_of(panel)[0]
    if stamp not in reference:
        raise ValueError(
            f"评估日 {stamp.date()} 不是交易日（不在数据的日期索引里）。"
            f"不替你取最近的交易日——那会让这一天的结果实际由别的某一天算出，而且是静默的。"
        )
    return Panel({name: frame.loc[:stamp] for name, frame in panel.fields.items()})


def _shape_of(panel: Panel) -> tuple[pd.Index, pd.Index]:
    frame = next(iter(panel.fields.values()))
    return frame.index, frame.columns


def _part_name(part) -> str:
    """部件在进度里的名字：优先用它自己的 ``__name__``。

    过滤器与因子在本项目里常写成**有名字的局部函数**（``def trend(panel)``），因为规则要能被
    人读懂；``lambda`` 也有名字（``"<lambda>"``）。只有**可调用对象**（实现了 ``__call__`` 的
    类实例）才没有 ``__name__``，那时退回类型名——宁可报个含糊的名字，也不去猜。
    """
    return getattr(part, "__name__", None) or type(part).__name__


def _validate(out, index, columns, label: str) -> pd.DataFrame:
    """部件返回的必须是与面板**逐格对应**的标的宽表，并归一成面板的列序。

    形状不对齐若不拦，`&` 会静默按列名对齐、缺的补缺失值——得到一张看着正常的结果表
    （ADR-0009 记的正是这类静默错答）。

    校验的是**标签集合**而非标签顺序：顺序不影响任何计算结果（``&`` 按标签对齐），把它也
    当错会造成一类莫名其妙的拒绝。归一成面板的列序则让下游的比较与输出稳定可复现。
    """
    if not isinstance(out, pd.DataFrame):
        raise ValueError(f"{label}必须返回 DataFrame（标的宽表），收到 {type(out).__name__}")
    if not _same_labels(out.index, index) or not _same_labels(out.columns, columns):
        raise ValueError(
            f"{label}的日期与标的必须与面板一致："
            f"面板是 {len(index)} 行 × {len(columns)} 列，而它给了 "
            f"{len(out.index)} 行 × {len(out.columns)} 列（或标签不同）。"
            f"选股只在同一行内比较，形状错位会静默错答。"
        )
    return out.reindex(index=index, columns=columns)


def _same_labels(left: pd.Index, right: pd.Index) -> bool:
    """两个索引是否装着同一批标签（忽略顺序）。"""
    return left.is_unique and right.is_unique and set(left) == set(right)


def _align(mask, index, columns, label: str) -> pd.DataFrame:
    """校验并归一掩码，再取布尔——复用 :func:`_validate` 的那一套形状校验，免得两处漂移。"""
    return _validate(mask, index, columns, label).astype(bool)


def _keep_top(selected: pd.DataFrame, scores: pd.DataFrame, top_n: int) -> pd.DataFrame:
    """每行只留因子值最大的前 ``top_n`` 个（在**已通过过滤**的标的之间比较）。

    先把未通过过滤者的因子值抹成缺失再排名，否则一个「分数很高但被过滤掉」的标的会占掉
    名次，把本该入选的挤出去。

    列先按**代码升序**排再排名，于是 ``method="first"`` 的同分处理恰好等于「同分按代码升序
    打破平局」——这可复现，否则同一天两次运行可能给出不同清单。
    """
    by_symbol = sorted(selected.columns)
    masked = scores.loc[:, by_symbol].where(selected.loc[:, by_symbol])
    rank = masked.rank(axis=1, ascending=False, method="first")
    kept = selected.loc[:, by_symbol] & (rank <= top_n)
    return kept.reindex(columns=selected.columns)


def _timestamp(value, label: str) -> pd.Timestamp:
    """把日期归一成 ``Timestamp``；解析不了就报错并说明**是哪一个日期**（评估日常有多个来源）。"""
    try:
        stamp = pd.Timestamp(value)
    except (ValueError, TypeError) as exc:
        raise ValueError(f"{label}无法解析为日期：{value!r}") from exc
    return stamp.tz_localize(None) if stamp.tz is not None else stamp


def _row(frame: pd.DataFrame, stamp: pd.Timestamp, label: str) -> pd.Series:
    if stamp not in frame.index:
        raise ValueError(f"{label}里没有 {stamp.date()} 这一行——它不是交易日，或不在数据范围内")
    return frame.loc[stamp]
