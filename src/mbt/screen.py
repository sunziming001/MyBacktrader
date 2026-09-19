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
from mbt.universe import CHINEXT, MAIN_BOARD, STAR_MARKET

#: **砖型**规则的板块范围：主板（沪深两市）、创业板、科创板，**排除北交所**。
#:
#: 「主板」这一个概念名已经合并了沪、深两个交易所口径（规则表按交易所分列是为了过户费，
#: 那是另一回事，见 :func:`mbt.universe.build_universe`）。故这里写三个名字就是需求里那三个。
BRICK_BOARDS = frozenset({MAIN_BOARD, CHINEXT, STAR_MARKET})

#: 砖型要看多深的历史才算得准（交易日）。**取两者之大**：
#:
#: 1. **递推记忆**：砖型图那三层平滑里最慢的一层是 α=1/6，按 ``(5/6)^t`` 衰减。取真实日线
#:    切片逐次截到末端 N 根重算，末位读数与全长之差为：
#:
#:    =========  ================  ====================
#:    N 根        末位读数相对差    ``(5/6)^N``
#:    =========  ================  ====================
#:    10         3.78e-01          1.6e-01
#:    20         1.13e-01          2.6e-02
#:    30         3.08e-02          4.2e-03
#:    40         1.54e-03          6.8e-04
#:    **50**     **0（逐位相同）**  1.1e-04
#:    100        0（逐位相同）      1.2e-08
#:    =========  ================  ====================
#:
#:    故 50 根已经足够，取 100 是留一倍余量。
#: 2. **黄线**：那道「收在黄线之上」的门，其最长一条均线要 **114 根**。窗口短于它就整段
#:    缺失、这道门**静默地把所有标的筛掉**——而闸门本该拦住这种窗口。
#:
#: 114 > 100，故声明值取 **114**。日更脚本用的 ``--panel-bars 200`` 仍有余量。
BRICK_LOOKBACK_BARS = 114

#: 组面板时提供给选股规则的字段。刻意**不含** ``amount``：它与信号无关，没有理由被带进
#: 计算（ADR-0009 的「字段显式声明」）。这是一份**固定声明**的集合，不是从数据里推断的。
SCREEN_FIELDS = ("open", "high", "low", "close", "volume")

#: 黄线各条均线的**默认**窗口。B1 与砖型共用这一个常量——``CONTEXT.md`` 把黄线定义成一个
#: 概念，两处各写一份字面量就等于造了第二条黄线（「它们相等」会沦为一句注释）。
DEFAULT_YELLOW_WINDOWS = (14, 28, 57, 114)

#: **选股分数**（``CONTEXT.md``）在**策略可读信号**里的字段名。
#:
#: 它是「名次只有一份」这条纪律的落点：``Screen`` 算出 ``ScreenResult.scores`` 之后，引擎把它
#: **原样**并进 ``broker.signals``，于是策略读到的就是规则算过的那一份。**不让策略照着重算**
#: ——复算必然漂移（权重与归一的默认值在规则那里），而漂移了不会报错。
#:
#: 形状与 :class:`~mbt.screen.ScreenResult` 的 ``scores`` 一致：**标的宽表**（行 = 交易日，
#: 列 = 标的），与评估日的面板同日同标的；过滤器筛掉的标的**其数值仍在**（是否合格由
#: ``selection_mask`` 回答，不靠这张表缺席）。
#:
#: 这条通道**对所有规则一律打开**：不读它的策略一个字节都不受影响（有测试钉住「成交逐笔
#: 相同」），故它是纯增量。规则**没有排序因子**时不注入——那时「这条规则不谈名次」，而
#: ``SCREEN_SCORE_FIELD in signals`` 正是策略分辨这一点的依据，塞一个空帧进去会让它说谎。
SCREEN_SCORE_FIELD = "screen_score"

#: **绿砖**信号（0.0 / 1.0）在信号里的字段名——由 :func:`brick_signals` 产出，
#: 由 ``mbt.strategy.Brick`` 读来决定何时清仓。
#:
#: 常量定义在**产出方**（这里）而不是消费方：名字只有一个来源，改一处不会让另一处静默读不到
#: （读不到一律是缺失，而缺失在策略里表现为「不动作」——一笔该卖的持仓就那样留着）。
GREEN_BRICK_FIELD = "green_brick"

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
    #: 这条规则**自己声明的板块范围**（``CONTEXT.md`` 的**板块**，四个概念名）。
    #:
    #: ``None``（默认）表示「随调用方的股票池准入规则」——规则不插嘴。给了值则是**默认值**
    #: 而不是硬约束：调用方显式指定了板块就以调用方为准（命令行上 ``--boards`` 就是这样）。
    #:
    #: 为什么声明在规则上、而不是由调用方每次手填：板块范围与「这条规则在哪些市场上成立」
    #: 是同一件事，把它与规则写在一起，才不会出现「同一条规则在两台机器上跑出不同的池子」
    #: ——而那**不会报错**，只是名单悄悄不一样。
    #:
    #: 规则**不建池**：它只声明范围，建池仍是 :func:`mbt.universe.build_universe` 的事
    #: （理由见 :meth:`apply` 的 ``universe_mask``）。
    boards: frozenset[str] | None = None
    #: 这条规则**要看多深的历史**才算得准（自评估日往前多少根）。
    #:
    #: ``None``（默认）表示「本规则不额外要求」——调用方按**所有规则里最深的那处**取
    #: （见 ``mbt.cli.PANEL_WARMUP_BARS``），那对短记忆的规则是偏严的，但它不会算错。
    #:
    #: 为什么要声明：面板窗口短于这个深度时，读数会**静默算在更短的历史上**——不报错，
    #: 只是名单与全长算出来的不一样（ADR-0014）。而「多深才算够」正是规则的属性：递推类
    #: 指标的深度由它的记忆衰减定，窗口类指标的深度由窗口本身定。故闸门要比的那个数
    #: 由规则给出，调用方只负责拦。
    #:
    #: 声明的是**读数稳定所需的最小深度**，不是「推荐值」。调用方要留余量自己留。
    lookback_bars: int | None = None
    #: 这条规则的**买单是否只给一次成交机会**（交付给回测的 ``one_shot_buys``）。
    #:
    #: 默认 ``False``：买单买不进就一直挂着、逐根重试。那对**入场条件跨天仍成立**的规则是
    #: 合理的（今天入选、明天还入选，晚一天买到也还是那个理由）。
    #:
    #: 但对**条件逐日重算**的规则，留着它会让一笔单在几天后成交，而那笔成交用的早已不是当日
    #: 那个信号——且**不会报错**。砖型就是这一类（票据 #86 之后它自己声明深度，理由同源：
    #: 「这条规则的时间性质」是规则的属性）。判据与理由见 ``mbt.backtest.costs.AStockBroker``。
    one_shot_buys: bool = False

    @property
    def given_factors(self) -> tuple[FrameTransform, ...]:
        """排序因子，**按给的写法**归一成一个序列——``factor`` 是单项的简写。

        两种写法只在这一个地方归一：从 ``_score_parts``（真去算分）到
        :func:`mbt.report.describe_screen`（只记名字）都读它，故「单数还是复数」这件事
        不会在两个地方各判一次、再各漂一次。两个都给是不合法的配置（该以谁为准没有正确
        答案），这里**报错**而不是挑一个——静默挑一个正是留痕与算分对不上的来路。
        """
        if self.factor is not None and self.factors:
            raise ValueError(
                "factor 与 factors 只能给一个：前者是单项的简写，后者是多因子。"
                "两个都给时该以谁为准没有正确答案，故不猜。"
            )
        if self.factor is not None:
            return (self.factor,)
        return tuple(self.factors)

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
                out = _validate(one_factor(working), index, columns, f"第 {position} 个排序因子")
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
        parts = tuple((one, 1.0) for one in self.given_factors)

        if not parts:
            if self.weights or self.normalize:
                raise ValueError("给了 weights / normalize 却没有给因子：没有可加权的对象。")
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


#: ``Screen.apply`` 递下来的面板在同一次调用里是**同一个对象**，故规则的中间量可以按面板
#: 对象记住——这样一次选股里那笔昂贵的计算只付一遍。缓存按 ``is`` 认而不是按 ``id``：
#: ``id`` 会在旧面板被回收之后指到新对象上。
#:
#: 抽成一处是因为它有两个使用者（B1 的摆动点与量能形态、砖型的砖型图），而「按什么认同一份
#: 面板」是个容易写错的判据——写错会**静默地**用上另一张面板的读数。
def _panel_memo(compute, progress, label):
    """返回一个「按面板对象记住一次计算结果」的记忆器。

    ``compute(panel)`` 只在面板换了对象时才真的跑，且**第一次**跑带 :func:`mbt.progress.timed`
    记时——它落在第一个用到它的那个部件名下，不单列的话那一阶段的账会把它的代价算成自己的。
    """
    from mbt.progress import timed

    cache: dict[str, object] = {}

    def memo(panel):
        if cache.get("panel") is not panel:
            cache["panel"] = panel
            with timed(progress, label):
                cache["value"] = compute(panel)
        return cache["value"]

    return memo


def b1_signals(
    markets,
    *,
    align_to=None,
    white_n=10,
    yellow_windows=DEFAULT_YELLOW_WINDOWS,
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
    yellow_windows=DEFAULT_YELLOW_WINDOWS,
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
    atr_n=14,
    shadow_threshold=1.0,
    contained_days=10,
    pe_above=0.0,
    percentile_below=0.20,
    top_n=None,
    progress=None,
) -> Screen:
    """**B1 策略**的选股规则：趋势 + 价格在慢线上 + 位置 + J 值 + 量能 + 调整期形态 + 估值，
    **按形态分数**排序（ADR-0012）。

    它是 ``Screen``，故同一条件既能用于 ``mbt screen`` 选股，也能作为回测的入场闸门——
    不必写两遍（ADR-0001）。

    ==================  ================================================
    过滤器              判据
    ==================  ================================================
    趋势                白线在黄线上
    价格在慢线上        收盘 > 黄线
    位置                「一波上涨之后的下跌阶段」
    J 值                ``J < j_max``（默认 8）
    量能                上涨放量 + 回调缩量（缩量那条的分母是上涨段**单日最大量**，见 ADR-0011）
    调整期不含横盘串     调整期内**没有**连续 ``contained_days`` 根「内含」
    PE 为正             ``PE > pe_above``（默认 0，剔除亏损）
    PE 百分位低         ``PE 百分位 < percentile_below``（默认 0.20）
    ==================  ================================================

    排序因子是**形态分数**：三项各自在**当日全市场（可交易池）**内转成百分位，再等权相加。

    ========================  ==================================================
    组件                       读法（都是**越小越好**，故先取负再进秩）
    ========================  ==================================================
    顶部无量                   ``− (vol(顶部那根) ÷ max(上涨段除去顶部那根))``
    调整缩量                   ``− (mean(回调段) ÷ mean(上涨段除去顶部那根))``
    长上影                     ``− max(0, 影 ÷ ATR − shadow_threshold)``
    ========================  ==================================================

    .. note::

        **门仍走旧口径的「量能」，只把另三条降级成分**（ADR-0012）。那三句量能条件原先都是
        门，② 的读数把它们分开了：`顶部无量` 在真实持有期尺度上的区分力是法定费用的 3.6 倍
        （h=3 价差 +0.3137%、t=3.03，三个尺度同号），而 `放量` 在同一尺度上正好是零
        （+0.0091%、t=0.09）。但按 ADR-0012 的元规则，「不赚钱」只能把它从判据**降级**、
        不能据此删掉原话里的某条，故这道门本轮不动。

        **这道门按旧口径算**（:func:`~mbt.signals.volume_contraction` 的两半，即放量**且**
        缩量），也就是 ① 那三版对照里的 A 版。③ 接线时曾把它一并换成新口径的
        ``surge_vs_base``、并去掉缩量那半条，④ 的全市场对照把那处连带改动量出来是
        **单独 −7.53 pt**——量级是分数那一步收益（+3.01 pt）的三倍，方向相反；而它本来就
        没有独立证据（② 说的是「放量的区分力是零」，不是「门的窗口该加宽」）。故退回旧口径。
        这样本规则与 A 的差别**只剩排序因子一处**，④ 的对照才可归因。

        **门为什么仍用「放量」而不是「顶部无量」**：① 与 ② 两条独立量法都指向「放量不挣
        自己的饭钱」，而改门得先拿留出期复现的数字。这是一个有意保留的、与读数相抵的选择。

    .. note::

        **``shadow_threshold`` 是这道扣分唯一的旋钮，而力度 ``k`` 在这里没有作用。**
        ADR-0012 把它写成 ``k × max(0, 影÷ATR − 1)`` 并标「力度 k 待定」——在**等权 +
        逐日秩归一**下这个 k 不可辨识：``k > 0`` 只是给整个分量乘一个正常数，百分位完全不变。
        故那条待办自动消解，剩下的只有门槛 ``shadow_threshold``（默认 1.0，由实测里
        ``>1.0`` 只占 12.5% 定）。

    .. warning::

        **② 量的是** ``−影÷ATR``（未截的原始读数），而这里接的是**截过的扣分**。两者的秩在
        门槛之下不同：截过之后约 87.5% 的候选格并列为 0、只由另两个分量分胜负。故 ② 那个
        ``+0.2183%`` 不能直接读作本实现的预期——它是**这一项有方向性**的证据，不是这一个
        函数形式的读数。④ 的对照会照出差别；若结果不理想，未截的原始形式是第一个该试的变体。

    排序因子从 :func:`~mbt.signals.factors.j_oversold` 换下来之后，J 值**只剩门槛**
    （``j_max``）这一个作用：因子排序、过滤器取舍，两者分工不同。``j_oversold`` 本身仍
    留在信号层（公开、有测试），只是不再被这条规则使用——与
    :func:`~mbt.signals.factors.yellow_proximity` 的处置相同。

    .. note::

        **「内含」与「横盘串」的定义见** :func:`~mbt.signals.filters.no_contained_run`：
        当天的**收盘价**落在**前一根**的 ``[最低价, 最高价]`` 之内。连续 ``contained_days``
        根及以上这样的 K 线意味着价格在原地震荡——既创不出新高、也砸不出新低，那说明这段
        「回调」其实是横盘而不是回调。取 ``[峰值+1, 当根]`` 为范围（与「位置」「量能」同
        一处边界）。

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

        **排序因子换成形态分数之后，「盈亏比」就只剩过滤这一个作用了。** 而 ``top_n=None``
        （默认，「本金不限」）意味着每个合格标的都买一份——那时排序因子对买入**没有影响**，
        只在给了 ``top_n`` 时才决定「取哪 N 只」。这是两条独立的旋钮，不要指望改排序会
        改变不限额下的结果。**回测走的就是不限额那条路**，故 ④ 的对照必须先给 ``top_n``，
        否则新旧两个因子会跑出逐笔完全相同的结果。

    .. warning::

        **``j_max`` 从 20 收到 8。** 九个样本的信号日 J 落在 −13.4 ~ 19.8（中位 −4.8），
        故 20 覆盖 9/9，而 8 只覆盖一部分（J 在 8~19.8 之间的样本会被挡掉）。收紧它是
        需求方的决定，不是标定出来的；实测影响见下面的注记。

    .. note::

        :func:`~mbt.signals.swings.swings` 与 :func:`~mbt.signals.volume_pattern` 在这里各
        只算**一次**：``Screen.apply`` 把**同一个**面板对象依次递给每条过滤器与每个因子
        （``as_of`` 为空时它原样返回），故这两项可以按面板对象记住。这不是精致化——
        ``volume_pattern`` 全市场约 59 秒而三个分量要用到它 3 次，``swings`` 约 33 秒而要用
        到 3 次（「位置」「量能」「调整期横盘」）。重算几遍会把每次选股多拖几分钟，而结果
        逐位相同。

        （这道「量能」门走的是另一个函数 :func:`~mbt.signals.volume_contraction`，它内部
        自算一遍 :func:`~mbt.signals.volume_structure`，**不复用**上面那份 ``volume_pattern``
        ——两者口径不同、不可互换，见 ADR-0012。故门那一项的耗时是它自己的，实测全市场约
        315 秒，比记一次 ``volume_pattern`` 贵得多；这是退回旧口径的代价之一。）
    """
    from mbt.data.valuation import PE, PE_PERCENTILE
    from mbt.signals import (
        above_yellow,
        j_below,
        no_contained_run,
        pullback_after_advance,
        swings,
        volume_contraction,
        volume_pattern,
        white_above_yellow,
    )

    windows = tuple(yellow_windows)

    #: `Screen.apply` 递下来的面板在同一次调用里是**同一个对象**（见上面那条注记），故中间量
    #: 按对象本身记住（见 :func:`_panel_memo`）——这两个首算都很贵，而下面有七个部件要用到它们。
    anchors_of = _panel_memo(
        lambda panel: swings(panel["close"], retracement=retracement),
        progress,
        "swings（按面板缓存，首个用到它的部件付这一次）",
    )
    pattern_of = _panel_memo(
        lambda panel: volume_pattern(
            panel, anchors_of(panel), base_bars=volume_base_bars, atr_n=atr_n
        ),
        progress,
        "volume_pattern（按面板缓存，三个因子共用这一次）",
    )

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
        # 门按**旧口径**（A 版）算：`volume_contraction` 的两半同时成立，即放量**且**缩量。
        # ③ 曾换成新口径的 `surge_vs_base`（并去掉缩量那半条），④ 量出那处连带改动单独值
        # −7.53 pt 且无独立证据，故退回。理由见函数 docstring 里那条注记。
        return volume_contraction(
            panel["volume"],
            anchors_of(panel),
            edge_bars=volume_edge_bars,
            base_bars=volume_base_bars,
            min_surge=min_surge,
            max_pullback=max_pullback,
        )

    def no_flat_pullback(panel):
        return no_contained_run(panel, anchors_of(panel), days=contained_days)

    def profitable(panel):
        return _valuation_field(panel, PE) > pe_above

    def cheap(panel):
        return _valuation_field(panel, PE_PERCENTILE) < percentile_below

    # 三个分量都「越小越好」，而因子契约是「越大越靠前」，故一律**取负**。取负之后
    # ``normalize="rank"`` 会把负值转成百分位，方向就统一了；不必各自手写换标度。
    def top_calm(panel):
        return -pattern_of(panel).top_calm

    def pullback_shrink(panel):
        return -pattern_of(panel).pullback_vs_advance

    def top_shadow(panel):
        # 「长上影扣分」是**门槛式**的：只有超过 ``shadow_threshold`` 的那部分才算扣分。
        # ``np.maximum`` 会把缺失原样留下（不会把 NaN 变成 0）——缺失不是「没有上影」。
        return -np.maximum(0.0, pattern_of(panel).top_shadow_atr - shadow_threshold)

    return Screen(
        filters=(
            trend,
            not_below_the_line,
            position,
            low_j,
            volume,
            no_flat_pullback,
            profitable,
            cheap,
        ),
        factors=(top_calm, pullback_shrink, top_shadow),
        weights=(1.0, 1.0, 1.0),
        normalize="rank",
        top_n=top_n,
    )


def brick_screen(
    *,
    atr_n: int = 14,
    yellow_windows: tuple[int, ...] | None = DEFAULT_YELLOW_WINDOWS,
    top_n=None,
    progress=None,
) -> Screen:
    """**砖型**的选股规则：前一天是**绿砖**、当天是**红砖**、且红砖**比那根绿砖大**，
    按四项排序（需求方给定，见票据 #81；后两项在 2026-09-19 加的）。

    四条过滤器。**前三条**读的是同一个**砖型图**
    （:func:`~mbt.signals.indicators.brick_line`），第四条读黄线：

    ==================  ====================================================
    过滤器              判据
    ==================  ====================================================
    前一天是绿砖        ``砖型图[t−1] < 砖型图[t−2]``
    当天是红砖          ``砖型图[t] > 砖型图[t−1]``
    红砖比绿砖大        ``砖的大小之比 > 1``（**严格**大于，等于即不合格）
    收在黄线之上        收盘 > **黄线**（见 :func:`~mbt.signals.filters.above_yellow`）
    ==================  ====================================================

    .. warning::

        **最后一条与「前一天绿砖」在方向上相冲**，这是需求方的设计，但代价要写明：绿砖那一族
        条件找的是「一波回调之后的转折向上」，而「收在黄线之上」要求价格已经站回慢线——
        **回调正深时收盘通常在黄线下方**。故两者同一天同时成立的格不多：合成行情上量到它把
        候选压到约**四成**（133 → 59 格）。

        （此前这一条曾是「收在前若干根的``最高价``之上」，那个更狠——只剩 **6%**
        （133 → 8 格），且「红砖比绿砖大」在那里几乎不可能成立：价格贴着 4 日高点走时砖型图
        会顶在天花板，任何回调都砸出一个很大的绿砖。现已换成黄线门，理由见票。）

    排序因子**四项**，前两项各占三分之一、后两项**合成**那第三个三分之一（各六分之一）：

    ==================  ====================================================
    因子                读法（**越大越靠前**，故「越小越好」的那几条先取负）
    ==================  ====================================================
    砖的大小之比        红砖的大小 ÷ 绿砖的大小
    量能                当根成交量 ÷ **前一根**成交量
    涨幅                当根相对**前收盘**的涨幅，**越小**越靠前
    上影                当根的上影 ÷ ATR，**越短**越靠前
    ==================  ====================================================

    后两条在需求方那边是「三点」——「砖的大小之比」「量比」「当天的形态（涨得越小、上影越
    短）」——故第三点由两条读数对半构成。写进 ``weights`` 就是 ``(1, 1, 0.5, 0.5)``（``Screen``
    会归一到和为 1）；这么写而不是写 ``(1/3, 1/3, 1/6, 1/6)``，是为了让「前两条各一份、后两条
    各半份」这件事在代码里看得出来，也少一处除法。

    .. warning::

        **「涨幅越小越好」与「红砖比绿砖大」方向相反，这是刻意的。** 大砖通常出现在大阳日，
        而这一项要求别追高——两条同时进排序，谁赢由上面的份额定。若将来发现这一项在名额不
        紧张的日子里**等于没生效**（排序只影响取前 N 时谁排前面），那说明它的意图其实是
        **一道门**（「今天涨太多就不买」）而不是排序，那是一次移除标的的动作，需要独立证据
        ——见 ADR-0012 记下的那条纪律。

    .. note::

        **因子在 2026-09-19 变过，旧产物与新的不可比**（同一天同一份行情会因为排序不同而给出
        不同名单）。之前那两份产物——667 只候选与那轮归零的全市场回测——用的是**两项等权**
        的版本。

        **这两条新因子量过：没有可用的收益证据**——按天配对（全市场、5,173 只、每天取前 5）
        的差在 k=1 是 +0.07pp（比值 1.60）、在 k=3/5 是噪声、在 k=20 反号（−0.22pp，且那处
        样本重叠、标准误偏小），而往返成本约 0.5%。把它当成「已验证有效」是错的；照需求留下
        它、并把「没证据」记在案，见
        [ADR-0017](../../docs/adr/0017-brick-new-factors-measured-no-evidence.md)。

    两只过滤器保证「大小之比」在入选格上**就是**「红砖 ÷ 绿砖」；分母恒为正，因为绿砖是严格
    下降（见 :class:`~mbt.signals.indicators.BrickLine` 的那条不变量）。其余形态下同一个读数
    算的是别的比值——会被过滤器筛掉，故这里不替它预设形态。

    .. note::

        **量能那一项不是「量比」，本项目刻意不用那个词。** 官方量比是「当日开盘后每分钟平均
        成交量 ÷ 过去 5 个交易日每分钟平均成交量」，是个**盘中**指标，用日线数据无法忠实复现；
        用一个有法定含义的词去指另一件事正是本项目反复警惕的失真（见
        :func:`~mbt.signals.indicators.volume_ratio` 的告警与 ``README.md``）。本项就是
        「当根整日量 ÷ 前一根整日量」，由现成的 :func:`~mbt.signals.indicators.volume_ratio`
        取 ``n=1`` 算出——**不另写一份**。

    .. note::

    **四项不是等权**：前两项各占三分之一，后两项**合成**那第三个三分之一（各六分之一）。
    同样，「等权」指的是**秩**等权而不是数值等权——故一个极端放量的标的不会因为倍数特别大而
    压过另一个；等比的是各自在池内的分位（``Screen`` 的 ``normalize="rank"``）。

    .. note::

        **板块范围写在规则上**（``boards``）：主板（沪深两市）、创业板、科创板，**排除北交所**。
        调用方显式给了板块就以调用方为准。

    .. note::

        **这条规则只读行情**，不需要财务数据目录（``--cw-root``）——它的四条过滤器与四项读数
        全部由开高低收与成交量算出，没有一项来自财报（故它比 B1 少一个必须配对的路径开关）。

    .. note::

        砖型图那三层递推平滑的记忆按 ``(5/6)^t`` 衰减，故本规则**自己声明**要看多深的历史
        （``lookback_bars``）——面板窗口短于它时读数会**静默算在更短的历史上**，那道闸门据此
        拦住。实测末位读数在 50 根上已与全长逐位相同，而**黄线最长那条均线要 114 根**，故声明
        值取**两者之大**（114），详见 ``BRICK_LOOKBACK_BARS`` 里那份账。

    参数:
        atr_n: 「上影」那一项折算用的 ATR 窗口。默认 14，与 B1 的 ``top_shadow_atr`` 同口径
            ——两条规则的这条读数才可比。
        yellow_windows: 黄线各条均线的窗口。默认 :data:`DEFAULT_YELLOW_WINDOWS`，与 B1 **同一个
            常量**——``CONTEXT.md`` 把黄线定义成一个概念，两处取不同窗口就等于造了第二条黄线。
            给 ``None`` 就**不设这道门**，那是给**对照与测量**用的（「加这道门之前之后差多少」
            必须能一次跑出来），不是给策略调的旋钮：真要用那条规则，就用它本来的样子。

            注意它与 :data:`BRICK_LOOKBACK_BARS` 的关系：声明深度取的是**这个窗口与递推记忆
            之大**，故传一组更长的窗口时要一并把声明深度提上去，否则闸门会放行一个太短的
            面板，而那道门在短面板上会因黄线整段缺失而**静默地筛掉所有标的**。
        top_n: 只取前 N 名。``None``（默认）表示不截断。
        progress: 进度上报的接收端。砖型图是三次递推平滑，**三个读它的部件**都要用它，故
            按面板对象**只算一次**并单独记时——否则它的代价会被算进第一个碰到它的那个部件名
            下（见 :func:`mbt.progress.timed`，那里叫它「三个读砖型图的部件」）。
    """
    from mbt.signals import (
        above_yellow,
        brick_line,
        green_brick,
        momentum,
        red_brick,
        upper_shadow_atr,
        volume_ratio,
    )

    readings = _panel_memo(
        brick_line,
        progress,
        "brick（按面板缓存，三个读砖型图的部件共用这一次）",
    )

    def previous_green(panel):
        # `fill_value=False`：调序在最后一行不成立，而「前一根是绿砖」在那时**为假**
        # （没有前一根）——不能让它变成缺失，过滤器会把缺失当不合格，两者恰好一致，
        # 但显式写出来才读得出「这是判据，不是缺数据」。
        return green_brick(readings(panel).line).shift(1, fill_value=False)

    def red_today(panel):
        return red_brick(readings(panel).line)

    def bigger_than_the_green(panel):
        # 比值缺失（分母为 0 或前一根没有砖）时比较恒为假 —— 正是「不合格」。
        return readings(panel).size_ratio > 1.0

    def above_the_yellow(panel):
        # 状态而非事件：今天**收在黄线上方**（`above_yellow` 与 B1 用的是同一个口径）。
        return above_yellow(panel["close"], yellow_windows)

    filters = (previous_green, red_today, bigger_than_the_green)
    if yellow_windows is not None:
        filters = (*filters, above_the_yellow)

    def size_ratio(panel):
        return readings(panel).size_ratio

    def traded_volume_ratio(panel):
        return volume_ratio(panel["volume"], n=1)

    def smaller_gain(panel):
        # 当根相对**前收盘**的涨幅，取负（因子一律「越大越靠前」）。复用现成的动量取 n=1
        # ——「涨幅」就是它，不另写一份。
        return -momentum(panel["close"], 1)

    def shorter_shadow(panel):
        # 当根的上影 ÷ ATR，取负。上影取正值、越大越差，故必须翻符号（见 upper_shadow_atr）。
        return -upper_shadow_atr(panel, atr_n)

    return Screen(
        # `yellow_windows=None` 时不挂「收在黄线之上」那道门（见 docstring：那是给对照与测量用的）。
        filters=filters,
        factors=(size_ratio, traded_volume_ratio, smaller_gain, shorter_shadow),
        # (1,1,0.5,0.5) 归一之后 = 1/3、1/3、1/6、1/6：前两条各一份，后两条合成第三份。
        weights=(1.0, 1.0, 0.5, 0.5),
        normalize="rank",
        top_n=top_n,
        boards=BRICK_BOARDS,
        lookback_bars=BRICK_LOOKBACK_BARS,
        # 买入条件逐日重算，故买单只有次根那一次机会——见 `Screen.one_shot_buys`。
        one_shot_buys=True,
    )


def brick_signals(markets, *, align_to=None) -> dict[str, pd.DataFrame]:
    """算出卖点要读的信号（**标的宽表**），供 ``run_portfolio_backtest(signals=...)``。

    砖型只有一条卖出规则——「当天是**绿砖**」——故这里只给一个字段。它是**状态**而不是
    事件：答的是「今天这根是不是比昨天低」，不是「今天是不是刚从红转绿」。两者在持有期里
    分岔得很多（绿砖可以连出好几根），而需求说的是前者。

    在**完整历史**上算，再截到回测区间——这正是「先算后截」的口径（:func:`mbt.data.panel.clip_fields`
    的说明）。砖型图的递推记忆约百根，若先截再算，区间开头那一段的读数会算在更短的历史上，
    于是同一段行情在两个区间里结论不同。

    参数:
        markets: 一组已质检行情。**必须是完整历史**（``load_market_data`` 的返回值），
            不是切过窗口的那些——否则「先算后截」就反了。
        align_to: 一组行情；给了就把结果的**标的集合与交易日范围**对齐到它。
            **回测时必须给**，且必须传**切过窗口的**那一组行情：引擎要求信号与行情面板
            逐日同列对齐，而完整历史的信号比切过的行情长、也可能多出被跳过的标的，
            不对齐就会以「index / columns 不一致」报错。

            刻意接受一个「行情组」而不是 ``(start, end)`` 两个值：那两个值若与实际切片
            不一致，结果仍是错的，而**行情的边界就在行情里**——从它推出来不会漂移。

    返回:
        ``{GREEN_BRICK_FIELD: 标的宽表}``，取值 0.0 / 1.0（**浮点**）：消费方是策略侧的
        :func:`~mbt.strategy.signal_value`，它读标量，而浮点省掉一处类型分支
        （与 :func:`b1_signals` 同一处置）。

        与 :func:`brick_screen` 一样走 :func:`~mbt.signals.indicators.brick_line`，故
        「绿砖怎么算」只有一处定义——选股与卖出读的是同一条线（ADR-0001）。
    """
    from mbt.data.panel import assemble_panel, clip_fields
    from mbt.signals import brick_line, green_brick

    panel = assemble_panel(markets, ("high", "low", "close"))
    fields = {GREEN_BRICK_FIELD: green_brick(brick_line(panel).line).astype(float)}

    if align_to is None:
        return fields

    bounds = [market.prices.index for market in align_to]
    lower = min(index.min() for index in bounds)
    upper = max(index.max() for index in bounds)
    return clip_fields(fields, align_to, start=lower, end=upper)


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
