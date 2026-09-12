"""股票池：按每个**调仓日**的当日状态动态计算（票据 #6）。

`CONTEXT.md` 里**股票池**是「某次回测或选股所考虑的全部标的的集合。按每个调仓日的当日
状态动态计算，不是一份固定名单。只从判为「股票」的品种中构建」。本模块就是那件事。

## 形状：boolean 标的宽表

返回 `日期 × 标的` 的布尔表（True = 在池）。三个理由（ADR-0001、ADR-0009）：

1. 「行 = 截面」，而「今天哪些可买」正是一行截面；
2. 「每个调仓日动态计算」自然表达为「每一行都算」；
3. 它与**过滤信号**同形，故准入规则本身就是过滤信号，是同一套东西——#7 的选股规则可以
   直接与它组合，不必再引入第二种表示。

**调仓日由策略决定**（backtrader 的 `next()` 逐日推进），引擎只把这张帧交给策略。

## 时点正确性

每个条件都只看**评估日及之前**的信息：根数用「截至当日的累计根数」而不是整段长度。
后者会引入前视偏差——一只后来涨到 70 根的股票，会在它只有 3 根的时候就获准进入；这不会
报错，只会让结论偏乐观（ADR-0006）。

## 已知局限：ST 排除目前不生效

判定 ST 需要「带生效日期的股票名称历史」，而一期只吃本地数据（ADR-0004），`.day` 不含名称。
接缝是通的——规则表的 `[[st_period]]` 登记了就会按它排除——但**出厂表不登记任何期间**，
故实际永不排除。这一点不得对外含糊（见 README 与 ADR-0002 的修订一节）。

## 成本

本模块按标的逐个处理，内存与「已加载的行情总量」同阶。股票池本可以惰性求值（先用
`TdxDataSource.symbols()` 拿到名单，再只读需要的标的），但一期先用朴素实现把语义钉住；
全市场约 12,000 个标的文件时这一点会明显，届时应交给调用方控制加载范围。
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from mbt.data.instrument import instrument_type, is_stock
from mbt.rules import RuleTable, board_of

#: 主板的概念名。规则表内部把它拆成「沪主板」与「深主板」两行（过户费沪深不同），
#: 而 `CONTEXT.md` 的**板块**只有主板 / 创业板 / 科创板 / 北交所四个值。
MAIN_BOARD = "主板"
CHINEXT = "创业板"
STAR_MARKET = "科创板"
BEIJING = "北交所"

ALL_BOARDS = frozenset({MAIN_BOARD, CHINEXT, STAR_MARKET, BEIJING})

#: 规则表用的五个板块名 → `CONTEXT.md` 的四个概念名。
#:
#: 这个映射存在，是因为规则表按交易所分列主板（过户费沪深口径不同），而**板块**这一概念
#: 在词汇表里不分交易所。两套命名各司其职，映射只此一处。
_BOARD_TO_CONCEPT = {
    "沪主板": MAIN_BOARD,
    "深主板": MAIN_BOARD,
    CHINEXT: CHINEXT,
    STAR_MARKET: STAR_MARKET,
    BEIJING: BEIJING,
}


@dataclass(frozen=True)
class UniverseRules:
    """股票池的准入规则。全部可配置，默认值即 AC 的出厂设定。

    属性:
        boards: 纳入哪些**板块**（用 `CONTEXT.md` 的四个概念名）。默认四个全收。
        min_bars: 至少需要多少根 K 线才可进入。默认 60，即「排除上市不足 60 个交易日的
            次新股」。该门槛度量的是**本地数据的可得量**，不是真实上市日——见
            :func:`build_universe` 的说明。

    **品种不在这里配置**：股票池只从判为**股票**的品种中构建（ADR-0004）。做成可配置的
    参数是没有意义的——非股票标的本就没有板块，规则层的 :func:`mbt.rules.board_of` 对它们
    直接报错，故「允许纳入指数」这种设定根本无法实现，摆在那里只会误导。
    """

    boards: frozenset[str] = field(default_factory=lambda: ALL_BOARDS)
    min_bars: int = 60
    #: 附加的准入掩码（boolean 标的宽表），由**调用方**提供并与上述结构性条件**取与**。
    #:
    #: 它存在的理由：**基本面过滤**（如「非亏损」）不是结构性条件——品种、板块、次新答的是
    #: 「这是什么、够不够格」，而财务条件答的是「它经营得怎么样」。混进上面那几条会让两者的
    #: 失败原因分不开（「它不在池内」到底是因为次新还是因为亏损？），而排查时最需要的就是
    #: 这个区别。
    extra_mask: pd.DataFrame | None = None


def why_not_in_universe(
    markets,
    *,
    rules: UniverseRules | None = None,
    rule_table: RuleTable | None = None,
) -> pd.Series:
    """逐个标的给出「为什么它不在池内」的一句话，供人工抽查。

    与 :func:`build_universe` 共用**同一份**判据（:func:`_exclusion_reasons` 产出有序的
    (条件, 理由) 列表，两者都按它走），故「为什么被排除」讲错的风险只有一处。
    """
    rules = rules or UniverseRules()
    frame = build_universe(markets, rules=rules, rule_table=rule_table)
    reasons = {}
    for market in markets:
        symbol = market.symbol
        if symbol not in frame.columns:
            continue
        reasons[symbol] = (
            "在池内" if frame[symbol].any() else _first_reason(market, rules, rule_table)
        )
    return pd.Series(reasons, name="reason")


def _first_reason(market, rules: UniverseRules, rule_table) -> str:
    """第一条不通过的理由，**顺序与 :func:`build_universe` 一致**。

    两者共用 :func:`_structural_failures`，故此处的顺序不会与那边漂移——而「讲错为什么被排除」
    比不讲更糟：它会把人引到错误的方向去查。
    """
    for reason in _structural_failures(market, rules, rule_table):
        return reason
    return "未通过准入"


def _structural_failures(market, rules: UniverseRules, rule_table):
    """按与 :func:`build_universe` **相同的顺序**列出不通过的理由（可能多条）。"""
    symbol = market.symbol
    if not is_stock(symbol):
        yield f"品种不是股票（{instrument_type(symbol)}）"
        return

    try:
        board = _BOARD_TO_CONCEPT[board_of(symbol)]
    except KeyError:
        yield "板块命名与词汇表不一致"
        return

    if board not in rules.boards:
        yield f"板块 {board} 未纳入"

    # 注意顺序：`build_universe` 先判根数、再判附加掩码、最后判 ST——这里必须一致，
    # 否则「为什么被排除」会指向另一条其实没拦住它的条件。
    if len(market.prices) < rules.min_bars:
        yield f"行情只有 {len(market.prices)} 根，不足 {rules.min_bars} 根"

    if rules.extra_mask is not None and symbol in rules.extra_mask.columns:
        if not rules.extra_mask[symbol].any():
            yield "未通过附加过滤（如「非亏损」）"
    elif rules.extra_mask is not None:
        yield "不在附加过滤的掩码里（未被评估）"

    if rule_table is not None and rule_table.has_st_period(symbol):
        yield "期间内有 ST 登记（被排除）"


def combine_masks(*masks: pd.DataFrame) -> pd.DataFrame:
    """把若干 boolean 标的宽表逐格取与。

    它存在的理由：**股票池与基本面过滤是两份掩码**，而消费它们的地方（`Screen.apply` 的
    ``universe_mask``、撮合的闸门）只收**一份**。合成这一步要显式、要有形状校验——静默按
    标签对齐得到一张看着正常的表，正是 ADR-0009 记的那类错答。

    形状不一致时报错并指出是哪一份，而不是 `&` 之后悄悄补缺失值。
    """
    if not masks:
        raise ValueError("至少要给一份掩码")

    reference = masks[0]
    if not isinstance(reference, pd.DataFrame):
        raise ValueError(f"掩码必须是 DataFrame，第 1 份是 {type(reference).__name__}")
    for position, mask in enumerate(masks[1:], start=2):
        if not isinstance(mask, pd.DataFrame):
            raise ValueError(f"掩码必须是 DataFrame，第 {position} 份是 {type(mask).__name__}")
        if not mask.index.equals(reference.index) or not mask.columns.equals(reference.columns):
            raise ValueError(
                f"第 {position} 份掩码的日期与标的必须与第 1 份一致，否则取与会静默错位"
            )

    combined = reference.astype(bool)
    for mask in masks[1:]:
        combined &= mask.astype(bool)
    return combined


def build_universe(
    markets,
    *,
    rules: UniverseRules | None = None,
    rule_table: RuleTable | None = None,
) -> pd.DataFrame:
    """由一组已质检的行情算出股票池。

    参数:
        markets: 一组 :class:`~mbt.data.market.MarketData`。只用到 ``symbol`` 与 ``prices``
            ——品种由符号判定，根数由价格表数出，故不需要权息事件。
        rules: 准入规则，默认 :class:`UniverseRules` 的出厂设定。
        rule_table: 规则表，用于查 **ST 期间**。``None`` 时不判 ST。请注意出厂表不登记
            任何 ST 期间，故默认情形下 ST 排除**不生效**（见模块说明）。

    返回:
        boolean 标的宽表（ADR-0009）：索引是各标的交易日的**并集**（与引擎时钟一致），
        列是标的，值表示该标的在该日是否在池。

    .. note::

        ``min_bars`` 度量的是**本地行情可得量**，不是真实上市日。对本机数据窗口
        （`sh600000` 自 1999-11-10 起，各标的起止不一）之前上市的老股，其首根即窗口起点，
        故 60 日门槛永不触发。结果正确（它们确实不是次新），但它不等于「自 IPO 起已满
        60 日」——真实上市日需要数据源提供带日期的上市信息，一期不含。
    """
    rules = rules or UniverseRules()

    if not markets:
        return pd.DataFrame()

    closes = pd.DataFrame({market.symbol: market.prices["close"] for market in markets})

    # 截至当日的累计 K 线根数（含当日）。用累计而非整段长度，是时点正确性的落点。
    bars_so_far = closes.notna().cumsum()
    in_universe = bars_so_far >= rules.min_bars

    for market in markets:
        symbol = market.symbol
        if not is_stock(symbol):
            in_universe[symbol] = False
            continue

        try:
            board = _BOARD_TO_CONCEPT[board_of(symbol)]
        except KeyError:
            # 规则层给出了词汇表里没有的板块名——两者命名漂移了。此时**报错而不猜**：
            # 猜错会把不该收的标的收进截面（错得无声），而报错会立刻被人看见。
            raise ValueError(
                f"规则层给出板块 {board_of(symbol)!r}，而词汇表只有 "
                f"{sorted(ALL_BOARDS)}——两处命名已漂移，不猜"
            ) from None

        if board not in rules.boards:
            in_universe[symbol] = False
            continue

        if rules.extra_mask is not None:
            if symbol not in rules.extra_mask.columns:
                # 掩码里没有这个标的——它没被评估过，故不放行（与「没有可用财报就排除」同一方向）。
                in_universe[symbol] = False
                continue
            for on in in_universe.index:
                if not bool(rules.extra_mask.at[on, symbol]):
                    in_universe.loc[on, symbol] = False

        if rule_table is not None and rule_table.has_st_period(symbol):
            for on in in_universe.index:
                if rule_table.is_st(symbol, on.date()):
                    in_universe.loc[on, symbol] = False

    return in_universe
