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

from mbt.data.instrument import is_stock
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

        if rule_table is not None and rule_table.has_st_period(symbol):
            for on in in_universe.index:
                if rule_table.is_st(symbol, on.date()):
                    in_universe.loc[on, symbol] = False

    return in_universe
