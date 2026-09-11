"""判别类别 1 事件是否**真的**造成了价格稀释（票据 #20）。

`gbbq` 把某些**不改变总股本**的公司行为（典型是 2005–2007 年**股权分置改革**的**对价送股**
——它在股东之间转移存量股份）也记为**类别 1（除权除息）**。而价格并未因此稀释：

```
sh600000  2006-03-20 → 2006-05-12（中间停牌约 35 个交易日）
  原始价              −5.99%              正常波动
  若按 10送3 稀释     参考价应为 8.35     实际收在 10.21
  后复权（旧行为）     +22.22%             ← 凭空插入的假跳空
```

ADR-0003 说「复权仅由除权除息类事件驱动」，而本模块把这句话补全为：**类别 1 只是必要条件，
不是充分条件**——还须该事件确实稀释了价格。否则复权会把整条后续序列乘上一个不该有的倍数，
污染均线、动量与所有收益数字，且**不会报错**。

## 判据：比「两个假说各自的涨跌停带」

对每个含稀释成分的事件，取事件当日（或停牌后的复牌首根）的收盘价，分别与两个假说的
**涨跌停带**比较：

- **未稀释**假说：带由**前收盘价**算出；
- **已稀释**假说：带由**除权除息参考价**算出。

落在哪个带里就是哪个假说。**带宽取自规则表**（``limit_for``，按板块与成交日）而非常数——
创业板 20%、北交所 30% 与主板 10% 不同，用一个常数必然在别的板块上判错。

三种结局：

- **只有一个带落进** → 就是它，判定可信。
- **两个带都落进** → 判据**结构性地**没有更多信息可用（宽限幅板块上「25% 的真实稀释」与
  「−28% 的正常大跌」在价格里是同一件事）。按**就近**取一个并标低置信
  （:attr:`~DilutionVerdict.low_confidence`）。实测约 10% 的含事件标的落这一档。
- **两个带都落不进** → 这个跳空**既不是**正常波动、**也不是**该事件造成的（多半是长期停牌
  复牌的大缺口），**报错**。实测约 5%。

规则表不覆盖该日期时（1996-12-16 之前、或非股票）`limit_for` 直接抛 ``RuleTableError``——
**不猜一个常数**，与上面「报错」同一立场。实测约 6%（都在 1996-12-16 之前）。

## 为什么不用其它判据

- **股本字段（类别 2 的某两列相等与否）**：实测 48% 的**真实**稀释事件也两列相等（假阳性），
  且只覆盖约 30% 的事件（大多数事件根本没有同日股本记录）。**不成立。**
- **按年代一刀切**（如「2005–2007 的送转一律不复权」）：实测该年代 88% 的稀释事件是
  **真实**的，一刀切会误伤绝大多数。**明确排除。**

## 判定只看事件当日及之前

实际比值用的是「事件日（或复牌首根）收盘 ÷ 其前一根收盘」，两者都不晚于事件日；而
``as_of`` 过滤仍由 :func:`mbt.data.adjust.adjustment_factors` 负责。故本模块**不引入前视**
（ADR-0006），这一点有测试钉住。
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import pandas as pd

from ..rules import RuleTable, limit_band, round_to_cent
from .adjust import AdjustmentEvent, combined_reference_price
from .errors import MarketDataError

#: 低于此稀释幅度的事件**不判**：真实事件与噪声在如此小的落差上无法分辨，而误判的代价
#: 也只有这个量级。超过它的稀释一旦判错就是 10% 以上，故只在那个量级上设防。
DILUTION_THRESHOLD = 0.25

#: 判定结论。
DILUTED = "diluted"
NOT_DILUTED = "not_diluted"
BELOW_THRESHOLD = "below_threshold"
OUTSIDE_SERIES = "outside_series"


@dataclass(frozen=True)
class DilutionVerdict:
    """一条（或同日一组）事件的判定记录。

    属性:
        ex_date: 除权除息日。
        dilution: 送转 + 配股的合计比例。
        actual_ratio: 实际比值 ``收盘 ÷ 前收盘``。
        diluted_ratio: **已稀释**假说下的理论比值（参考价 ÷ 前收盘）。
        verdict: :data:`DILUTED` / :data:`NOT_DILUTED` / :data:`BELOW_THRESHOLD`
            / :data:`OUTSIDE_SERIES`。
        low_confidence: 两个假说的带**重叠**时判据无法分辨，此时按「就近假说」取一个，
            并在此标出。宽限幅板块（创业板/科创板 20%、北交所 30%）上必然出现这种情形——
            「25% 的真实稀释」与「−28% 的正常大跌」在价格里是同一件事，没有更多信息可用。
    """

    ex_date: object
    dilution: float
    actual_ratio: float
    diluted_ratio: float
    verdict: str
    low_confidence: bool = False


def resolve_dilution(
    prices: pd.DataFrame,
    events,
    symbol: str,
    rules: RuleTable,
) -> tuple[tuple[AdjustmentEvent, ...], tuple[DilutionVerdict, ...]]:
    """判定每条事件是否真的稀释，返回（**参与复权**的事件集, 判定记录）。

    判为**未稀释**的事件只丢掉它的**稀释成分**（送转与配股置零），**保留现金分红**——
    股改对价送股不稀释价格，但那一天发的现金分红是真的（见 Q3 的既定决定）。

    参数:
        prices: 原始价字段宽表，须含 ``close``，索引升序。
        events: :class:`AdjustmentEvent` 序列（通常来自 ``GbbqDataSource.events``）。
        symbol: 标的符号，用于按板块取限幅。
        rules: 规则表，提供当日限幅。

    返回:
        ``(参与复权的事件, 判定记录)``。事件顺序与输入一致。

    抛:
        MarketDataError: 某条事件的两个假说带**都落不进**——那个跳空既非正常波动、亦非该事件
            所致。消息给出标的、事件日、实际比值与两个假说各自的期望比值，便于判断该显式
            切片还是把该标的记为存疑。
        RuleTableError: 该事件日期不在规则表覆盖范围内（1996-12-16 之前等），无从取当日带宽。
    """
    if "close" not in prices.columns:
        raise ValueError("价格表必须含 close 列：判定要用事件日前后的收盘价")

    index = prices.index
    closes = prices["close"].to_numpy(dtype="float64")

    ordered = sorted(events, key=lambda event: (event.ex_date, event.bonus_per_10))
    effective: list[AdjustmentEvent] = []
    verdicts: list[DilutionVerdict] = []

    for group in _group_by_date(ordered):
        dilution = sum(e.bonus_per_share + e.rights_per_share for e in group)

        if dilution < DILUTION_THRESHOLD:
            effective.extend(group)
            verdicts.append(_record(group[0], dilution, verdict=BELOW_THRESHOLD))
            continue

        position = int(index.searchsorted(pd.Timestamp(group[0].ex_date), side="left"))
        if position == 0 or position >= len(index):
            # 事件落在序列之外——复权本就会忽略它，判定也无从谈起。
            effective.extend(group)
            verdicts.append(_record(group[0], dilution, verdict=OUTSIDE_SERIES))
            continue

        prev_close = float(closes[position - 1])
        actual = float(closes[position])
        on = index[position].date()
        limit = rules.limit_for(symbol, on)

        # 两个假说各自的带。宽度按板块与成交日取自规则表，不用常数。
        plain_low, plain_high = limit_band(prev_close, limit)
        reference = round_to_cent(combined_reference_price(prev_close, group))
        diluted_low, diluted_high = limit_band(reference, limit)

        in_plain = plain_low <= actual <= plain_high
        in_diluted = diluted_low <= actual <= diluted_high
        actual_ratio = actual / prev_close
        diluted_ratio = reference / prev_close

        if in_plain and not in_diluted:
            effective.extend(_without_dilution(group))
            verdicts.append(
                _record(group[0], dilution, actual_ratio, diluted_ratio, verdict=NOT_DILUTED)
            )
            continue

        if in_diluted and not in_plain:
            effective.extend(group)
            verdicts.append(
                _record(group[0], dilution, actual_ratio, diluted_ratio, verdict=DILUTED)
            )
            continue

        if in_plain and in_diluted:
            # 两个假说都成立——判据在这里**结构性地**没有更多信息可用（宽限幅板块上，
            # 「真实稀释」与「正常大跌」在价格里是同一件事）。按**就近**取一个，并标为低置信。
            closer_to_diluted = abs(actual_ratio - diluted_ratio) < abs(actual_ratio - 1.0)
            if closer_to_diluted:
                effective.extend(group)
            else:
                effective.extend(_without_dilution(group))
            verdicts.append(
                _record(
                    group[0],
                    dilution,
                    actual_ratio,
                    diluted_ratio,
                    verdict=DILUTED if closer_to_diluted else NOT_DILUTED,
                    low_confidence=True,
                )
            )
            continue

        # 两个带的落不进：这个价格跳空**既不是**正常波动、**也不是**该事件造成的。
        # 它多半是长期停牌复牌后的一次大缺口——那是另一回事，不该被假装解释掉。
        raise MarketDataError(
            f"{symbol} 在 {group[0].ex_date.isoformat()} 的除权事件判不动"
            f"（两个假说的带都落不进）："
            f"实际比值 {actual_ratio:.4f}（收盘 {actual:g} ÷ 前收 {prev_close:g}）；"
            f"未稀释假说期望 {1.0:.4f}（带 [{plain_low:g}, {plain_high:g}]）；"
            f"已稀释假说期望 {diluted_ratio:.4f}（参考价 {reference:g}，"
            f"带 [{diluted_low:g}, {diluted_high:g}]）。"
            f"该跳空既非正常波动、亦非该事件所致，说不清就不猜——"
            f"请显式切片避开该区间，或把该标的记为存疑。"
        )

    return tuple(effective), tuple(verdicts)


def _group_by_date(events) -> list[list[AdjustmentEvent]]:
    """同日事件合为一组——参考价必须把同日各量求和后**一次**代入（同日多条不能逐个套用）。"""
    groups: list[list[AdjustmentEvent]] = []
    for event in events:
        if groups and groups[-1][0].ex_date == event.ex_date:
            groups[-1].append(event)
        else:
            groups.append([event])
    return groups


def _without_dilution(group) -> list[AdjustmentEvent]:
    """丢掉稀释成分、保留现金分红。"""
    return [
        replace(event, bonus_per_10=0.0, rights_per_10=0.0, rights_price=0.0) for event in group
    ]


def _record(
    event,
    dilution: float,
    actual_ratio: float = float("nan"),
    diluted_ratio: float = float("nan"),
    *,
    verdict: str,
    low_confidence: bool = False,
) -> DilutionVerdict:
    return DilutionVerdict(
        ex_date=event.ex_date,
        dilution=dilution,
        actual_ratio=actual_ratio,
        diluted_ratio=diluted_ratio,
        verdict=verdict,
        low_confidence=low_confidence,
    )
