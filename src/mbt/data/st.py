"""从价格数据**推断 ST 期间**（票据 #53）。

## 为什么需要推断

ST 股的涨跌幅限制是 5%（主板），而普通主板股是 10%。本地数据**没有股票名称**，出厂规则表
按 ADR-0002 的立场**刻意不登记任何 ST 期间**——于是引擎会给一只 5% 的股票按 10% 撮合，
**把 ST 的一字板当成可成交**。这是「不报错、只是让结论失真」那类缺陷里最贵的一种。

## 判据（以及为什么单看一条不行）

`ST` ⟺ 涨跌幅被 **5% 封顶**。于是窗口内应当满足两条**同时**成立：

1. **最大单日幅度 ≤ 5.4%** —— 它被封顶了；
2. **至少 2 天恰好落在 ±5% 限价上** —— 它确实在打 5% 的板，而不是**只是没波动**。

**第 2 条是必需的。** 只用第 1 条会把安静的大盘股全判成 ST：实测全市场主板股票里
**87.9%** 都有某个 120 日窗口「最大幅度 ≤ 5.5%」，包括工商银行（1.97%）。而「恰好打到
5% 限价」这件事，安静股不会发生（实测工行、茅台、平安、浦发四只都命中 0 天）。

反过来，**只看第 2 条也不行**：实测 79.5% 的主板股票都有 ≥3 天「收盘恰落在 ±5% 限价上」
——因为「恰好涨跌 5.00%」本身常见（±10% 的带里，落在 5.00% 是个正常事件）。故两条必须合用。

## 已知漏检（必须登记）

- **创业板 / 科创板看不到**：那里的 ST 限幅与普通股同为 20%，价格里没有区分信号。本模块对
  这两个板块返回空元组——那是「**判不了**」，不是「判为非 ST」。
- **跨度不足 30 个交易日的期间被丢弃**（见 :data:`MIN_SPAN_DAYS`）。这是**必需的取舍**：
  不丢的话，实测有 **17.2%** 的合规标的会因「限幅收紧后出现越界」被整只拒收，而误杀它们的
  正是那些跨度只有几天到几周的巧合。代价是真 ST 期间两端的零星碎片会被切掉。
- **一字板之外的封顶日**：ST 股多数日子不会打到限价，故判据依赖「窗口内**有**打板日」；
  一段 ST 期里若完全没有打板日，会漏判。
- 推断出的期间是**下界**（只覆盖有证据的日子），不是精确区间。
- **短期间有假阳性**：这是本判据的固有性质，靠 :data:`MIN_SPAN_DAYS` 压制，不是消除。
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass

import pandas as pd

from ..rules import RuleTable, limit_price, same_price

#: 判「被 5% 封顶」的容差。限价按分取整，故低价股的百分比会略偏离 5.00%。
CEILING_TOLERANCE = 0.054

#: 窗口内至少要打到几次 ±5% 限价——用来排除「只是安静」的股票。
MIN_LIMIT_HITS = 2

#: 滚动窗口的交易日数。**这是个取舍**：窗口越长越稳（少误判），但越容易漏掉短暂的 ST 期间。
DEFAULT_WINDOW = 40

#: 断段：两段证据相隔超过这么多交易日就认为是两个独立的 ST 期间。
GAP_TOLERANCE = 10

#: 跨度用「首个证据日 → 末个证据日」算。**满足其一即采信**——见 :data:`ONGOING_TOLERANCE`。
MIN_SPAN_DAYS = 30

#: 「仍在 ST」的容差（交易日）：末个证据日距数据末端不超过它，就认为该期间**尚未结束**。
#:
#: **为什么要与 :data:`MIN_SPAN_DAYS` 取「或」**：刚开始的 ST 天然只有少量证据。实测
#: ``sz002731`` 的证据只跨 **15** 个交易日（2026-04-09~04-30），而它确实是 ST——它的证据
#: 在 4 月底就断了，是因为此后 40 日窗口里不再有 ≥2 根打板（见模块文档的漏检说明），
#: **不是**因为 ST 结束了。而真正的假阳性（``sh600106`` 跨度 7、``sh600100`` 跨度 21）
#: 的末个证据日距数据末端有 **264 / 382** 个交易日——它们早已结束。
#:
#: 故判据是「**跨度够长**（长期 ST）**或** **距末端够近**（近期发作且仍在）」。
#: 这也正是本地数据能提供的两种可信情形。
ONGOING_TOLERANCE = 80


@dataclass(frozen=True)
class InferredStPeriod:
    """推断出的一个 ST 期间。``end`` 为 ``None`` 表示延续到数据末端。"""

    start: dt.date
    end: dt.date | None
    evidence_days: int

    def covers(self, on: dt.date) -> bool:
        return self.start <= on and (self.end is None or on <= self.end)


def infer_st_periods(
    prices: pd.DataFrame,
    symbol: str,
    rules: RuleTable,
    *,
    window: int = DEFAULT_WINDOW,
) -> tuple[InferredStPeriod, ...]:
    """由**完整**价格历史推断 ST 期间（判据见模块文档）。

    只对**限幅 10% 的板块**（沪/深主板）做——创业板与科创板的 ST 限幅与普通股相同，
    价格里没有信号，故直接返回空（**不是**「判为非 ST」，而是「判不了」）。

    参数:
        prices: 字段宽表，须含 ``close``，索引升序。
        symbol: 标的符号，用于查板块与限幅。
        rules: 规则表（提供板块与限幅）。
        window: 滚动窗口的交易日数。

    返回:
        推断出的 ST 期间（按时间升序）。可能为空。
    """
    if "close" not in prices.columns:
        raise ValueError("价格表必须含 close 列：ST 判据要用收盘价算单日幅度")
    if len(prices) < window + 1:
        return ()

    try:
        if rules.price_limit(rules.board_of(symbol), prices.index[-1].date()) != 0.10:
            # 非 10% 限幅的板块（创业板/科创板/北交所）：判不了，不是判为非 ST。
            return ()
    except Exception:  # noqa: BLE001
        return ()

    close = prices["close"]
    previous = close.shift(1)
    change = (close / previous - 1.0).abs()

    on_limit = pd.Series(False, index=close.index)
    for stamp, value in close.items():
        base = previous.get(stamp)
        if base is None or base != base or base <= 0:
            continue
        price = float(value)
        base_price = float(base)
        on_limit.at[stamp] = same_price(price, limit_price(base_price, 0.05, 1)) or same_price(
            price, limit_price(base_price, 0.05, -1)
        )

    rolling_ceiling = change.rolling(window, min_periods=window).max()
    rolling_hits = on_limit.rolling(window, min_periods=window).sum()
    candidate = (rolling_ceiling <= CEILING_TOLERANCE) & (rolling_hits >= MIN_LIMIT_HITS)

    days = list(candidate[candidate].index)
    if not days:
        return ()

    periods: list[InferredStPeriod] = []
    begin = previous_day = days[0]
    count = 1
    calendar = close.index

    def finish(first, last, evidence):
        """收尾一个期间；**两个可信情形都不满足**则返回 ``None``（判为巧合）。

        判据是「跨度够长」**或**「距末端够近」——理由见 :data:`ONGOING_TOLERANCE`。
        """
        span = int(calendar.searchsorted(last) - calendar.searchsorted(first))
        tail = len(calendar) - 1 - int(calendar.searchsorted(last))
        if span < MIN_SPAN_DAYS and tail > ONGOING_TOLERANCE:
            return None
        end = None if tail <= window else last.date()
        return InferredStPeriod(first.date(), end, evidence)

    for day in days[1:]:
        gap = int(calendar.searchsorted(day) - calendar.searchsorted(previous_day))
        if gap > GAP_TOLERANCE:
            period = finish(begin, previous_day, count)
            if period is not None:
                periods.append(period)
            begin, count = day, 1
        else:
            count += 1
        previous_day = day
    # 最后一段：若延伸到数据末端附近，视为**仍在 ST**（end=None），否则按已结束处理。
    period = finish(begin, previous_day, count)
    if period is not None:
        periods.append(period)
    return tuple(periods)
