"""行情数据的异常检测：把「越出涨跌停带的跳空」区分为**公司行为**与**坏数据**（ADR-0005）。

## 为什么不能比「日间涨跌幅比率」

实测（``docs/research/tdx-halt-and-limit-representation.md`` 结论五）：本机 16 只主板
股票的 43,982 个相邻交易日里，日间变动超过 10% 的有 460 次，其中 **444 次合法**——含
四舍五入效应（``11.26 × 1.1 = 12.386`` 进位到 12.39，涨 10.04%）与「收在涨停但盘中
未锁死」。照字面「涨跌幅超过限幅即报错」会把正常交易日全数误报。

故判据是**比限价带**（由前收盘价与当日限幅算出的那两分钱），且用 :func:`mbt.rules.limit_band`
——与撮合层共用同一个算法，不会出现「撮合认为没触限、质检却报异常」。

## 归因：用权息事件解释跳空，而不是见到事件就豁免

除权除息日的限幅以**除权除息参考价**为基数，故那一天的合法价格区间会整体下移。本模块：

1. 取**前一根 K 线之后、当根 K 线之前（含当根）**的全部除权除息事件，逐个折算参考价
   （同日多条求和、跨日逐个链式取整），得到当日的限幅基数；
2. 用该基数算限价带，若价格仍在带内 → 跳空**被公司行为解释**，不是异常；
3. 若**没有**事件而价格越界，或**有**事件但按事件重算的带仍被越出 → 报异常。

第 1 步的「跨 K 线」很关键：长期停牌期间的权息事件在复牌日才体现为跳空，只查当根 K 线
会漏掉它们（实测 5 例长期停牌复牌日全靠这一步才归因正确）。

## 已知的误报源：制度空窗，而非判据错误

全市场实测（9,720 个文件、1,135 万根 K 线）检出 761 处越界，**没有一处**是「无制度空窗
可解释」的坏数据。761 处全部落在下面这些制度空窗内：

- **序列前 5 根**（601 处）：新股上市初期不设涨跌幅；
- **北交所**（76 处，集中在 2021–2023）：本地 ``bj/`` 混存北交所与新三板，``920xxx`` 的历史
  回溯到上市前的新三板挂牌期——那段时期只有 ±30%/±60% 的**盘中临时停牌**，没有硬性涨跌幅
  限制；
- **其余约 82 处**：股改复牌首日不设涨跌幅（上证交字〔2005〕7 号）、重大资产重组/借壳复牌、
  退市整理期，以及缩股/债转股等**非除权除息类**公司行为——``gbbq`` 里属类别 2 及以后，按
  ADR-0003 不参与价格复权，故本模块看不到它们。

不建模的理由与规则表一致：本模块拿不到「那一天这只股票处于哪种制度之下」（本地数据无上市
日、无退市日、无重组停复牌标记、无「G」标记），而用停牌天数或价格幅度之类的代理变量去猜，
正是本项目一贯拒绝的模糊判据。

**因此本检查的定位是「报错让人看」，不是无人值守的批量过滤**：它不放过坏数据（限价若算错
会在除权日与涨跌停日大面积误报，而实测误报密度仅 0.0067%），但会把制度空窗误报成坏数据。
全市场选股应在调用处捕获 :class:`~mbt.data.errors.MarketDataError` 并把该标的记为**存疑**、
再人工抽查。逐条数字与分类见 ``docs/research/tdx-halt-and-limit-representation.md`` 结论五的补记。
"""

from __future__ import annotations

import datetime as dt
from bisect import bisect_right
from dataclasses import dataclass

from ..rules import PRICE_TOLERANCE, RuleTable, limit_band, round_to_cent
from .adjust import AdjustmentEvent, combined_reference_price
from .errors import MarketDataError

#: 检测所需的列。``close`` 用于取前收盘价，``high`` / ``low`` 是受限幅约束的盘中极值。
_REQUIRED_COLUMNS = ("high", "low", "close")


@dataclass(frozen=True)
class Anomaly:
    """一根越出当日涨跌停带的 K 线。

    属性:
        date: 当根 K 线的日期。
        prev_date: 前一根 K 线的日期。它与 ``date`` 之间可能隔着停牌。
        prev_close: 前一根 K 线的收盘价，即限幅的初始基数。
        low / high: 当根 K 线的盘中最低 / 最高价。
        lower / upper: 当日限价带，**已按区间内的权息事件重算基数**。
        events: 前一根 K 线之后、当根 K 线之前（含当根）的除权除息事件。
    """

    date: dt.date
    prev_date: dt.date
    prev_close: float
    low: float
    high: float
    lower: float
    upper: float
    events: tuple[AdjustmentEvent, ...] = ()

    @property
    def kind(self) -> str:
        """异常的性质，用于区分「有没有公司行为可解释」。"""
        return "beyond_event" if self.events else "unexplained"

    def describe(self) -> str:
        """一句可读的说明。"""
        if self.events:
            why = f"区间内有 {len(self.events)} 条除权除息事件，但按其重算的带仍被越出"
        else:
            why = "区间内无权息事件，跳空无公司行为可解释"
        parts = []
        if self.low < self.lower - PRICE_TOLERANCE:
            parts.append(f"最低 {self.low} 低于跌停价 {self.lower}")
        if self.high > self.upper + PRICE_TOLERANCE:
            parts.append(f"最高 {self.high} 高于涨停价 {self.upper}")
        return (
            f"{self.date.isoformat()}：{'；'.join(parts)}"
            f"（基数 {self.prev_close}@{self.prev_date.isoformat()} 经权息折算后为"
            f"参考带 [{self.lower}, {self.upper}]）；{why}"
        )


def find_anomalies(prices, symbol: str, rules: RuleTable, events=()) -> list[Anomaly]:
    """找出价格表中所有**越出当日涨跌停带**且**无公司行为可解释**的 K 线。

    参数:
        prices: 原始价宽表，须含 ``high`` / ``low`` / ``close``，索引为升序日期索引。
        symbol: 标的符号，用于按板块与 ST 状态取限幅。
        rules: 规则表。
        events: 该标的的除权除息事件（如 ``GbbqDataSource.events(symbol)``）。
            **应传全量**——本函数自行按区间取用，传入时不必也不能预筛。

    返回:
        按日期升序的异常列表；无异常则为空列表。

    抛:
        MarketDataError: 价格非正（无法作为限幅基数，且不可能是合法价格）。
    """
    _require(prices)
    _reject_non_positive(prices, symbol)

    grouped: dict[dt.date, list[AdjustmentEvent]] = {}
    for event in events:
        grouped.setdefault(event.ex_date, []).append(event)
    event_dates = sorted(grouped)

    index = prices.index
    high = prices["high"].to_numpy(dtype="float64")
    low = prices["low"].to_numpy(dtype="float64")
    close = prices["close"].to_numpy(dtype="float64")

    found = []
    for i in range(1, len(index)):
        on = index[i].date()
        prev_date = index[i - 1].date()

        # 限幅基数是「前收盘价经区间内权息事件折算后的参考价」。首个事件之前无成交，
        # 故跨日的多条事件必须**链式**折算（每一步的基数都不同），同日多条才可求和。
        base = float(close[i - 1])
        events_in_gap = []
        for k in range(bisect_right(event_dates, prev_date), bisect_right(event_dates, on)):
            day_events = grouped[event_dates[k]]
            base = round_to_cent(combined_reference_price(base, day_events))
            events_in_gap.extend(day_events)

        lower, upper = limit_band(base, rules.limit_for(symbol, on))
        if high[i] > upper + PRICE_TOLERANCE or low[i] < lower - PRICE_TOLERANCE:
            found.append(
                Anomaly(
                    date=on,
                    prev_date=prev_date,
                    prev_close=float(close[i - 1]),
                    low=float(low[i]),
                    high=float(high[i]),
                    lower=lower,
                    upper=upper,
                    events=tuple(events_in_gap),
                )
            )
    return found


#: 报错信息里最多列出几条异常——余下只报数量，避免价格表整体错位时刷屏。
_MAX_REPORTED = 5


def require_no_anomalies(prices, symbol: str, rules: RuleTable, events=()) -> None:
    """价格表若含无法解释的越界跳空则**报错**，否则返回 ``None``。

    这是缺口纪律（ADR-0005）在本层的落点：坏数据必须让回测停下来，而不是静默算出一
    个看似合理的收益。回测因此不再「沉默接受」坏数据。
    """
    anomalies = find_anomalies(prices, symbol, rules, events)
    if not anomalies:
        return
    lines = [f"  - {a.describe()}" for a in anomalies[:_MAX_REPORTED]]
    more = len(anomalies) - _MAX_REPORTED
    if more > 0:
        lines.append(f"  - …另有 {more} 处")
    raise MarketDataError(
        f"{symbol} 的行情数据有 {len(anomalies)} 处越出涨跌停带且无公司行为可解释，"
        f"按缺口纪律不予回测（ADR-0005）：\n" + "\n".join(lines)
    )


def _require(prices) -> None:
    missing = [column for column in _REQUIRED_COLUMNS if column not in prices.columns]
    if missing:
        raise ValueError(f"价格表缺少列 {missing}：判限价带需要 high / low / close")
    if not prices.index.is_monotonic_increasing:
        raise ValueError("价格表的日期索引必须升序——限价带按前一根 K 线定位，乱序会算错")


def _reject_non_positive(prices, symbol: str) -> None:
    for column in _REQUIRED_COLUMNS:
        values = prices[column]
        if (values <= 0).any():
            bad = values[values <= 0]
            raise MarketDataError(
                f"{symbol} 在 {bad.index[0].date().isoformat()} 的 {column} 为 "
                f"{float(bad.iloc[0])}，非正价格不能作为限幅基数，也不可能是合法成交价"
            )
