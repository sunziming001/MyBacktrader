"""估值信号：动态PE、PE 百分位、PEG（票据 #37）。

把「收盘价」与「按期财务」合成 **date × 标的** 形状的估值序列，供选股规则与策略共用
（ADR-0001：同一条件不必写两遍）。本模块与 :mod:`mbt.data.fundamental` 同属数据层——财务
派生的信号放这里而非 ``mbt.signals``，先例是 ``non_loss_mask``：那是纯 pandas 运算，但它
要读财务数据，而 ``mbt.signals`` 是「无 I/O 无状态」的纯函数库。

## 三个口径（都已钉死；改动会改变每一个数字）

**动态PE = 收盘价 ÷ 年化每股收益**，年化系数按**报告期**取（通达信口径）::

    报告期   03-31   06-30   09-30   12-31
    系数         4       2     4/3       1

刻意用「累计 EPS × 系数」而不是 PE(TTM)：前者就是行情软件里显示的「动态市盈率」，与
需求口径一致。代价是 **Q1 × 4 会放大季节性**——一家收入集中在四季度的公司，用一季报年化会
高估其估值。

**PE 百分位 = (PE − LLV(PE, w)) / (HHV(PE, w) − LLV(PE, w))**，``w`` 默认 1000 个交易日。
``LLV``/``HHV`` 取**含当日**的窗口，且**不足 w 根时按可得历史算**（通达信语义）。故次新股的
「百分位」是短窗口上的百分位，其含义随上市时间漂移——这是刻意保留的口径，不是疏漏。

.. warning::

    **非正的 PE 也参与 LLV/HHV**（公式原样，与通达信一致），这会让百分位失真：某公司有过
    亏损季度时 PE 曾为负，于是 ``LLV`` 是个负数，此后即使 PE 只是 20（对一家持续盈利的公司
    算便宜），百分位也会被抬高。例：``LLV=-50、HHV=40、PE=20`` → 百分位 ``0.78``。

    之所以照原样实现：需求方明确选了「公式原样（通达信语义）」，而「只让正 PE 参与」是本项目
    无法替你决定的口径取舍。若要去掉这个失真，改 :func:`pe_percentile` 的入参即可（先把
    非正值置为缺失）——那会变成另一个口径，需要另行登记。

**PEG = 动态PE ÷ 盈利增长率(%)**，增长率是**归母净利润累计同比**::

    (本期累计 − 去年同期累计) / |去年同期累计| × 100

用累计同比而非单季是为了避开季节性；分母取**绝对值**，则「由亏转盈」这类负基数情形的符号
仍有意义（否则负基数会让「增长」的符号反过来）。

## 时点正确性（ADR-0006）

一切按**公告日**对齐：某交易日的估值用当天**已公告**的最近一期财报，而不是「报告期落在当天
之前」的那一期。故 4 月底之前用不到一季报——即使它的报告期是 3 月 31 日。

:attr:`~mbt.data.fundamental.FinancialRecord.usable` 那道门（公告日 > 报告期、且年份 ≥ 2005）
在这里同样生效：不可用的记录**不进序列**，而不是给它编一个日期。

## 有效起点

财务数据 2005 年前不可用，加上 ``w`` 个交易日的回看，能算出百分位的最早日子在 2010 前后。
这不是本模块的限制，是数据事实——**回测起点因此被财务数据决定，而不是被行情数据决定**。
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass

import numpy as np
import pandas as pd

from .errors import MarketDataError

#: 动态PE 字段名。
PE = "pe"
#: PE 百分位字段名。
PE_PERCENTILE = "pe_percentile"
#: PEG 字段名。
PEG = "peg"

#: 本模块产出的字段，按构造顺序。给调用方**显式声明**用（与 ``SCREEN_FIELDS`` 同一精神）。
VALUATION_FIELDS = (PE, PE_PERCENTILE, PEG)

#: 报告期月份 → 年化系数。只认四个季末；别的月份说明报告期不是季末，**报错而不猜**。
_ANNUALISATION = {3: 4.0, 6: 2.0, 9: 4.0 / 3.0, 12: 1.0}

#: 百分位的默认窗口（交易日）——约 4 年。
DEFAULT_WINDOW = 1000


def annualisation_factor(report_period: dt.date) -> float:
    """报告期对应的年化系数（见模块文档的表）。

    抛:
        MarketDataError: 报告期不在四个季末。**不猜**一个系数——猜错会让 PE 整体偏一个倍数，
            而那种错误不会报错。
    """
    try:
        return _ANNUALISATION[report_period.month]
    except KeyError:
        raise MarketDataError(
            f"报告期 {report_period.isoformat()} 不是季末，无从取年化系数。"
            f"本模块只认 {sorted(_ANNUALISATION)} 四个月份"
        ) from None


def price_earnings_ratio(close: pd.DataFrame, earnings: pd.DataFrame) -> pd.DataFrame:
    """``PE = 收盘价 ÷ 年化每股收益``。两者任一缺失处即缺失。

    每股收益为 0 处也判为缺失（除零无意义），而**负值照常算出负 PE**——「PE > 0」是过滤条件
    的事，不是计算的事（少一个隐含分支，就少一处会漂移的地方）。
    """
    _require_same_shape(close, earnings, "收盘价", "年化每股收益")
    safe = earnings.where(earnings != 0)
    return close / safe


def pe_percentile(pe: pd.DataFrame, window: int = DEFAULT_WINDOW) -> pd.DataFrame:
    """PE 在最近 ``window`` 个交易日里的 min-max 归一化位置，取值 ``[0, 1]``。

    含当日，且不足 ``window`` 根时按**可得历史**算（通达信 ``LLV``/``HHV`` 语义）。窗口内
    ``HHV == LLV``（PE 恒定）时**返回缺失**而不是 0 或 1——那种日子「百分位」没有定义，硬给
    一个数会让过滤条件凭空通过或凭空拦下。

    非正 PE 照常参与 ``LLV``/``HHV``，其失真见模块文档的 warning。
    """
    if window < 1:
        raise ValueError(f"窗口至少为 1，收到 {window}")

    rolling = pe.rolling(window, min_periods=1)
    lowest = rolling.min()
    highest = rolling.max()
    span = highest - lowest
    return ((pe - lowest) / span.where(span != 0)).clip(lower=0.0, upper=1.0)


def price_earnings_growth(pe: pd.DataFrame, growth_pct: pd.DataFrame) -> pd.DataFrame:
    """``PEG = PE ÷ 盈利增长率(%)``。增长率为 0 处缺失（除零无意义）。

    增长率是**百分数**（30 表示 30%），故 PEG 的量纲与常用读数一致（PE 15、增长 30 → 0.5）。
    """
    _require_same_shape(pe, growth_pct, "动态PE", "盈利增长率")
    safe = growth_pct.where(growth_pct != 0)
    return pe / safe


@dataclass(frozen=True)
class Valuation:
    """三个估值序列（都是 **date × 标的** 的标的宽表，与价格同日对齐）。

    属性:
        pe: 动态PE。
        pe_percentile: PE 百分位，``[0, 1]``。
        peg: PEG。
    """

    pe: pd.DataFrame
    pe_percentile: pd.DataFrame
    peg: pd.DataFrame

    def as_fields(self) -> dict[str, pd.DataFrame]:
        """字段名 → 标的宽表，供装进 :class:`~mbt.data.panel.Panel` 或交给策略。"""
        return {PE: self.pe, PE_PERCENTILE: self.pe_percentile, PEG: self.peg}


def build_valuation(
    close: pd.DataFrame,
    financials,
    *,
    window: int = DEFAULT_WINDOW,
) -> Valuation:
    """由收盘价与按期财务算出三个估值序列。

    参数:
        close: **date × 标的** 的收盘价标的宽表。索引须升序，列即标的。
        financials: :class:`~mbt.data.fundamental.CwDataSource`（或任何提供 ``records(symbol)``
            的对象）。按**公告日**点取（ADR-0006）。
        window: 百分位窗口，默认 :data:`DEFAULT_WINDOW`。

    返回:
        :class:`Valuation`。三个序列与 ``close`` 的索引、列完全一致。

    复杂度是每个标的「一遍公告 + 一遍交易日」：公告是几十条，交易日是几千条，故实际代价由
    交易日决定（全市场 5000 标的 × 6400 日 ≈ 3200 万格，numpy 支撑下可接受）。
    """
    if not isinstance(close.index, pd.DatetimeIndex):
        raise MarketDataError(
            f"收盘价的索引必须是 DatetimeIndex，收到 {type(close.index).__name__}"
        )
    if not close.index.is_monotonic_increasing:
        raise MarketDataError("收盘价的索引必须升序——按公告日点取依赖有序索引")
    if window < 1:
        raise ValueError(f"窗口至少为 1，收到 {window}")

    earnings = pd.DataFrame(np.nan, index=close.index, columns=close.columns, dtype=float)
    growth = pd.DataFrame(np.nan, index=close.index, columns=close.columns, dtype=float)

    for symbol in close.columns:
        records = [record for record in financials.records(symbol) if record.usable]
        if not records:
            continue
        earnings[symbol] = _point_in_time(_earnings_steps(records), close.index)
        growth[symbol] = _point_in_time(_growth_steps(records), close.index)

    pe = price_earnings_ratio(close, earnings)
    return Valuation(
        pe=pe,
        pe_percentile=pe_percentile(pe, window),
        peg=price_earnings_growth(pe, growth),
    )


def _earnings_steps(records) -> list[tuple[dt.date, float]]:
    """``(公告日, 年化每股收益)``，按公告日升序。"""
    steps = [
        (
            record.announcement_date,
            record.values["eps_ytd"] * annualisation_factor(record.report_period),
        )
        for record in records
    ]
    return sorted(steps, key=lambda step: step[0])


def _growth_steps(records) -> list[tuple[dt.date, float]]:
    """``(公告日, 归母净利润累计同比%)``，按公告日升序。

    去年同期那一期**必须也是可用记录**（公告日可信）——否则算出的同比会建立在一条占位符
    记录上，而那正是 :attr:`FinancialRecord.usable` 要挡的东西。
    """
    by_period = {record.report_period: record for record in records}
    steps = []
    for record in records:
        previous = by_period.get(_same_period_last_year(record.report_period))
        if previous is None:
            continue
        before = previous.values["net_profit_ytd"]
        if before == 0:
            continue  # 基数为 0，同比无定义
        now = record.values["net_profit_ytd"]
        steps.append((record.announcement_date, (now - before) / abs(before) * 100.0))
    return sorted(steps, key=lambda step: step[0])


def _same_period_last_year(period: dt.date) -> dt.date | None:
    """上年同期的报告期。2 月 29 日在季末不会出现，但仍防御一下。"""
    try:
        return dt.date(period.year - 1, period.month, period.day)
    except ValueError:
        return None


def _point_in_time(steps: list[tuple[dt.date, float]], index: pd.DatetimeIndex) -> pd.Series:
    """把 ``(公告日, 值)`` 铺成逐日序列：某日取**当日及以前**最近一条公告的值。

    这正是 ADR-0006 的落点：按公告日而非报告期。公告之前的交易日是**缺失**（那时还不知道），
    而不是 0——给 0 会让 PE 变成无穷大或 0，属于凭空造数。
    """
    if not steps:
        return pd.Series(np.nan, index=index, dtype=float)

    stamps = pd.DatetimeIndex([stamp for stamp, _ in steps])
    values = pd.Series([value for _, value in steps], index=stamps, dtype=float)
    # 同日多条公告（罕见，例如年报与一季报同日）保留**后出现**的那条：``_earnings_steps`` 已按
    # 公告日排序，而 Python 的排序是稳定的，故「后出现」即列表里靠后的那条。
    values = values[~values.index.duplicated(keep="last")].sort_index()
    return values.reindex(index, method="ffill")


def clip_fields(
    fields: dict[str, pd.DataFrame],
    markets,
    *,
    start=None,
    end=None,
) -> dict[str, pd.DataFrame]:
    """把算好的信号截到回测区间，并对齐到 ``markets`` 的标的与顺序。

    **为什么必须先算后截**：百分位的回看窗口是 1000 个交易日。若拿着**已截断**的行情去算，
    区间开头的窗口就只剩几天——而它是本策略的核心输入，截短了不会报错，只会让开头那段日子
    的「分位」失真。故顺序是：用**完整历史**算，再截到回测区间。

    截断口径与 :func:`~mbt.data.loader.slice_markets` 一致（`.loc` 两端含），并保留
    ``markets`` 的列顺序——引擎那边的行情面板正是按那个顺序组装的，顺序不同会被
    :class:`~mbt.data.panel.Panel` 判为不一致（这是对的：顺序不同往往意味着两份数据来源
    对不上，而静默重排会掩盖它）。
    """
    symbols = [market.symbol for market in markets]
    lower = pd.Timestamp(start) if start else None
    upper = pd.Timestamp(end) if end else None

    clipped = {}
    for name, frame in fields.items():
        one = frame.loc[lower:upper] if lower is not None or upper is not None else frame
        missing = [symbol for symbol in symbols if symbol not in one.columns]
        if missing:
            raise MarketDataError(
                f"信号 {name!r} 缺少标的 {missing[:3]}…——它与行情不是同一次取数的产物"
            )
        clipped[name] = one[symbols]
    return clipped


def valuation_for(markets, financials, *, window: int = DEFAULT_WINDOW) -> Valuation:
    """由一组**原始**行情算出估值三序列——库里唯一那个「组装」入口。

    .. warning::

        **必须传原始行情，不能传后复权行情。** PE 是「当时的成交价 ÷ 当时的每股收益」，
        而后复权价以首根为基准向后放大——实测 `sh600000` 末根原始价 9.26 对应后复权价
        **123.88（放大 13.38 倍）**，故用后复权价算出的 PE 会从 5.20 变成 **69.60**。

        这个错误不会报错，只会让估值整体偏一个倍数，而「PE 百分位」这类相对量甚至可能
        看起来仍然合理（因为整条序列同比放大）——除了放大倍数在历史各段不同，于是百分位
        本身也失真。故这里的入参刻意叫 ``markets`` 并在此写明。

    参数:
        markets: 一组 :class:`~mbt.data.market.MarketData`（**原始价**）。
        financials: :class:`~mbt.data.fundamental.CwDataSource` 之类。
        window: 百分位窗口。

    返回:
        :class:`Valuation`，索引与列取自各标的收盘价的**并集**（与引擎时钟一致）。
    """
    from .panel import assemble_panel

    if not markets:
        raise MarketDataError("算估值至少要有一个标的")

    close = assemble_panel(list(markets), ["close"])["close"]
    return build_valuation(close, financials, window=window)


def _require_same_shape(left: pd.DataFrame, right: pd.DataFrame, left_name, right_name) -> None:
    """两个标的宽表必须**同日同标的**，否则逐格相除会静默错位（ADR-0009）。"""
    if not left.index.equals(right.index):
        raise MarketDataError(f"{left_name}与{right_name}的 index 不一致，逐格相除会静默错位")
    if not left.columns.equals(right.columns):
        raise MarketDataError(f"{left_name}与{right_name}的 columns 不一致，逐格相除会静默错位")
