"""估值信号：动态PE、PE 百分位、PEG（票据 #37、#48）。

把「收盘价」与「按期财务」合成 **date × 标的** 形状的估值序列，供选股规则与策略共用
（ADR-0001：同一条件不必写两遍）。本模块与 :mod:`mbt.data.fundamental` 同属数据层——财务
派生的信号放这里而非 ``mbt.signals``，先例是 ``non_loss_mask``：那是纯 pandas 运算，但它
要读财务数据，而 ``mbt.signals`` 是「无 I/O 无状态」的纯函数库。

## 口径照抄通达信原式，不自创

三个公式与 TDX 的「市盈率」公式**逐项对齐**（历史分支），因为行情软件里的 PE/PEG 就是它算的，
而自创口径的最初一版实测已经造成过假阳性买入（见下）::

    PE历史   = FINVALUE(238) * C / FINVALUE(276)      # 总股本 × 收盘价 ÷ 归母净利TTM
    增历史   = FINVALUE(184)                            # 归母净利**当期**累计同比(%)
    PE百分位 = (PE − LLV(PE,DUR)) / (HHV(PE,DUR) − LLV(PE,DUR))
    PEG      = IF(PE>0 AND ABS(增)>0.1 AND 财年T>0, PE/增, DRAWNULL)

**动态PE 用 `利润TTM` 这个现成字段**，而不是「累计每股收益 × 年化系数」。后者是自创口径，
`Q1 × 4` 假定一季度占全年四分之一——实测 `sh600988` 的 2023Q1 只占全年约 10%（PE 被高估到
85）、`sz002692` 的 2021H1 净利仅 870 万（PE 冲到 931）。**代价落在百分位上**：被撑大的极值
进入 LLV/HHV 窗口后，会把之后整整 `DUR` 个交易日的分位都压低——实测 `sz002692` 在
2025-09-02 因此得到百分位 **2.55%**（通过「< 8%」而买入），而通达信口径是 **91.82%**
（根本不该买）。

**增长率用 `增历史`（当期累计同比）**，不是滞后一年的同比。滞后口径实测会把恶化藏住：
`sz002692` 的 2025-06-30 当期同比是 **−1.67%**，而「上一年同期那一期」的同比是 **+120.89%**
（基期净利仅 1932 万），于是 PEG 从通达信的 **−36.57** 变成 **+0.42**——符号相反。

**PEG 有 `|增长率| > 0.1` 的守卫**：增长率接近 0 时 PEG 是个巨大的数，那不是「贵」，是
没有意义。

.. note::

    通达信原式里 PEG 还有第三个条件 ``财年T>0``，而 `财年T` 来自 `GPONEDAT(4)`——那是
    **专业财务数据**（``vipdoc/cw/gp*.dat``），本项目尚未解析。同一条公式的**前瞻分支**
    （``PE前瞻 = C/EPS_T``、``增预期``）也依赖它。故**当前只实现历史分支**，且省略该条件。

**PE 百分位**取**含当日**的窗口，且**不足 ``window`` 根时按可得历史算**（通达信 ``LLV``/``HHV``
语义）。故次新股的「百分位」是短窗口上的百分位，其含义随上市时间漂移——这是照原式的结果，
不是疏漏。

.. warning::

    **非正的 PE 也参与 LLV/HHV**（公式原样，与通达信一致），这会让百分位失真：某公司有过
    亏损季度时 PE 曾为负，于是 ``LLV`` 是个负数，此后即使 PE 只是 20（对一家持续盈利的公司
    算便宜），百分位也会被抬高。例：``LLV=-50、HHV=40、PE=20`` → 百分位 ``0.78``。

    这与通达信一致，故保留；要改就是**另一个口径**，需另行登记。

## 时点正确性（ADR-0006）

一切按**公告日**对齐：某交易日的估值用当天**已公告**的最近一期财报，而不是「报告期落在当天
之前」的那一期。故 4 月底之前用不到一季报——即使它的报告期是 3 月 31 日。

:attr:`~mbt.data.fundamental.FinancialRecord.usable` 那道门（公告日 > 报告期、且年份 ≥ 2005）
在这里同样生效：不可用的记录**不进序列**，而不是给它编一个日期。

## 有效起点

财务数据 2005 年前不可用，加上 ``window`` 个交易日的回看，能算出百分位的最早日子在 2010
前后。这不是本模块的限制，是数据事实——**回测起点因此被财务数据决定，而不是被行情数据决定**。
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
#: 总市值字段名（元）。
#:
#: ``= 总股本 × 收盘价``——**用原始价**，与 PE 同一个理由：市值是当时的真实市值，用后复权价
#: 会把它随首根放大（实测 ``sh600000`` 放大 13.38 倍）。它只作**过滤条件**用（如「市值 >
#: 100 亿」），不参与收益计算。
MARKET_CAP = "market_cap"

#: 本模块产出的字段，按构造顺序。给调用方**显式声明**用（与 ``SCREEN_FIELDS`` 同一精神）。
VALUATION_FIELDS = (PE, PE_PERCENTILE, PEG, MARKET_CAP)

#: 百分位的默认窗口（交易日）——约 4 年。
DEFAULT_WINDOW = 1000


def price_earnings_ratio(
    close: pd.DataFrame, shares: pd.DataFrame, profit_ttm: pd.DataFrame
) -> pd.DataFrame:
    """``PE = 总股本 × 收盘价 ÷ 归母净利润TTM``（总市值 ÷ TTM 净利）。

    这就是通达信 `FINVALUE(238)*C/FINVALUE(276)` 的口径，与行情软件里显示的市盈率一致。

    .. warning::

        **不要退回「累计每股收益 × 年化系数」那一版。** 它靠 `×4 / ×2 / ×4÷3` 把一季报
        年化，而 `Q1 × 4` 假定一季度占全年四分之一——实测 `sh600988` 的 2023Q1 只占全年
        约 10%，于是 PE 被高估到 **85**（真实约 34.7），`sz002692` 的 2021H1 净利仅 870 万，
        年化后 PE 冲到 **931**。

        单看「当天 PE 偏一点」还不够严重；真正的代价在**百分位**：一个被撑大的极值进入
        LLV/HHV 窗口后，会把之后**整整 `window` 个交易日**的分位都压低。实测 `sz002692`
        在 2025-09-02 的年化口径给出百分位 **2.55%**（假阳性买入），而 TTM 口径是 **91.82%**。

    分子分母任一缺失处即缺失；``利润TTM`` 为 0 处也判缺失（除零无意义），而**负值照常算出
    负 PE**——「PE > 0」是过滤条件的事，不是计算的事。
    """
    _require_same_shape(close, shares, "收盘价", "总股本")
    _require_same_shape(close, profit_ttm, "收盘价", "归母净利润TTM")
    safe = profit_ttm.where(profit_ttm != 0)
    return close * shares / safe


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


#: PEG 只在**增长率绝对值**大于它时才有意义。
#:
#: 通达信原式：``PEG:IF(选用PE>0 AND ABS(选用增)>0.1 AND 财年T>0, 选用PE/选用增, DRAWNULL)``。
#: 增长率接近 0 时除数极小，PEG 会是个巨大的数——那不是「贵」，是**没有意义**，故取缺失。
MIN_GROWTH_ABS = 0.1


def price_earnings_growth(pe: pd.DataFrame, growth_pct: pd.DataFrame) -> pd.DataFrame:
    """``PEG = PE ÷ 盈利增长率(%)``；``PE ≤ 0`` 或 ``|增长率| ≤ 0.1`` 处为缺失。

    增长率是**百分数**（30 表示 30%），故 PEG 的量纲与常用读数一致（PE 15、增长 30 → 0.5）。

    .. note::

        通达信原式里还有第三个条件 ``财年T>0``——`财年T` 来自 `GPONEDAT(4)`，属于**专业财务
        数据**（`vipdoc/cw/gp*.dat`），本项目尚未解析。故本实现只保留前两个条件；这个差额
        已登记在票据待办里。
    """
    _require_same_shape(pe, growth_pct, "动态PE", "盈利增长率")
    usable = (pe > 0) & (growth_pct.abs() > MIN_GROWTH_ABS)
    return (pe / growth_pct).where(usable)


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
    market_cap: pd.DataFrame

    def as_fields(self) -> dict[str, pd.DataFrame]:
        """字段名 → 标的宽表，供装进 :class:`~mbt.data.panel.Panel` 或交给策略。"""
        return {
            PE: self.pe,
            PE_PERCENTILE: self.pe_percentile,
            PEG: self.peg,
            MARKET_CAP: self.market_cap,
        }


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

    shares = _point_in_time_frame(close, financials, "total_shares")
    ttm = _point_in_time_frame(close, financials, "profit_ttm")
    growth = _point_in_time_frame(close, financials, "growth_ytd")

    pe = price_earnings_ratio(close, shares, ttm)
    return Valuation(
        pe=pe,
        pe_percentile=pe_percentile(pe, window),
        peg=price_earnings_growth(pe, growth),
        market_cap=close * shares,
    )


def _point_in_time_frame(close: pd.DataFrame, financials, field: str) -> pd.DataFrame:
    """把某个财务字段铺成与 ``close`` 同形的逐日序列（按**公告日**点取，ADR-0006）。

    公告之前的交易日是**缺失**（那时还不知道），而不是 0——给 0 会让 PE 变成无穷大或 0，
    属于凭空造数。
    """
    out = pd.DataFrame(np.nan, index=close.index, columns=close.columns, dtype=float)
    for symbol in close.columns:
        steps = sorted(
            (record.announcement_date, record.values[field])
            for record in financials.records(symbol)
            if record.usable
        )
        if steps:
            out[symbol] = _point_in_time(steps, close.index)
    return out


def _point_in_time(steps: list[tuple[dt.date, float]], index: pd.DatetimeIndex) -> pd.Series:
    """把 ``(公告日, 值)`` 铺成逐日序列：某日取**当日及以前**最近一条公告的值。

    同日多条公告（罕见，例如年报与一季报同日）保留**后出现**的那条。
    """
    if not steps:
        return pd.Series(np.nan, index=index, dtype=float)

    stamps = pd.DatetimeIndex([stamp for stamp, _ in steps])
    values = pd.Series([value for _, value in steps], index=stamps, dtype=float)
    values = values[~values.index.duplicated(keep="last")].sort_index()
    return values.reindex(index, method="ffill")


def valuation_for(
    markets,
    financials,
    *,
    window: int = DEFAULT_WINDOW,
) -> Valuation:
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
