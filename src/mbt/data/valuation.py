"""估值信号：动态PE、PE 百分位、PEG、市值、ROE、净资产（票据 #37、#48、#62）。

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

#: 净资产收益率字段名，**单位是百分数**（``10.0`` 表示 10%）。
#:
#: ``= 归母净利润TTM ÷ 归母股东权益 × 100``。
#:
#: .. warning::
#:
#:     **不要用 ``gpcw`` 那个现成的「净资产收益率」字段（索引 5）。** 它是**累计**值——
#:     一季报给的是「年初到 3 月底」的 ROE。实测茅台 2026-03-31 那个字段是 **10.06**，而
#:     同期按 TTM 算是 **31.63**；平安银行更是 **2.67** vs **7.91**。若拿它做「ROE > 10%」
#:     的过滤，**一季度报告期里几乎所有股票都会被排除**——与 PE 那个「年化 ×4」的陷阱同源，
#:     只是方向相反（那个偏大、这个偏小）。
#:
#:     现成字段仍可作**交叉验证**：年末时它与本口径接近（茅台 2025-12-31：33.65 vs 34.87）。
ROE = "roe"

#: 归母股东权益字段名（净资产），单位是**元**。
#:
#: 暴露它是为了给选股规则一条**显式**的下界过滤（``equity > 0``，票据 #62）。
#:
#: .. note::
#:
#:     **它是防御性的，不是当前在起作用的那道闸门。** 在 ``pe_above >= 0`` 的前提下，
#:     「净资产 > 0」被「动态PE > 0」**蕴含**，故不改变候选集：
#:
#:         equity < 0 且 ROE > 10
#:         ⟹ ROE = 净利TTM ÷ 权益 > 0 ⟹ 净利TTM < 0
#:         ⟹ PE = 市值 ÷ 净利TTM < 0 ⟹ 已被「PE > 0」排除
#:
#:     （第一步之所以成立，是因为 ROE 的分子分母**同时为负**时商为正——这正是净资产为负的
#:     公司会带着巨大正 ROE 的原因。）
#:
#:     真实数据核验（全市场 4,792 只、区间内 4,697 只、2018-01-02 → 2026-09-11）：
#:
#:     - 有完整数据的格 **8,361,752**，其中「净资产 ≤ 0」**29,914** 格（0.36%）；
#:     - 这 29,914 格里 **29,393** 格（**98%**）的 ROE > 10——负净资产几乎必然伴随巨大
#:       正 ROE，正是上面那条推论；
#:     - 但满足「净资产 ≤ 0 且 ROE > 10 且 PE > 0」的反例：**0 格**；
#:     - 加过滤前后，候选集都是 **8,602 格**，逐格无差异。
#:
#:     它会在 ``pe_above < 0``（显式放行亏损公司）时**才**真正起作用——那时亏损公司不再被
#:     PE 拦下，净资产为负的会带着巨大正 ROE 挤进排序（排序因子正是 ROE）。
#:     保留它是为了让「净资产为正」成为一个**声明出来的前提**，而不是 PE 符号的偶然推论。
#:     这条冗余关系有测试钉住：``test_the_equity_filter_only_bites_when_negative_pe_is_allowed``。
EQUITY = "equity"

#: 本模块产出的字段，按构造顺序。给调用方**显式声明**用（与 ``SCREEN_FIELDS`` 同一精神）。
VALUATION_FIELDS = (PE, PE_PERCENTILE, PEG, MARKET_CAP, ROE, EQUITY)

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


def return_on_equity(profit_ttm: pd.DataFrame, equity: pd.DataFrame) -> pd.DataFrame:
    """``ROE = 归母净利润TTM ÷ 归母股东权益 × 100``（**百分数**）。

    用 TTM 净利而非累计净利，是为了避开季节性——见 :data:`ROE` 的 warning。权益取**最新
    时点**值（不是期间平均）：那是可得的、且与「市值 ÷ 净资产」这类同期口径一致；代价是
    权益在期内大幅变动（增发、回购）时会略有偏差。

    权益为 0 处判缺失（除零无意义）；**负权益照常算出负 ROE**——「ROE > 0」是过滤条件的事，
    不是计算的事（与 PE 的处理一致）。
    """
    _require_same_shape(profit_ttm, equity, "归母净利润TTM", "归母股东权益")
    safe = equity.where(equity != 0)
    return profit_ttm / safe * 100.0


@dataclass(frozen=True)
class Valuation:
    """估值序列（都是 **date × 标的** 的标的宽表，与价格同日对齐）。

    属性:
        pe: 动态PE。
        pe_percentile: PE 百分位，``[0, 1]``。
        peg: PEG。
        market_cap: 总市值（元）。
        roe: ROE，**百分数**。
        equity: 归母股东权益（净资产，元）。用来做 ``equity > 0`` 的下界过滤，
            见 :data:`EQUITY`。
    """

    pe: pd.DataFrame
    pe_percentile: pd.DataFrame
    peg: pd.DataFrame
    market_cap: pd.DataFrame
    roe: pd.DataFrame
    equity: pd.DataFrame

    def as_fields(self) -> dict[str, pd.DataFrame]:
        """字段名 → 标的宽表，供装进 :class:`~mbt.data.panel.Panel` 或交给策略。"""
        return {
            PE: self.pe,
            PE_PERCENTILE: self.pe_percentile,
            PEG: self.peg,
            MARKET_CAP: self.market_cap,
            ROE: self.roe,
            EQUITY: self.equity,
        }


def build_valuation(
    close: pd.DataFrame,
    financials,
    *,
    window: int = DEFAULT_WINDOW,
) -> Valuation:
    """由收盘价与按期财务算出估值序列。

    参数:
        close: **date × 标的** 的收盘价标的宽表。索引须升序，列即标的。
        financials: :class:`~mbt.data.fundamental.CwDataSource`（或任何提供 ``records(symbol)``
            的对象）。按**公告日**点取（ADR-0006）。
        window: 百分位窗口，默认 :data:`DEFAULT_WINDOW`。

    返回:
        :class:`Valuation`。各序列与 ``close`` 的索引、列完全一致。

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

    # **一次**把四个字段铺出来（装载一次、每标的取记录一次），理由见 `_point_in_time_fields`。
    point_in_time = _point_in_time_fields(
        close, financials, ("total_shares", "profit_ttm", "growth_ytd", "equity")
    )
    shares = point_in_time["total_shares"]
    ttm = point_in_time["profit_ttm"]
    growth = point_in_time["growth_ytd"]
    equity = point_in_time["equity"]

    pe = price_earnings_ratio(close, shares, ttm)
    return Valuation(
        pe=pe,
        pe_percentile=pe_percentile(pe, window),
        peg=price_earnings_growth(pe, growth),
        market_cap=close * shares,
        roe=return_on_equity(ttm, equity),
        equity=equity,
    )


def _point_in_time_frame(close: pd.DataFrame, financials, field: str) -> pd.DataFrame:
    """把某个财务字段铺成与 ``close`` 同形的逐日序列（按**公告日**点取，ADR-0006）。

    公告之前的交易日是**缺失**（那时还不知道），而不是 0——给 0 会让 PE 变成无穷大或 0，
    属于凭空造数。

    这是 :func:`_point_in_time_fields` 的单字段薄封装，留给「只要一个字段」的调用方。
    """
    return _point_in_time_fields(close, financials, (field,))[field]


def _point_in_time_fields(close: pd.DataFrame, financials, fields) -> dict[str, pd.DataFrame]:
    """把**多个**财务字段一次铺成与 ``close`` 同形的逐日序列。

    存在的理由只有一条，但它是分钟量级的：**一遍顶四遍**。原先是四个字段各调一次
    :func:`_point_in_time_frame`，于是「装载（含目录指纹）+ 取标的记录 + 铺到交易日」这一整套
    对每个字段各跑一遍。实测（2026-09-17，401 标的 × 7,125 交易日）本函数所在的
    ``build_valuation`` 里：

    ==========================================  =======  ======
    子步骤                                        用时     占比
    ==========================================  =======  ======
    目录指纹（``_load`` 里的 glob + 147 个 stat）   5.86s     40%
    ``records()``（含上面的指纹）                   3.35s     23%
    ``_point_in_time``（真正在铺数据）              3.48s     24%
    ``build_valuation`` 整体                       14.74s   100%
    ==========================================  =======  ======

    三个都只该发生**一次**。故这里：装载一次（``financials.tables()``）、每个标的取一次记录
    （``records(symbol, tables=...)``）、再对多个字段各铺一次。

    参数:
        close: **date × 标的** 的收盘价标的宽表（只用来定形状与索引，值不参与）。
        financials: 提供 ``records(symbol, *, tables=)`` 与 ``tables()`` 的对象
            （如 :class:`~mbt.data.fundamental.CwDataSource`）。
        fields: 要铺的字段名。

    返回:
        ``{字段: DataFrame}``，各帧与 ``close`` 的索引、列完全一致。
    """
    if not isinstance(close.index, pd.DatetimeIndex):
        raise MarketDataError(
            f"收盘价的索引必须是 DatetimeIndex，收到 {type(close.index).__name__}"
        )
    if not close.index.is_monotonic_increasing:
        raise MarketDataError("收盘价的索引必须升序——按公告日点取依赖有序索引")

    wanted = tuple(fields)
    rows, width = close.shape
    out = {field: np.full((rows, width), np.nan) for field in wanted}

    # **本模块对 ``financials`` 的依赖仍是「一个方法」**：``records(symbol)``。``tables()`` 是
    # 可选的加速口——真实数据源（``CwDataSource``）有它，于是装载与目录指纹只算一次；而只在
    # 测试里存在的极小鸭子类型（见 ``tests/test_valuation.py`` 的 ``FakeFinancials``）没有它，
    # 照旧走逐标的 ``records(symbol)``。不为了省时间把依赖面撑大。
    tables = financials.tables() if hasattr(financials, "tables") else None
    for position, symbol in enumerate(close.columns):
        records = (
            financials.records(symbol, tables=tables)
            if tables is not None
            else financials.records(symbol)
        )
        for field in wanted:
            steps = sorted(
                (record.announcement_date, record.values[field])
                for record in records
                if record.usable
            )
            if steps:
                out[field][:, position] = _point_in_time(steps, close.index)

    return {
        field: pd.DataFrame(values, index=close.index, columns=close.columns, dtype=float)
        for field, values in out.items()
    }


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
