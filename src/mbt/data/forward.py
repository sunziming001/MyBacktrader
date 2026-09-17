"""估值的前瞻分支（票据 #50）：把「一致预期」按通达信原式接进来。

## 只能在**非回溯**场景使用

    **这一模块给的是「现在这一版」一致预期，不是「评估日当时」的一致预期。**

一致预期的来源 :mod:`mbt.data.gpone` 是**快照**：每个 ``(代码, 字段)`` 只有一条记录，
文件里**没有日期字段**。机构上一回给出的预测已被下一回覆盖——不是「旧记录被挤掉」，是
**根本没有旧记录**（这是 ADR-0006 修订二登记的缺口）。

所以：**不得**把本模块的读数用于任何声称时点正确的回测。它适用于「今天的池子、今天的读数」
这类非回溯场景。这也是本模块**没有** ``as_of`` 参数的原因——不是漏了，是不给这个口子。

正因如此，本模块**刻意不接进** :func:`mbt.data.valuation.build_valuation`：那条路按公告日
点取（ADR-0006），把一份「最新值」混进去会让整张估值表静默失去时点正确性。有一条测试钉住
这件事（``test_valuation_does_not_reach_for_the_forward_branch``）。

## 原式（票据 #50 正文）

```
财年T := GPONEDAT(4);                     K线年 := YEAR;
用前瞻 := 财年T>0 AND K线年=财年T;

PE前瞻 := IF(EPS_T>0, C/EPS_T, IF(PE预期>0, PE预期, 0));
   其中 EPS_T := GPONEDAT(5)   PE预期 := GPONEDAT(23)
增预期 := IF(基期净利万>0 AND 净利T>0, (净利T-基期净利万)/基期净利万*100, 0);
   其中 净利T := GPONEDAT(8)   基期净利万 := 基期年报归母净利

选用PE := IF(用前瞻, PE前瞻, PE历史) NODRAW;
选用增 := IF(用前瞻, 增预期, 增历史);      ← PE 与增长率**成对切换**
```

**「成对切换」在本模块是结构性保证，不是约定**：:func:`choose` 一次返回 PE 与增长率两个值
（:class:`Chosen`），没有「只换 PE 不换增长」的写法。

单位与缺失照原式：``净利T`` 与 ``基期净利万`` 都是**万元**；两者任一不大于 0 时 ``增预期``
取 **0**（原式如此，不是缺失）。``C`` 用**原始价**——与 :mod:`mbt.data.valuation` 同一个理由：
PE 要的是当时的成交价，用后复权价会整体放大（实测 `sh600000` 放大 13.38 倍）。

## ``选用增`` 唯一的去处是 PEG

原式里 ``选用增`` 与 ``选用PE`` 只喂同一件事::

    PEG := IF(选用PE>0 AND ABS(选用增)>0.1 AND 财年T>0, 选用PE/选用增, DRAWNULL)

本模块据此**重述** PEG（:func:`chosen_valuation` 的 ``restate_peg`` 参数），而**不**把
``选用增`` 当独立信号发出去：没有规则读裸增长率（:data:`mbt.data.valuation.VALUATION_FIELDS`
是公开词汇表，塞一个没人用的名字进去只会让面板多一列死数据）。

**重述带一个降级，它比换口径更狠。** ``valuation`` 里那份 PEG 只守两条门（``PE>0``、
``增`` 不近 0）；原式还多一条 ``财年T>0``——即**没有一致预期就没有 PEG**。这一条把评估日那天
约六成的标的从 PEG 这道门上整片摘掉（实测池子 4,752 只里 57.7%）。而它**只在快照可用于评估日
时**才施加（:func:`snapshot_admissible`），否则「整张表逐格不变」那条不变式就破了——回测里
PEG 也就跟着变了。
"""

from __future__ import annotations

import datetime as dt
from collections.abc import Mapping
from dataclasses import dataclass

import pandas as pd

from .valuation import DEFAULT_WINDOW, MIN_GROWTH_ABS

#: 取「基期年报」的字段名：归母净利润（元，累计）。年末那一期即全年数。
_BASE_PROFIT_FIELD = "net_profit_ytd"

#: 元 → 万元。``GPONEDAT(8)`` 的单位就是万元，故基期净利要换算过来才能相减。
_YUAN_PER_WAN = 1e4

#: 字段号（见 :data:`mbt.data.gpone.FIELDS`）。写成常量免得正文里散着魔数。
FIELD_FISCAL_YEAR = 4
FIELD_EPS_T = 5
FIELD_NET_PROFIT_T = 8
FIELD_PE_EXPECTED = 23

#: 调用方要一并交代给用户的那句话。非回溯场景里也该让人知道读到的是什么。
FORWARD_CAVEAT = (
    "这是「现在这一版」一致预期，不是评估日当时的；本地没有带发布日的一致预期历史，"
    "故不能用于声称时点正确的回测。"
)


@dataclass(frozen=True)
class ForwardReading:
    """某标的**今天**这一版的前瞻原料。

    属性:
        symbol: 标的。
        fiscal_year: ``GPONEDAT(4)`` 财年T；没有时 0（原式：「空值显示为 0」）。
        eps_t: ``GPONEDAT(5)`` 一致预期T年每股收益（元）。
        net_profit_t: ``GPONEDAT(8)`` 一致预期T年净利润（**万元**）。
        pe_expected: ``GPONEDAT(23)`` 一致预期T年PE。
        base_net_profit: 基期年报归母净利（**万元**）；拿不到时报缺失。
    """

    symbol: str
    fiscal_year: float
    eps_t: float
    net_profit_t: float
    pe_expected: float
    base_net_profit: float | None

    @property
    def growth_expected(self) -> float:
        """``增预期``（**百分数**）——见模块级 :func:`growth_expected`。"""
        return growth_expected(self.net_profit_t, self.base_net_profit)

    def pe_forward(self, close: float) -> float:
        """``PE前瞻``——见模块级 :func:`pe_forward`。"""
        return pe_forward(self.eps_t, self.pe_expected, close=close)


@dataclass(frozen=True)
class Chosen:
    """「选用PE / 选用增」这一对值。

    两个值**同出一处**是有意的：原式要求 PE 与增长率成对切换，而这个类型没有「只给一个」的
    构造方式，故调用方拿不到「换了 PE 没换增长」的组合。
    """

    pe: float
    growth: float
    used_forward: bool


def pe_forward(eps_t: float, pe_expected: float, *, close: float) -> float:
    """``PE前瞻 = IF(EPS_T>0, C/EPS_T, IF(PE预期>0, PE预期, 0))``。

    参数:
        close: **原始价**（当时的成交价），不是后复权价。

    公式只写这一处：:meth:`ForwardReading.pe_forward` 与
    :func:`chosen_valuation` 都走它，免得同一条式子长成两份而各自漂移。
    """
    if eps_t > 0:
        return close / eps_t
    if pe_expected > 0:
        return pe_expected
    return 0.0


def growth_expected(net_profit_t: float, base_net_profit: float | None) -> float:
    """``增预期 = IF(基期净利万>0 AND 净利T>0, (净利T-基期净利万)/基期净利万*100, 0)``。

    两处的「取 0」是原式写的、不是本模块的取舍：**基期为负**（亏损年谈不上同比）与**预测为负**
    都落在同一支上。0 的后果是下面 :func:`pe_growth_ratio` 的 ``|增|>0.1`` 那道门必不过，
    故「算不出增长」最终表现为「PEG 缺失」——而不是被当成「增长为零、PEG 无穷大」。
    """
    if not base_net_profit or base_net_profit <= 0 or net_profit_t <= 0:
        return 0.0
    return (net_profit_t - base_net_profit) / base_net_profit * 100.0


def pe_growth_ratio(pe: float, growth: float) -> float:
    """``PEG = 选用PE ÷ 选用增``，两道守卫不过就报缺失。

    原式：``IF(选用PE>0 AND ABS(选用增)>0.1, 选用PE/选用增, DRAWNULL)``。增长率近 0 时
    PEG 会是个巨大的数——那不是「贵」，是**没有意义**，故取 ``NaN`` 而不是一个很大的值。

    门槛取自 :data:`mbt.data.valuation.MIN_GROWTH_ABS`，与历史那一支**同一个常量**：
    这条边界在两处必须一样，否则同一个增长率在两条路上一个算得出、一个算不出。
    """
    if pe > 0 and abs(growth) > MIN_GROWTH_ABS:
        return pe / growth
    return float("nan")


def uses_forward(fiscal_year: float, bar_year: int) -> bool:
    """``用前瞻 := 财年T>0 AND K线年=财年T``。

    两条缺一不可：``财年T`` 为 0 表示这只票没有一致预期；年份不匹配表示预期落在**别的年份**
    （例如到了次年，T 还是去年那个）——那时用预期口径会拿旧预测配新价格。
    """
    return fiscal_year > 0 and int(fiscal_year) == bar_year


def snapshot_admissible(symbol: str, *, gpone, as_of) -> bool:
    """评估日**不早于**该文件的最后写入日——一致预期快照的时点闸门（ADR-0006 修订二）。

    这是 :func:`forward_available` 的**前一半**，单独拿出来是因为它还有第二个用处：原式那条
    ``财年T>0`` 守卫也必须**先过这道闸门**才许施加。若不分出来，「没有一致预期 → PEG 缺失」
    就会落到评估日早于写入日的那些行上——等于拿一份当时还不存在的快照，去抹掉那一行的历史
    PEG，而回测的评估日全在过去。

    它在写入日**当天**即为真：日常任务收盘后下载、随即选股，两个日期正是同一天。
    """
    if isinstance(as_of, dt.datetime):
        as_of = as_of.date()
    return gpone.updated_on(symbol) <= as_of


def forward_available(symbol: str, *, gpone, as_of) -> bool:
    """该标的在评估日**能不能**用前瞻口径。

    两条同时成立才算可用：

    ==================  ==================================================
    条件                为什么
    ==================  ==================================================
    ``评估日 ≥ 文件写入日``  一致预期是**快照**、没有历史（ADR-0006 修订二）。早于文件写入日的
                        评估日上，我们手上这份内容可能已经不是那天的内容——不可知，故不许用。
    ``用前瞻`` 成立        原式自己的那条（见 :func:`uses_forward`）：有预期、且财年对得上当前年份。
    ==================  ==================================================

    第一条把「前瞻只在盘后选股生效」从约定变成了**判据**：日常任务收盘后下载数据、随即选股，
    评估日正好等于写入日；而任何回溯到更早的评估日，手里这份快照都可能已被后面的下载覆盖过。

    拿不到前瞻的（没一致预期、财年对不上）返回 ``False``——「没有就用历史」。

    .. warning::

        文件**读不到**时本函数**报错**，不是返回 ``False``。「这只票没有一致预期」与「目录
        指错了」是两回事，判据本身分不出来。调用方（:func:`chosen_valuation`）接住它、退回
        历史，并把读不到的标的点名列出来——降级可以是暗的，但**指错目录不能**。

        调用方要先把「这市场压根没有这族文件」摘出去（:meth:`GponeDataSource.covers`），
        否则北交所那几百只每天都会顶上来当「读不到」。

    这只回答「能不能用」，不回答「用了多少」；调用方要统计并印出来。

    参数:
        as_of: 评估日（``datetime.date``）。**这是可用性的闸门，不是取数的键**——本模块仍然
            取不到「评估日当时」的一致预期，只是拒绝在不足以支撑它的日子上使用。
    """
    if isinstance(as_of, dt.datetime):
        as_of = as_of.date()
    if not snapshot_admissible(symbol, gpone=gpone, as_of=as_of):
        return False
    return uses_forward(gpone.value(symbol, FIELD_FISCAL_YEAR) or 0.0, as_of.year)


def choose(
    reading: ForwardReading,
    *,
    close: float,
    bar_year: int,
    trailing_pe: float,
    trailing_growth: float,
) -> Chosen:
    """按原式在历史与前瞻之间**成对**切换。

    参数:
        reading: 该标的的前瞻原料。
        close: **原始价**。
        bar_year: ``K线年``，即这根 K 线所属年份。
        trailing_pe: ``PE历史``（:func:`mbt.data.valuation.price_earnings_ratio`）。
        trailing_growth: ``增历史``（:data:`mbt.data.fundamental` 的 ``growth_ytd``）。
    """
    if uses_forward(reading.fiscal_year, bar_year):
        return Chosen(reading.pe_forward(close), reading.growth_expected, True)
    return Chosen(trailing_pe, trailing_growth, False)


def reading(
    symbol: str,
    *,
    gpone,
    financials,
    tables=None,
    fields: tuple[int, ...] = (
        FIELD_FISCAL_YEAR,
        FIELD_EPS_T,
        FIELD_NET_PROFIT_T,
        FIELD_PE_EXPECTED,
    ),
) -> ForwardReading:
    """读一只标的的前瞻原料。

    参数:
        symbol: 标的。
        gpone: :class:`~mbt.data.gpone.GponeDataSource`。
        financials: :class:`~mbt.data.fundamental.CwDataSource` 之类，提供 ``records(symbol)``。
            用来取**基期年报**的归母净利（``净利T`` 是预测值，增长率要与**已公告的实际值**比）。
        tables: ``financials.tables()`` 的结果，批量调用方**应当**传进来复用一次装载
            （理由见 :func:`_records_of`）。
        fields: 要读的字段号，默认就是原式那四个。
    """
    values = {field: (gpone.value(symbol, field) or 0.0) for field in fields}
    return ForwardReading(
        symbol=symbol,
        fiscal_year=values[FIELD_FISCAL_YEAR],
        eps_t=values[FIELD_EPS_T],
        net_profit_t=values[FIELD_NET_PROFIT_T],
        pe_expected=values[FIELD_PE_EXPECTED],
        base_net_profit=_base_annual_profit(financials, symbol, tables=tables),
    )


def readings(symbols, *, gpone, financials, tables=None) -> dict[str, ForwardReading]:
    """一批标的的前瞻原料。缺数据的标的**不出现**在结果里（不补 0 假装有预期）。"""
    if tables is None:
        tables = _tables_of(financials)
    out: dict[str, ForwardReading] = {}
    for symbol in symbols:
        try:
            gpone.path_for(symbol)
        except Exception:  # noqa: BLE001 — 不覆盖的市场（如北交所）直接跳过
            continue
        out[symbol] = reading(symbol, gpone=gpone, financials=financials, tables=tables)
    return out


def _records_of(financials, symbol: str, tables):
    """取某标的的报告期记录。``tables`` 不为 ``None`` 时复用**一次装载**（见 ``records`` 的说明）。

    ``CwDataSource.records`` 不传 ``tables`` 时会先算一遍**目录指纹**（glob 147 个 ``.dat``
    再逐个 ``stat``，实测单次约 8.4 ms）。本模块要按标的问几千次，那就是**分钟量级**——而这
    笔开销与数据量无关，纯粹是目录调用。故批量入口一律把 ``tables`` 传下来。
    """
    if tables is None:
        return financials.records(symbol)
    return financials.records(symbol, tables=tables)


def _tables_of(financials):
    """要一份可复用的 ``tables``；给不出（如测试里的最小替身）就返回 ``None``。"""
    getter = getattr(financials, "tables", None)
    return getter() if callable(getter) else None


def _base_annual_profit(financials, symbol: str, *, tables=None) -> float | None:
    """基期年报归母净利（**万元**）：最近一期**可用的**年末财报。

    只取 ``12-31`` 那一期——``net_profit_ytd`` 是**累计**数，一季报给的是「年初到 3 月底」，
    拿它当「基期年报」会把增长率算成一个与全年无关的数。

    ``usable`` 那道门（公告日 > 报告期、且年份 ≥ 2005）照旧要过：2005 年之前的公告日是
    「等于报告期」的占位符，用它会提前数月知道尚未公布的财报（ADR-0006）。
    """
    base = None
    for record in _records_of(financials, symbol, tables):
        if record.report_period.month != 12 or record.report_period.day != 31:
            continue
        if not record.usable:
            continue
        base = record  # records() 是升序的，最后一条即最近一期
    if base is None:
        return None
    return base.values[_BASE_PROFIT_FIELD] / _YUAN_PER_WAN


def readings_frame(readings_by_symbol: Mapping[str, ForwardReading]) -> pd.DataFrame:
    """把原料摊成一张表（一行一标的），便于盘后筛一遍。索引是标的。"""
    return pd.DataFrame(
        {
            symbol: {
                "fiscal_year": item.fiscal_year,
                "eps_t": item.eps_t,
                "net_profit_t": item.net_profit_t,
                "pe_expected": item.pe_expected,
                "base_net_profit": item.base_net_profit,
                "growth_expected": item.growth_expected,
            }
            for symbol, item in readings_by_symbol.items()
        }
    ).T


@dataclass(frozen=True)
class PegRestatement:
    """重述 PEG 所需的**另两样东西**：历史 PEG 表，与算「增预期」的财务数据。

    两者**必须同来同去**，故捆在一个类型里：只给表格不给财务，算不出 ``增预期``；只给财务
    不给表格，没有底表可改。分开当两个参数就会留下「给了一个」的中间态，而这正是本模块一贯
    要消掉的东西（同 :class:`Chosen` 的用意）。
    """

    trailing_peg: pd.DataFrame
    financials: object


@dataclass(frozen=True)
class ChosenValuation:
    """盘后选股用的**「选用PE」两张表**：数值表与由它重算的百分位表（PEG 可选）。

    两张**必须同出一处**：百分位是「在窗口里的相对位置」，换了分子却留着旧尺子，读出来的
    便宜/贵就不是同一个口径了。故这里不给「只要一张」的构造方式。

    属性:
        pe: ``选用PE``——评估日那一行前瞻优先（有前瞻数据的标的），其余行是历史口径。
        pe_percentile: 由 ``pe`` 重算的百分位，窗口与 :func:`mbt.data.valuation.pe_percentile` 同。
        peg: ``PEG = 选用PE ÷ 选用增``（只重述**评估日那一行**）。只在调用方给了
            ``restate_peg`` 时才有值——读 PEG 的规则（``undervalued_growth``）才需要它，
            而 ``b1`` 一个 PEG 门都没有，为它多算一遍基期净利是白花钱。``None`` 表示
            「这次没重述」，不是「重述后写不进去」。
        forward_symbols: 走成了前瞻的标的（评估日那行确实换了值的）。
        unsupported: 该市场**根本没有**这族文件的标的（本机是北交所，没有 ``gpbjone.dat``——
            公式文档说 ``GPONEDAT`` 适用沪深京，但客户端没落这个文件）。这是**已知的覆盖
            缺口**，安静退回历史即可，不该报错。
        unreadable: 市场**有**这族文件、但文件读不到（如 ``--forward-root`` 指错目录）。
            与 ``unsupported`` 分开正是为了这一条：它需要有人去看一眼。
        peg_missing: 重述后**评估日那天没有 PEG** 的标的。这是从结果数的，不是从原因数的——
            三条守卫（没有一致预期、预测净利 ≤ 0、基期亏损）摘掉的都在里头，历史口径本来就缺
            的也在里头。数目必须报出来：它就是「PEG 这道门只看得到池子四成」这件事在产物里
            的落点。

        ``unsupported`` 与 ``unreadable`` 若合成一个数，盘后选股会**每天**为几百只北交所
        标的喊「指对目录了吗」——天天喊就等于不喊，真正指错的那天反而看不出来。
    """

    pe: pd.DataFrame
    pe_percentile: pd.DataFrame
    peg: pd.DataFrame | None = None
    forward_symbols: tuple[str, ...] = ()
    unsupported: tuple[str, ...] = ()
    unreadable: tuple[str, ...] = ()
    peg_missing: tuple[str, ...] = ()


def chosen_valuation(
    close: pd.DataFrame,
    trailing_pe: pd.DataFrame,
    *,
    gpone,
    as_of,
    restate_peg: PegRestatement | None = None,
    window: int = DEFAULT_WINDOW,
) -> ChosenValuation:
    """把历史 PE 表换成**盘后选股口径**的 ``选用PE``：评估日那行前瞻优先，其余照旧。

    「有前瞻就用前瞻，没有就用历史」——逐标的降级，判据是 :func:`forward_available`
    （评估日 ≥ 文件写入日，且原式的 ``用前瞻`` 成立）。

    「没有前瞻」有两种，走的是两条分支、报的也是两种话：市场**没有这族文件**（北交所）是
    已知缺口，安静退回；市场**有文件但读不到**才记进 ``unreadable`` 让人去看。合起来报会让
    盘后选股每天为几百只北交所标的喊「目录指错了吗」，真正指错的那天就被淹了。

    **为什么只动评估日那一行。** 本地的一致预期是快照、没有历史（ADR-0006 修订二），故历史
    各行**没有**前瞻值可取——它们只能是历史口径。这也顺带给出了本模块最重要的那条性质：
    任何早于文件写入日的评估日上，整张表逐格不变，于是**回测里前瞻永不生效**（回测的评估日
    全在过去）。有一条测试钉着它。

    **百分位是重算的**，且窗口里混着两种口径：历史的那些点是历史 PE，评估日那点是前瞻 PE。
    这是机械结果，不是选择——历史那一段确实没有前瞻可取。代价是**偏移**：一致预期 EPS 高于
    实际 EPS 的标的，``PE前瞻 < PE历史``，于是显得更便宜、更容易过「PE 百分位低」那道门。
    要用这张表就得认下这个偏移，别把它当成两把尺子量出来的同一个数。

    **PEG 只在给了 ``restate_peg`` 时重述**，且只重述评估日那一行，规则同原式::

        PEG := IF(选用PE>0 AND ABS(选用增)>0.1 AND 财年T>0, 选用PE/选用增, DRAWNULL)

    重述的三种落点，判据是「财年T **知道**还是**不知道**」：

    ====================  =========================================================
    ``财年T`` 的状态      评估日那一格 PEG
    ====================  =========================================================
    知道，且 > 0          用 ``选用PE ÷ 选用增``（走前瞻时两者都是预测口径）
    知道，且 ≤ 0          **缺失**——原式的第三道守卫（「没有一致预期就没有 PEG」）
    不知道                **不动**，退回历史那一格
    ====================  =========================================================

    「知道就是 0」的是两种：文件里这只票没有一致预期，以及本机压根没有这个市场的文件
    （``unsupported``，北交所）。这两种都**抹掉** PEG。
    「不知道」的也有两种：评估日早于文件写入日（快照不可用，见 :func:`snapshot_admissible`），
    以及文件读不到（``unreadable``，多半是 ``--forward-root`` 指错目录）。这两种**一格不动**：
    前者是那条不变式的要求，后者是不拿一个配置错误去悄悄改候选集——它已经另行报出来了。

    ``peg_missing`` 报的是评估日那一行**最后**缺了哪些格：从结果数，不从原因数。抹掉的在里头，
    被另外两道守卫（预测净利 ≤ 0、基期亏损）挡下的也在里头，历史口径本来就缺的还在里头——
    读的人问的是「这道门这次看得见谁」，那就得按门数。

    参数:
        close: **原始价**（当时的成交价）表，日期 × 标的。PE 要用原始价，不是后复权价。
        trailing_pe: 历史口径的动态 PE 表，形状须与 ``close`` 一致。
        as_of: 评估日，必须落在两张表的索引里。
        restate_peg: 历史 PEG 表 + 算 ``增预期`` 要用的财务数据；不给就整段跳过 PEG。
    """
    frames = [("close", close), ("trailing_pe", trailing_pe)]
    if restate_peg is not None:
        frames.append(("restate_peg.trailing_peg", restate_peg.trailing_peg))
    for name, frame in frames:
        if not _same_labels(close, frame):
            raise ValueError(
                f"close 与 {name} 的标的/交易日不一致（{name} 对不上）——"
                "两者必须同源同形，否则会按标签各对齐各的、静默错位"
            )

    from .errors import MarketDataError
    from .valuation import pe_percentile

    # `as_of` 收 `date` / `datetime` / 字符串，与 `Screen.apply` 同宽；但对表下笔时得是
    # `Timestamp`——`DatetimeIndex` 不做隐式转换，拿 `date` 去比会**恒不命中**。
    stamp = pd.Timestamp(as_of)
    if stamp not in close.index:
        raise ValueError(
            f"评估日 {as_of} 不在表里——`选用PE` 只写在评估日那一行，日子对不上就无处可写。"
            "调用方须传一个数据里真有的交易日（见 `mbt screen` 的评估日解析）。"
        )

    chosen = trailing_pe.copy()
    peg = None if restate_peg is None else restate_peg.trailing_peg.copy()
    financials = None if restate_peg is None else restate_peg.financials
    # 复用一次装载：不给 tables 的话，下面每个标的都会自己 glob 一遍 cw 目录。
    tables = None if financials is None else _tables_of(financials)

    went_forward: list[str] = []
    unsupported: list[str] = []
    unreadable: list[str] = []

    for symbol in chosen.columns:
        if not gpone.covers(symbol):
            # 已知的覆盖缺口（北交所）：PE 退回历史，不报警——见 ChosenValuation 的说明。
            unsupported.append(symbol)
            if peg is not None:
                # 但 PEG 不能退：本机没有这个市场的文件，就意味着原式的 `财年T` 在这儿
                # **恒为 0**——「知道就是 0」，故按守卫取缺失。
                peg.at[stamp, symbol] = float("nan")
            continue
        try:
            admissible = snapshot_admissible(symbol, gpone=gpone, as_of=stamp)
        except MarketDataError:
            # 这个市场有文件、但读不到。降级为历史，但**点名记下**——指错目录时要看得出来。
            # PEG 同样不动：这里连「有没有预期」都读不到，属于「不知道」，不拿它去改候选集。
            unreadable.append(symbol)
            continue
        if not admissible:
            # 快照不能用在今天（评估日早于写入日）：整格不动。回测走的就是这一支。
            continue

        fiscal_year = gpone.value(symbol, FIELD_FISCAL_YEAR) or 0.0
        if not uses_forward(fiscal_year, stamp.year):
            if fiscal_year <= 0 and peg is not None:
                # 原式的第三道守卫：财年T<=0 即「没有一致预期」，那就没有 PEG。
                peg.at[stamp, symbol] = float("nan")
            continue

        price = close.at[stamp, symbol]
        if pd.isna(price):
            continue
        forward_pe = pe_forward(
            gpone.value(symbol, FIELD_EPS_T) or 0.0,
            gpone.value(symbol, FIELD_PE_EXPECTED) or 0.0,
            close=float(price),
        )
        chosen.at[stamp, symbol] = forward_pe
        went_forward.append(symbol)

        if peg is not None:
            # `选用增` 就是走前瞻那一支的 `增预期`；不走前瞻的标的上面 `continue` 掉了，
            # 那一格仍是历史 PEG（= PE历史 ÷ 增历史），与原式的成对切换等价。
            net_profit_t = gpone.value(symbol, FIELD_NET_PROFIT_T) or 0.0
            base = _base_annual_profit(financials, symbol, tables=tables)
            peg.at[stamp, symbol] = pe_growth_ratio(forward_pe, growth_expected(net_profit_t, base))

    return ChosenValuation(
        pe=chosen,
        pe_percentile=pe_percentile(chosen, window),
        peg=peg,
        forward_symbols=tuple(went_forward),
        unsupported=tuple(unsupported),
        unreadable=tuple(unreadable),
        # **从表里数**，而不是在上面各分支顺手记：守卫有三条，分头记就会漏掉后加的那条。
        # 这样报出来的才是那句实话——「评估日那天 PEG 这道门看不见谁」。
        peg_missing=() if peg is None else _missing_on(peg, stamp),
    )


def _missing_on(frame: pd.DataFrame, stamp) -> tuple[str, ...]:
    """某一行缺的那些格是谁。数「这道门看不见谁」只能从结果数，不能从原因数。"""
    row = frame.loc[stamp]
    return tuple(symbol for symbol in row.index if pd.isna(row[symbol]))


def _same_labels(left: pd.DataFrame, right: pd.DataFrame) -> bool:
    """两张表是否同日同标的——顺序不比，集合与索引类型要比。"""
    return (
        left.shape == right.shape
        and left.index.equals(right.index)
        and left.columns.equals(right.columns)
    )
