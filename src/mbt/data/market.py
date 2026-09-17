"""经质检的行情数据：把「原始价 + 权息事件」打包成一个**已做过越界检查**的对象。

检查是构造的一部分，故「拿到对象」即「数据已质检」（ADR-0005）。

## 为什么要有这个入口

越界检查（:func:`mbt.data.anomaly.require_no_anomalies`）要三样东西同时在场：原始价、
权息事件、规则表。三者来自不同地方（行情目录、权息文件、出厂规则表），调用方很容易
只取前两者就开跑，坏数据于是被沉默接受。把三者的组合收进这一个构造函数后，「拿到
:class:`MarketData`」本身就证明检查跑过了——缺口纪律（ADR-0005）不再依赖调用方记得。

## 本对象只装原始价

检查只对**原始价**成立，故本对象也只保存原始价，复权视图由 :meth:`MarketData.backward_adjusted`
用时计算（ADR-0003 本就要求不落盘已复权序列）。把后复权价也塞进来的话，就会诱使调用方
对它做同一次检查，而那是错的：后复权价已经把除权跳空抹平，再拿权息事件去重算限价基数
等于把交给复权的那部分**再折算一次**。例如某股 10.00 元 10 送 10，原始价次日 5.00、
后复权两日都是 10.00，两者都合法；但按事件重算的基数是 5.00、限价带 [4.5, 5.5]，
后复权的 10.00 就被判越界——凭空造出假异常。

## 适用范围：仅股票

检查要按板块与 ST 状态取限幅，而规则表只覆盖股票（见 :meth:`mbt.rules.RuleTable.board_of`）。
指数、基金、债券的代码传进来会抛 ``RuleTableError``。这不是遗漏：这些品种没有涨跌幅
限价制度，硬套股票的带才是错的。
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass, replace
from pathlib import Path

import pandas as pd

from ..rules import RuleTable
from .adjust import AdjustmentEvent, backward_adjusted, forward_adjusted
from .anomaly import require_no_anomalies
from .dilution import DilutionVerdict, resolve_dilution
from .gbbq import GbbqDataSource
from .st import InferredStPeriod, infer_st_periods
from .tdx import TdxDataSource


@dataclass(frozen=True)
class MarketData:
    """一个标的的**经质检**行情：原始价 + 权息事件。

    只能由 :func:`load_market_data` 造出——那条路径上跑过越界检查，故本对象的存在
    即是「无越界且无公司行为可解释的跳空」的证据。

    属性:
        symbol: 标的符号，形如 ``sh600000``。
        prices: **原始**价字段宽表，交易日为索引。
        events: **判定后**、参与价格复权的除权除息事件（含序列之外的）。
            判为「未稀释」者已丢掉其稀释成分、只保留现金分红（见
            :mod:`mbt.data.dilution`）。
        verdicts: 逐条事件的判定记录，含低置信标记。默认空元组——直接构造
            ``MarketData`` 的调用方（如测试）不必提供。
        st_periods: **推断出的** ST 期间（票据 #53）。默认空元组。它由价格数据推出
            （判据与已知漏检见 :mod:`mbt.data.st`），**不是**权威的 ST 名单——本地数据
            没有股票名称。它的用途有二：收紧涨跌幅带（引擎据此撮合），以及给股票池一个
            可用的「排除 ST」依据（此前那条规则是**空条件**）。
    """

    symbol: str
    prices: pd.DataFrame
    events: tuple[AdjustmentEvent, ...]
    verdicts: tuple[DilutionVerdict, ...] = ()
    st_periods: tuple[InferredStPeriod, ...] = ()

    def was_st(self, on: dt.date) -> bool:
        """该日是否处于**推断出的** ST 期间。判不出来时返回 ``False``（不猜）。"""
        return any(period.covers(on) for period in self.st_periods)

    def backward_adjusted(self, as_of: dt.date | dt.datetime | None = None) -> pd.DataFrame:
        """**后复权**视图：回测用。首根 K 线价格不变，除权日不再有假跳空。

        参数:
            as_of: 评估日；晚于它的事件（含「已公告未除权」）不纳入，防前视偏差
                （ADR-0006）。默认取序列末根 K 线的日期。
        """
        return backward_adjusted(self.prices, self.events, as_of=as_of)

    def forward_adjusted(self, as_of: dt.date | dt.datetime | None = None) -> pd.DataFrame:
        """**前复权**视图：展示用。最新价等于原始价，历史价被下调。"""
        return forward_adjusted(self.prices, self.events, as_of=as_of)


def backward_adjusted_markets(markets) -> list[MarketData]:
    """把一批行情换成**后复权**视图：选股、股票池与撮合都该落在同一条序列上（ADR-0003）。

    为什么要有这个函数：`mbt backtest` 与 `mbt screen` 都要一份「能拿来算信号的价格」，
    而它们此前各写一遍这段构造（回测写过、选股**漏了**，于是同一个规则在原始价与后复权价
    两条序列上被评估——票据 #73）。两种写法只要分头演进，就会再次漂开；集中在**一处**定义
    之后，「两条路的面板是同一条序列」由构造保证，而不是靠两边各自记得。

    **返回的是同一个标的的另一条价格序列，不是一份新数据**：`symbol` / `verdicts` /
    `st_periods` 原样带过来（它们描述的是标的本身，与价格尺度无关），只有两样变了——

    - ``prices`` 换成 :meth:`MarketData.backward_adjusted` 的结果；
    - ``events`` 清空。它们已经折进价格了，留着会诱使调用方再折一次；而按事件给后复权价
      重算涨跌停基数是**错的**（除权跳空已被抹平，再折算等于多算一遍），见本模块的说明。

    参数:
        markets: 一串 :class:`MarketData`（原始价，已过越界检查）。

    返回:
        等长的列表，逐项对应，顺序不变。
    """
    return [replace(market, prices=market.backward_adjusted(), events=()) for market in markets]


def load_market_data(
    symbol: str,
    *,
    tdx_root: str | Path,
    gbbq_path: str | Path,
    rules: RuleTable | str | Path | None = None,
    start=None,
    end=None,
    listing_date=None,
    quality_bars=None,
) -> MarketData:
    """读取一个股票的原始行情与权息事件，**并在返回前做越界检查**。

    检查不设开关：本函数就是「要一份能拿去回测的行情」的唯一正门，放行坏数据的档位
    没有存在的理由。全市场扫描若需要「跳过坏标的、继续扫」的语义，请在调用处
    ``except MarketDataError`` 记下该标的存疑——而不是让它以干净的名义混进样本。

    参数:
        symbol: 标的符号，形如 ``sh600000``。
        tdx_root: 通达信 ``vipdoc`` 根目录。
        gbbq_path: 权息事件文件路径。
        rules: 规则表，``RuleTable`` 或 TOML 路径。默认取出厂规则表
            ``src/mbt/rules/a_share.toml``。
        start / end: **回测区间**（含两端），``None`` 表示不设该侧边界。

    关于 ``start`` / ``end``::

        越界检查**只在这个区间内做**。理由是纪律本身：ADR-0005 要的是「不在不可信的
        数据上交易」，而区间之外的数据本来就不交易。本机实测——``sh600519`` 有一根
        2006-05-25 的复牌 K 线越出涨跌幅带（股改复牌首日**不设涨跌幅**，而本地数据无从
        得知这一点），逐条校验会让**整只茅台**被拒收；可那段数据落在任何 2010 年之后的
        回测之外。全市场抽样里，同一现象涉及 5.7% 的标的，其中九成的坏日子都在 2015-08
        之前（票据 #45）。

        返回的 ``prices`` 仍是**完整历史**（不是切过的），故调用方照常自行切片；
        判定记录与事件集则只覆盖区间内的事件。

        ``listing_date`` 用于**豁免制度空窗**（ADR-0013）：注册制下上市后前 5 个交易日
        不设涨跌幅，那几天的「越界」是制度使然，不是坏数据。实测全市场被拒标的里 86%
        的越界落在第 2~5 根，全是这一类误报。不给则一处都不豁免（改动前的严格口径）。

        ``quality_bars`` 把**质检窗口**从末端再收窄 N 根（ADR-0014），覆盖两处检查：
        :func:`~mbt.data.dilution.resolve_dilution` 的稀释判定与 ``require_no_anomalies``
        的越界检查。给它的理由与 ``start``/``end`` 是同一条，只是后者不够用：**选股没有
        回测区间**。``mbt screen`` 要的是「评估日往前若干根」的那段行情（各规则最深回看
        1000 根，见 :data:`~mbt.cli.DEFAULT_QUALITY_BARS`），而它的评估日在取数时**还没
        定出来**（``--as-of`` 可省，默认取「最近一个齐全的交易日」，那要读数据才知道）。
        于是 ``start`` 无从算起，检查只能覆盖全部历史——全市场实测因此**多拒了 1,028 只
        （17.4%）**，它们的越界全在十几年前（870/990 在 2006 年及之前，多为股改复牌或
        那时的除权判定）。``sh600547`` 就是其中之一：它被 **2006-03-31 的一处除权判定**
        挡住，而它在 2026-09-15 过得了全部 8 道 B1 门。

        取了 ``end`` 时，这 N 根是**相对 ``end`` 的末端**（即「评估日往前 N 根」）；不取
        ``end`` 时相对数据的末端。故 ``mbt screen`` 已有的 ``end=--as-of`` 会把窗口上界
        定住，这一项只需补下界。

        .. warning::

            **它会把窗口之外的权息事件降级为「不判定」**：``resolve_dilution`` 对落在序列
            之外的事件记为 ``OUTSIDE_SERIES`` 并**原样保留**，故那些事件的稀释成分不再被
            置零。对选股无影响（选股用的是窗口内的行情），但**回测不要用这一项**——回测有
            ``--start``，本来就该用它把窗口定准。

    返回:
        含原始价与权息事件的 :class:`MarketData`。

    抛:
        MarketDataError: 行情文件缺失 / 损坏，或**区间内**存在无法用公司行为解释的越界跳空。
        RuleTableError: 符号不是规则表覆盖的股票（指数、基金、债券）。
    """
    table = rules if isinstance(rules, RuleTable) else RuleTable.load(rules)

    prices = TdxDataSource(tdx_root).daily(symbol)
    raw_events = tuple(GbbqDataSource(gbbq_path).events(symbol))

    # **推断 ST 期间，并用它收紧限幅。** 本地数据没有股票名称，出厂表按 ADR-0002 刻意不登记
    # ST——于是引擎会给一只 5% 的股票按 10% 撮合，把 ST 的一字板当成可成交。判据与已知漏检
    # 见 `mbt.data.st`。推断在**完整历史**上做（ST 判定要看长窗口），而校验在窗口内做。
    st_periods = infer_st_periods(prices, symbol, table)
    if st_periods:
        table = table.with_st_periods(
            {symbol: [(period.start, period.end) for period in st_periods]}
        )

    # 质检窗口可以比复权窗口**更窄**（ADR-0014）：`quality_bars` 只收窄被检查的那一段。
    # 选股没有回测区间，只剩这一个办法把「十几年前的越界与除权判定」排除在判定之外。
    #
    # 收窄的是**整段窗口**，故两处检查同时受益：`resolve_dilution` 的稀释判定与
    # `require_no_anomalies` 的越界检查。`sh600547` 栽在前者（2006-03-31 的除权判不动），
    # 而真正多拒 1,028 只的是两者之和。
    #
    # 取 `quality_bars + 1` 根而不是 `quality_bars` 根：与 `_window_of` 同一个理由——涨跌停带
    # 与稀释判定都要用**前一根**算，窗口首根若没有基数，它上面的跳空就判不出来。多要一根
    # 作基数之后，「最后 N 根」里的每一根都真的被查过。
    if quality_bars is not None and quality_bars < 1:
        raise ValueError(f"quality_bars 至少为 1（或 None 表示不限制），收到 {quality_bars!r}")
    window = _window_of(prices, start, end)
    if quality_bars is not None:
        window = window.iloc[-(quality_bars + 1) :]

    # 判定一次，**复权与质检吃同一份**结果。否则会出现「复权已经不算它了，质检却还在按它
    # 报异常」这种自相矛盾（与 ADR-0002 里「涨跌停带必须与撮合共用同一个带」同类）。
    events, verdicts = resolve_dilution(window, raw_events, symbol, table)

    require_no_anomalies(window, symbol, table, events, listing_date=listing_date)

    return MarketData(
        symbol=symbol,
        prices=prices,
        events=events,
        verdicts=verdicts,
        st_periods=st_periods,
    )


def _window_of(prices: pd.DataFrame, start, end) -> pd.DataFrame:
    """把价格表切到 ``[start, end]``（含两端），并**往前多取一根**。两端都 ``None`` 时不切。

    多取那一根是必需的：涨跌停带要用**前一根**算，而窗口首根如果就是切片的第一根，它上面的
    跳空根本无从判定（没有基数）。多取一根之后，「窗口首根上的跳空」照样能被抓到，而窗口
    **之前**那些说不清的日子仍然不参与判定——实测 ``sh600519`` 的坏日子在 2006-05-25，
    窗口从 2015-08 起时它落在多取的那一根之外（票据 #45）。

    切出来的窗口可能**为空**（该标的在区间内没有 K 线）：那是合法情形——调用方的
    ``slice_markets`` 会以 ``OUT_OF_RANGE`` 把它记为跳过，比在这里报错更合适。
    """
    if start is None and end is None:
        return prices

    left = 0
    if start is not None:
        left = max(int(prices.index.searchsorted(pd.Timestamp(start), side="left")) - 1, 0)

    right = len(prices)
    if end is not None:
        right = int(prices.index.searchsorted(pd.Timestamp(end), side="right"))

    return prices.iloc[left:right]
