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
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from ..rules import RuleTable
from .adjust import AdjustmentEvent, backward_adjusted, forward_adjusted
from .anomaly import require_no_anomalies
from .dilution import DilutionVerdict, resolve_dilution
from .gbbq import GbbqDataSource
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
    """

    symbol: str
    prices: pd.DataFrame
    events: tuple[AdjustmentEvent, ...]
    verdicts: tuple[DilutionVerdict, ...] = ()

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


def load_market_data(
    symbol: str,
    *,
    tdx_root: str | Path,
    gbbq_path: str | Path,
    rules: RuleTable | str | Path | None = None,
    start=None,
    end=None,
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

    返回:
        含原始价与权息事件的 :class:`MarketData`。

    抛:
        MarketDataError: 行情文件缺失 / 损坏，或**区间内**存在无法用公司行为解释的越界跳空。
        RuleTableError: 符号不是规则表覆盖的股票（指数、基金、债券）。
    """
    table = rules if isinstance(rules, RuleTable) else RuleTable.load(rules)

    prices = TdxDataSource(tdx_root).daily(symbol)
    raw_events = tuple(GbbqDataSource(gbbq_path).events(symbol))

    # 校验只做在**回测区间**内，理由见上面的 docstring。
    window = _window_of(prices, start, end)

    # 判定一次，**复权与质检吃同一份**结果。否则会出现「复权已经不算它了，质检却还在按它
    # 报异常」这种自相矛盾（与 ADR-0002 里「涨跌停带必须与撮合共用同一个带」同类）。
    events, verdicts = resolve_dilution(window, raw_events, symbol, table)

    require_no_anomalies(window, symbol, table, events)

    return MarketData(symbol=symbol, prices=prices, events=events, verdicts=verdicts)


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
