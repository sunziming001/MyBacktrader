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

    返回:
        含原始价与权息事件的 :class:`MarketData`。

    抛:
        MarketDataError: 行情文件缺失 / 损坏，或存在无法用公司行为解释的越界跳空。
        RuleTableError: 符号不是规则表覆盖的股票（指数、基金、债券）。
    """
    table = rules if isinstance(rules, RuleTable) else RuleTable.load(rules)

    prices = TdxDataSource(tdx_root).daily(symbol)
    raw_events = tuple(GbbqDataSource(gbbq_path).events(symbol))

    # 判定一次，**复权与质检吃同一份**结果。否则会出现「复权已经不算它了，质检却还在按它
    # 报异常」这种自相矛盾（与 ADR-0002 里「涨跌停带必须与撮合共用同一个带」同类）。
    events, verdicts = resolve_dilution(prices, raw_events, symbol, table)

    require_no_anomalies(prices, symbol, table, events)

    return MarketData(symbol=symbol, prices=prices, events=events, verdicts=verdicts)
