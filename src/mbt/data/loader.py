"""批量取数：逐个标的走正门，失败的**跳过并留痕**（票据 #11）。

为什么这件事在库里而不在 CLI：它是**策略性的**行为，不是参数搬运——「哪些跳过、为什么、
跳过多少算太多」都要有定义、也要能被单独测。CLI 明令「不含独立业务逻辑」，而 notebook 与
脚本同样需要这件事。

为什么不「失败即整体中止」：实测约 **21%** 的含除权事件的标会被稀释判定（#20）挡下，另有
北交所 2021-11-15 之前的标的一律失败。全市场跑批时中止等于没用。**但也绝不静默**——跳过
必须出现在返回值、标准错误与产物元数据里（ADR-0005 的缺口纪律：说得出、记得住，不假装）。
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence, Sized
from dataclasses import dataclass

import pandas as pd

from ..rules.errors import RuleTableError
from .errors import MarketDataError
from .instrument import is_stock
from .market import MarketData, load_market_data

#: 跳过原因的归类。便于汇总时按类计数，而不是把一堆自由文本摆给人看。
NOT_STOCK = "非股票"
RULE_WINDOW = "规则表未覆盖"
UNTRUSTWORTHY = "数据不可信"
UNREADABLE = "读取失败"
OUT_OF_RANGE = "不在所选区间内"


@dataclass(frozen=True)
class SkippedSymbol:
    """一个被跳过的标的及其原因。

    属性:
        symbol: 标的符号。
        kind: 归类，取值见本模块的四个常量。
        detail: 原始错误消息（截断到合理长度），便于定点排查。
    """

    symbol: str
    kind: str
    detail: str


@dataclass(frozen=True)
class UniverseLoad:
    """批量取数的结果。

    属性:
        markets: 成功加载的行情，顺序与传入的标的顺序一致。
        skipped: 被跳过的标的及原因，顺序同上。
    """

    markets: tuple[MarketData, ...]
    skipped: tuple[SkippedSymbol, ...]

    @property
    def total(self) -> int:
        return len(self.markets) + len(self.skipped)

    @property
    def failure_rate(self) -> float:
        return len(self.skipped) / self.total if self.total else 0.0

    def count_by_kind(self) -> dict[str, int]:
        """按原因归类计数，供汇总打印。"""
        counts: dict[str, int] = {}
        for item in self.skipped:
            counts[item.kind] = counts.get(item.kind, 0) + 1
        return counts


def load_universe_data(
    symbols: Iterable[str],
    *,
    tdx_root,
    gbbq_path,
    rules=None,
    skip_errors: bool = True,
    start=None,
    end=None,
    listing_dates=None,
    quality_bars=None,
    progress=None,
) -> UniverseLoad:
    """逐个标的走正门取数，把失败者与失败原因一并交出。

    参数:
        symbols: 要加载的标的。**非股票一律跳过**，不尝试加载——它们没有板块，规则表会直接
            拒绝（实测全市场 12,243 个文件里只有 5,898 个是股票）。
        tdx_root: 通达信 ``vipdoc`` 根目录。
        gbbq_path: 权息文件路径。
        rules: 规则表（``RuleTable`` 或路径）。默认取出厂表。
        skip_errors: ``True``（默认）跳过加载失败的标的并留痕；``False`` 则让第一个失败
            直接抛出——**调试单个标的时用**，那时你正想知道它为什么失败。
        start / end: **回测区间**，透给 :func:`~mbt.data.market.load_market_data` 决定
            「越界检查做在哪一段」。这是**必须传对**的一项：不传就等于在整段历史上校验，
            而历史上任何一处说不清的跳空都会让整只标的被拒收——哪怕它落在你的回测之外
            （实测 5.7% 的标的是这个原因，其中九成的坏日子在 2015 之前，票据 #45）。
        listing_dates: ``{符号: 上市日}``，透给 ``load_market_data`` 用于**豁免制度空窗**
            （ADR-0013）：注册制下上市后前 5 个交易日不设涨跌幅，那几天的越界是制度使然。
            实测全市场被拒的标的里 86% 的越界落在第 2~5 根，正是这一类。
            **不给（默认）则一处都不豁免**——严格口径，改动前的行为。
        quality_bars: **质检（稀释判定 + 越界检查）只覆盖窗口末端 N 根**（ADR-0014），
            透给 ``load_market_data``。选股没有回测区间（``start`` 无从算起），只剩这一个
            办法把十几年前的越界与除权判定排除在外——全市场实测因此多拒了 1,028 只（17.4%）。
            **不给（默认）则检查覆盖整个 ``[start, end]``**，与改动前一致。
        progress: 进度上报的接收端（:class:`~mbt.progress.ProgressReporter`）。``None``
            （默认）时不输出。逐标的取数在**全市场规模下要几分钟**，而这几分钟里此前一行
            输出都没有——于是「在正常地慢」与「卡死了」从外部看一模一样（见
            :mod:`mbt.progress`）。给出时按标的粒度上报。

    返回:
        :class:`UniverseLoad`。失败率过高这件事留给调用方判断（CLI 用它决定退出码）。
    """
    markets: list[MarketData] = []
    skipped: list[SkippedSymbol] = []

    total = len(symbols) if isinstance(symbols, Sized) else None
    if progress is not None:
        progress.stage("取数", note=f"{total} 个候选标的" if total else "", unit="只")

    for position, symbol in enumerate(symbols, start=1):
        if progress is not None:
            # 报的是**已完成**的数量，故减一：这一轮还没走完。时点用当前标的——取数没有
            # 「哪一天」可言，有的是「读到哪一个」。
            progress.tick(position - 1, total, lambda name=symbol: name)
        if not is_stock(symbol):
            skipped.append(SkippedSymbol(symbol, NOT_STOCK, "品种不是股票"))
            continue

        try:
            markets.append(
                load_market_data(
                    symbol,
                    tdx_root=tdx_root,
                    gbbq_path=gbbq_path,
                    rules=rules,
                    start=start,
                    end=end,
                    listing_date=None if listing_dates is None else listing_dates.get(symbol),
                    quality_bars=quality_bars,
                )
            )
        except MarketDataError as exc:
            if not skip_errors:
                raise
            skipped.append(SkippedSymbol(symbol, UNTRUSTWORTHY, _short(exc)))
        except RuleTableError as exc:
            # 规则表查不到该日期或板块——最典型的是成交日早于**费用口径**覆盖的起点
            # （沪主板 2015-08-01、深主板 2012-06-01），或北交所 2021-11-15 之前的历史。
            if not skip_errors:
                raise
            skipped.append(SkippedSymbol(symbol, RULE_WINDOW, _short(exc)))
        except Exception as exc:  # noqa: BLE001
            # 兜底：任何其它失败也必须被记下。静默吞掉会让「跳过了什么」变成谜。
            if not skip_errors:
                raise
            skipped.append(
                SkippedSymbol(symbol, UNREADABLE, f"{type(exc).__name__}: {_short(exc)}")
            )

    if progress is not None:
        # 收尾那一次：循环里报的最后一个数是 total-1，不补这一下，阶段结束行会少一个。
        progress.tick(len(markets) + len(skipped), total, None)

    return UniverseLoad(markets=tuple(markets), skipped=tuple(skipped))


def slice_markets(
    markets: Sequence[MarketData], start=None, end=None, *, skipped: Sequence[SkippedSymbol] = ()
) -> UniverseLoad:
    """把行情切到 ``[start, end]``（含两端），落在区间外或区间内没有 K 线的标的**跳过并留痕**。

    为什么在库里而不是在 CLI：这是数据操作，且「区间选得不对会怎样」需要被定义与测试。
    CLI 只做参数搬运。

    区间由调用方（CLI 的 ``--start`` / ``--end``）给出，**默认不截断**——库不替人猜一个
    默认窗口。CLI 那边的默认值取「费用口径全可查」的最早日期，理由写在那里。

    参数:
        markets: 已加载的行情。
        start: 起始日（含）。``None`` 表示不设下限。
        end: 结束日（含）。``None`` 表示不设上限。
        skipped: **上游阶段已经记下的跳过项**，会原样并入返回值。

            ``skipped`` 是**必须**传的：调用方（CLI）先取数、再切片，而本函数只收
            ``markets``，于是「取数时跳过了谁」会在这里被整段丢掉、从报告里消失。实测一次
            300 只的运行：取数阶段跳了 84 只，而回测摘要只报「跳过 2」（那 2 是切片的），
            84 只无声无息（票据 #45 的「跳过要点名」正因此落空）。

    返回:
        :class:`UniverseLoad`，其中 ``skipped`` 含 :data:`OUT_OF_RANGE` 一类，以及 ``skipped``
        参数带进来的那些。
    """
    import datetime as dt

    def bound(value, label: str):
        if value is None:
            return None
        if isinstance(value, dt.datetime):
            return value.date()
        if isinstance(value, dt.date):
            return value
        try:
            return dt.date.fromisoformat(str(value))
        except ValueError as exc:
            raise ValueError(f"{label} 不是合法日期：{value!r}（应形如 2026-09-11）") from exc

    lower, upper = bound(start, "起始日"), bound(end, "结束日")
    if lower and upper and lower > upper:
        raise ValueError(f"起始日 {lower} 晚于结束日 {upper}，区间为空")

    kept: list[MarketData] = []
    # 上游的跳过项**原样带上**，否则它们会从报告里消失（见 docstring）。
    recorded: list[SkippedSymbol] = list(skipped)
    for market in markets:
        frame = market.prices
        if lower is not None:
            frame = frame.loc[pd.Timestamp(lower) :]
        if upper is not None:
            frame = frame.loc[: pd.Timestamp(upper)]

        if frame.empty:
            recorded.append(
                SkippedSymbol(market.symbol, OUT_OF_RANGE, f"{lower}–{upper} 区间内没有 K 线")
            )
            continue
        kept.append(
            MarketData(
                symbol=market.symbol,
                prices=frame,
                events=market.events,
                verdicts=market.verdicts,
            )
        )

    return UniverseLoad(markets=tuple(kept), skipped=tuple(recorded))


def stock_symbols(symbols: Sequence[str]) -> list[str]:
    """从扫描结果里挑出股票，保持原顺序。

    这是 CLI 的第一道过滤，也是「为什么 12,243 个文件最后只用到 5,898 个」的答案所在——
    把它做成显式函数而不是散在调用处，是为了让这条事实有地方写清楚。
    """
    return [symbol for symbol in symbols if is_stock(symbol)]


def _short(exc: BaseException, limit: int = 200) -> str:
    text = str(exc).replace("\n", " ").strip()
    return text if len(text) <= limit else text[: limit - 1] + "…"
