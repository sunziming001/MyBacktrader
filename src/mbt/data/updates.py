"""更新检查：数据变了没有、变了要不要重跑（票据 #9）。

## 这里没有「增量解析」

AC 原文要求「记录每个标的已解析到的最后日期，只追加新记录」。**那件事在本项目里没有落点**：
`TdxDataSource.daily()` 每次都从 `.day` **全量解析**，没有任何缓存或已解析边界的概念（规格提过的
「列式存储」从未落地）。而实测全量重解析并不痛——**裸读全市场 93 秒、走正门约 5 分钟**。

故本模块把「增量」重定义为**「源数据变更的检测与报告」**：它回答的问题变成**「我的数据变了没有、
上次的回测结果还算不算数」**。这是**改变了 AC 的意图**，不是实现了它——票据正文记着这次改写。

## 为什么不用 mtime 判变化

通达信客户端**每天重写** `.day` 文件（实测本机文件 mtime 全是近期）。若用 ``mtime + size``
判「有没有变」，会**每天报「全部标的都变了」**，这个功能就废了。本项目在 #21 与 #8 已各踩过
一次同类问题（缓存键用错依据）。

判据改用**文件内容里能自证的东西**：**最后 :data:`TAIL_BARS` 根 K 线的日期与收盘价**。于是三种
情形可分：纯追加（末段不变、根数变多）、回补/修正（**末段值变了**）、无变化。

## 「停牌」这个词用得克制

本模块报的是**事实**：「这段没有成交记录」。它**不**直接断言「停牌」——因为数据商偶发缺行也会
长成同一个样子（那是数据缺陷，由 :mod:`mbt.data.anomaly` 那一层负责）。故两个归类名都写成
「无成交」/「空缺」，而不是「停牌」。

- **区间内**：其后已恢复成交，故**可判为非退市**（退市不会恢复）；
- **尾部**：其后无数据，**停牌与退市不可区分**（见下）。

理由：自某日起无 K 线，既可能是长期停牌（A 股真实存在停牌一年以上），也可能是退市，**在数据里
同形**。实测全市场 5,898 只股票里尾部空缺 > 60 个交易日的有 325 只，多为 1997–2002 的老退市股，
但**没有依据**把它们与长期停牌分开。本项目一贯拒绝用代理变量去猜（见 :mod:`mbt.data.anomaly`）。

## 「交易日」从哪来，以及它的**前提**

本地没有交易日历文件，故用**被检查的那批标的**观测到的日期并集当市场交易日——只要有一个标的在
该日成交，那一天就是交易日。实践中这足够（市场开门的每一天总有人成交）。

**但这个做法有一个前提**：日历必须来自**多个**标的，否则**两类缺口都会失真**：

- **区间内缺口**：只检查一个标的时，它自己停牌的日子会**连同从日历里消失**，于是缺口测不出来；
- **尾部空缺**：``market_last`` 取自日历的末尾，故只检查一个过时标的时，日历的末尾就是它自己的
  末根——**尾部空缺恒为零**，静默漏报。

故调用方应当传**全市场（或至少一批互不相关的标的）**。CLI 的 ``--limit`` / ``--symbols-file``
会削弱这一点，故它们被使用时**会打印警告**。两条局限各有测试钉住。
"""

from __future__ import annotations

import csv
import datetime as dt
import hashlib
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from .errors import MarketDataError
from .instrument import is_stock
from .tdx import TdxDataSource

#: 边界摘要取最后多少根 K 线。取 5 是权衡：足以发现「最近被回补/修正」，又不至于让摘要对
#: 很久以前的历史变化过敏。**代价是显式的**：早于这 5 根之内的**数值**改动测不出来
#: （根数的增删仍能测出来——见 :func:`compare`）。
TAIL_BARS = 5

#: 变更的归类。
UNCHANGED = "无变化"
APPENDED = "纯追加"
REVISED = "回补/修正"
NEW = "新标的"
MISSING = "已消失"

#: 缺口与尾部空缺的归类。
INSIDE_GAP = "区间内无成交"
TAIL_GAP = "尾部空缺"

#: 需要重跑的归类。
NEEDS_RERUN = (REVISED, MISSING)

#: 边界清单的列。
BOUNDARY_COLUMNS = ("symbol", "first_date", "last_date", "bars", "tail_digest")


@dataclass(frozen=True)
class SymbolBoundary:
    """一个标的的数据边界。

    属性:
        symbol: 标的符号。
        first_date: 首根 K 线的日期。
        last_date: 末根 K 线的日期。
        bars: 根数。
        tail_digest: **最后 :data:`TAIL_BARS` 根 K 线的日期与收盘价**的摘要。
    """

    symbol: str
    first_date: dt.date
    last_date: dt.date
    bars: int
    tail_digest: str


@dataclass(frozen=True)
class GapReport:
    """一处没有成交的区间。

    属性:
        symbol: 标的符号。
        kind: :data:`INSIDE_GAP`（区间内无成交，其后恢复）或 :data:`TAIL_GAP`（自某日起无数据）。
        start: 缺失的首个**交易日**（按市场日历）。
        end: 缺失的末个交易日；``TAIL_GAP`` 时即最后一根 K 线的日期。
        missing: 缺失的**交易日数**（按市场日历数出，不是估算）。
        resumed: 恢复成交的那一天；``TAIL_GAP`` 时为 ``None``。
    """

    symbol: str
    kind: str
    start: dt.date
    end: dt.date
    missing: int
    resumed: dt.date | None

    @property
    def undecidable(self) -> bool:
        """**停牌与退市**是否不可区分。

        尾部空缺如此——其后再无数据，两种可能的形状完全一样。
        区间内空缺则**可判为非退市**：它其后已恢复成交，而退市不会恢复。

        （区间内空缺仍有另一层不确定：数据商偶发缺行也会长成这个样子。那不是「停牌还是退市」
        的问题，而是数据缺陷，由 :mod:`mbt.data.anomaly` 那一层负责——故这里的标签写「无成交」
        这个**事实**，不写「停牌」这个**诊断**。）
        """
        return self.kind == TAIL_GAP


@dataclass(frozen=True)
class UpdateReport:
    """一次更新检查的结果。

    属性:
        by_kind: 归类 → 标的列表。
        gaps: 缺口明细。
        boundaries: **新的**边界清单（可直接落盘作为下次的基线）。
        calendar_days: 市场交易日数（由**被检查的**那批标的的日期并集得到——见模块说明的局限）。
        unreadable: 取不到数据而**没被检查**的标的。它们既不算变化、也不该静默消失。
        compared_with: 上次基线的标的数；``None`` 表示首次建立基线。
    """

    by_kind: dict[str, list[str]]
    gaps: tuple[GapReport, ...]
    boundaries: dict[str, SymbolBoundary]
    calendar_days: int
    unreadable: tuple[str, ...] = ()
    compared_with: int | None = None

    @property
    def revised(self) -> list[str]:
        """**让上次的回测结果作废**的标的：被回补/修正的，加上从数据源消失的。"""
        return self.by_kind.get(REVISED, []) + self.by_kind.get(MISSING, [])

    @property
    def needs_rerun(self) -> bool:
        """要不要重跑。有回补/修正、或有标的从数据源消失时必须。"""
        return bool(self.revised)

    def counts(self) -> dict[str, int]:
        return {kind: len(symbols) for kind, symbols in self.by_kind.items()}

    def gaps_by_kind(self, kind: str) -> list[GapReport]:
        return [gap for gap in self.gaps if gap.kind == kind]


def boundary_of(symbol: str, prices: pd.DataFrame) -> SymbolBoundary:
    """由价格表算边界。空表报错——没有数据的标的应当先被过滤掉。

    摘要**永远取这份数据自身的最后 :data:`TAIL_BARS` 根**，故边界是**自洽**的：
    「截至 ``last_date``，末 5 根是这些」。与上次的比较**不在这里做**——比较要把摘要
    **回到上次的窗口**去重算，那是 :func:`compare` 的事。

    .. note::

        本模块的第一版让 ``boundary_of`` 收一个「锚」，把摘要算在上次的窗口上、却把新的
        ``last_date`` 记进去——于是边界自相矛盾，**下一次比较的锚又跟着移**，导致**连续追加
        从第二次起全被误判成「回补/修正」**。自洽 + 在比较侧重算，才不会有这个毛病。
    """
    if prices.empty:
        raise MarketDataError(f"{symbol} 没有 K 线，算不出边界")
    return SymbolBoundary(
        symbol=symbol,
        first_date=prices.index[0].date(),
        last_date=prices.index[-1].date(),
        bars=len(prices),
        tail_digest=_digest(prices),
    )


def _digest(prices: pd.DataFrame) -> str:
    """末 :data:`TAIL_BARS` 根的**日期与收盘价**的摘要。"""
    tail = prices.iloc[-TAIL_BARS:]
    payload = ",".join(
        f"{stamp:%Y%m%d}:{close:.4f}"
        for stamp, close in zip(tail.index, tail["close"], strict=True)
    )
    return hashlib.sha256(payload.encode("ascii")).hexdigest()[:16]


def compare(before: SymbolBoundary, prices: pd.DataFrame) -> str:
    """上次的边界在当前数据里还成不成立。

    先把当前数据**截到上次的末根**，再拿那段与上次的记录比——两步都必要：

    1. **根数**：截出来的根数必须与上次记的 ``bars`` 相同。少了是截断/删行，多了是回补。
       这一条能测出**任何位置**的增删，包括末 5 根之外的。
    2. **摘要**：那段数据自身的末 5 根必须与上次记的摘要一致。这一条测出**末 5 根内的数值
       改动**。

    第 2 条只覆盖末 5 根，是 :data:`TAIL_BARS` 的既定代价（要覆盖全历史就得对整段做内容摘要，
    那会随历史增长而变慢）。第 1 条把「增删」这一类补上了，故代价只落在「很早以前的**数值**
    改动」上——那类改动极罕见，且诚实写在文档里。
    """
    window = prices.loc[: pd.Timestamp(before.last_date)]
    if len(window) != before.bars:
        return REVISED
    if _digest(window) != before.tail_digest:
        return REVISED
    return UNCHANGED if len(prices) == before.bars else APPENDED


def load_boundaries(path) -> dict[str, SymbolBoundary]:
    """读上次的边界清单。文件不存在时报错——**不静默当成首次**。

    「文件不在」与「首次运行」是两件事：前者多半是路径写错，而静默当成首次会让所有「回补/修正」
    退化成「新标的」，那个功能就白做了。
    """
    path = Path(path)
    if not path.is_file():
        raise MarketDataError(
            f"边界清单不存在：{path}。首次运行请先不带 --baseline 建立基线，"
            f"而不是把一个不存在的路径当成「还没有清单」。"
        )

    out: dict[str, SymbolBoundary] = {}
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        missing = set(BOUNDARY_COLUMNS) - set(reader.fieldnames or ())
        if missing:
            raise MarketDataError(f"{path} 缺少列 {sorted(missing)}，不是本工具写出的清单")
        for row in reader:
            out[row["symbol"]] = SymbolBoundary(
                symbol=row["symbol"],
                first_date=dt.date.fromisoformat(row["first_date"]),
                last_date=dt.date.fromisoformat(row["last_date"]),
                bars=int(row["bars"]),
                tail_digest=row["tail_digest"],
            )
    return out


def save_boundaries(path, boundaries: dict[str, SymbolBoundary]) -> Path:
    """写边界清单。CSV 而非二进制：要能被人打开看、被 git 追踪。"""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(BOUNDARY_COLUMNS)
        for symbol in sorted(boundaries):
            item = boundaries[symbol]
            writer.writerow(
                [item.symbol, item.first_date, item.last_date, item.bars, item.tail_digest]
            )
    return path


def check_updates(
    symbols,
    *,
    tdx_root,
    previous: dict[str, SymbolBoundary] | None = None,
) -> UpdateReport:
    """逐个标的算边界，与上次比较，并按**市场交易日日历**报告缺口。

    参数:
        symbols: 要检查的标的。非股票与取不到数据的会被跳过（跳过本身不算「变化」）。
        tdx_root: 通达信 ``vipdoc`` 根目录。
        previous: 上次的边界清单；``None`` 表示**首次建立基线**。

    返回:
        :class:`UpdateReport`。

    分两趟：先收齐各标的的日期（由此得到**市场交易日日历**），再算缺口。日历必须来自全市场——
    单看一个标的的日期分不出「它停牌」与「那天不是交易日」。
    """
    source = TdxDataSource(tdx_root)

    frames: dict[str, pd.DataFrame] = {}
    unreadable: list[str] = []
    for symbol in symbols:
        if not is_stock(symbol):
            continue
        try:
            prices = source.daily(symbol)
        except MarketDataError:
            # 取不到数据 ≠ 没变化。**必须报出来**，否则它会静默地从基线里消失，
            # 而「上次检查过、这次没了」恰恰是需要人看的事（ADR-0005 的异常报错）。
            unreadable.append(symbol)
            continue
        if prices.empty:
            unreadable.append(symbol)
        else:
            frames[symbol] = prices

    calendar = _market_calendar(frames)

    boundaries: dict[str, SymbolBoundary] = {}
    by_kind: dict[str, list[str]] = {}
    gaps: list[GapReport] = []

    for symbol, prices in frames.items():
        current = boundary_of(symbol, prices)
        boundaries[symbol] = current
        before = (previous or {}).get(symbol)
        kind = NEW if before is None else compare(before, prices)
        by_kind.setdefault(kind, []).append(symbol)
        gaps.extend(_gaps(symbol, prices.index, calendar))

    # 上次基线里有、这次取不到的：标的从数据源**消失**了。同样让旧结果作废，且必须报出来。
    for symbol in sorted(set(previous or ()) - set(frames) - set(unreadable)):
        by_kind.setdefault(MISSING, []).append(symbol)

    return UpdateReport(
        by_kind=by_kind,
        gaps=tuple(gaps),
        boundaries=boundaries,
        calendar_days=len(calendar),
        unreadable=tuple(unreadable),
        compared_with=None if previous is None else len(previous),
    )


def _market_calendar(frames: dict[str, pd.DataFrame]) -> pd.DatetimeIndex:
    """市场交易日 = **被检查的**那批标的观测到的日期并集（局限见模块说明）。"""
    if not frames:
        return pd.DatetimeIndex([])
    union = set()
    for prices in frames.values():
        union.update(prices.index)
    return pd.DatetimeIndex(sorted(union))


def _gaps(symbol: str, index: pd.DatetimeIndex, calendar: pd.DatetimeIndex) -> list[GapReport]:
    """按市场日历找缺口：**区间内**的计入 :data:`INSIDE_GAP`，最后的计入 :data:`TAIL_GAP`。

    缺口天数取该标的缺失的**市场交易日**数——那是数出来的，不是按日历天估算的。
    """
    found: list[GapReport] = []
    if len(index) == 0:
        return found

    own = pd.DatetimeIndex(index)
    market_last = calendar[-1] if len(calendar) else own[-1]

    # 尾部：该标的最后一根之后、市场最后一天（含）之间的交易日。
    tail = calendar[(calendar > own[-1]) & (calendar <= market_last)]
    if len(tail):
        found.append(
            GapReport(
                symbol=symbol,
                kind=TAIL_GAP,
                start=tail[0].date(),
                end=own[-1].date(),
                missing=len(tail),
                resumed=None,
            )
        )

    # 区间内：相邻两根之间缺失的交易日。
    for position in range(1, len(own)):
        previous, current = own[position - 1], own[position]
        missing = calendar[(calendar > previous) & (calendar < current)]
        if len(missing) == 0:
            continue
        found.append(
            GapReport(
                symbol=symbol,
                kind=INSIDE_GAP,
                start=missing[0].date(),
                end=missing[-1].date(),
                missing=len(missing),
                resumed=current.date(),
            )
        )

    return found


def classify(before: SymbolBoundary | None, current: SymbolBoundary) -> str:
    """**不要用这个。** 保留名字只为让旧调用点显式失败——

    它只拿到两份**边界**，而正确的判断需要**当前的价格表**（要把摘要回到上次的窗口去重算）。
    本模块的早期版本正因如此把「连续追加」从第二次起全判成「回补/修正」。请用
    :func:`compare`。
    """
    raise NotImplementedError(
        "classify 已被 compare 取代：判断变化需要当前的价格表，而不只是两份边界。"
    )
