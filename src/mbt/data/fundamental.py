"""按报告期的财务数据：解析 `vipdoc/cw/gpcwYYYYMMDD.dat`（票据 #8）。

用上本机已存在的 2.5 GB 财务数据资产，以最低成本兑现「非亏损」这类过滤。

## 文件布局（已在真实文件上逐字节复核）

```
20 字节头 + nstk × 11 字节索引 + nstk × 2336 字节数据块（= 584 个 float32）
```

帧恒等式 ``20 + nstk×11 + nstk×2336 == filesize`` 在**本机全部 147 个文件上成立**（非空 121、
空 26）。头部**按偏移读**，不用 struct 串：

===========  ======  ====================================
偏移         类型    含义
===========  ======  ====================================
0            u16     格式版本（本机恒为 1）
2            u32     **报告期** YYYYMMDD
6            u16     nstk（记录数）
8            u16     未用
10           u16     索引项大小（本机恒为 11）
12           u32     数据块大小（本机恒为 2336）
16           u32     未用
===========  ======  ====================================

.. warning::

    研究文档 `docs/research/tdx-cw-format.md` 曾把头部写成一个 struct 串
    ``'<1hI1H3L'``，而该串只产出 6 个值、与它自己的偏移表对不上。以上表为准——它是
    用**显式偏移**在 147 个文件上验过帧恒等式的那个。

## 公告日：这是本模块存在的理由

每块里 **float 索引 313**（字节偏移 1252）是 ``财报公告日期``，以 float32 存 ``YYMMDD`` /
``YYYYMMDD`` 整数。实测 ``000001`` 的 2025 年报 → 2026-03-21、``600519`` → 2026-04-17，
与真实披露日吻合。

**按报告期对齐等于提前数月知道尚未公布的财报**（ADR-0006 的前视偏差）。故取数一律走
「评估日 → 该日**已公告**的最近一期」（:meth:`CwDataSource.as_of`），而不是「最近一期」。

**2005 年之前不可用**：实测 1995 年报里 670/673 条的公告日**等于报告期**（占位符），
2001 年报有一批约 700 天的荒诞滞后。故 `公告日 > 报告期末` **且** `公告日年份 ≥ 2005`
才认；否则该期视为**无财务数据**，相关过滤返回假（排除），并如实记录原因。

## 累计 vs 单季：字段名自带区分

`gpcw` 在同一块里**同时**存累计（年初至今）与单季值，且**不同偏移**。字段名单独看
（``营业收入`` vs ``其中：营业收入``）**分不出哪个是哪个**，必须按索引取。故本模块对外的
字段名**显式带后缀**：``revenue_ytd`` / ``revenue_quarter``、``net_profit_ytd`` /
``net_profit_quarter``。刻意**不提供**一个含糊的 ``revenue``——混用会让同比/环比算错，
而那种错误不会报错。

## 幸存者偏差（数据事实，不是决策）

`gpcw` 的历史文件是**按当前证券主表重新生成的**：实测已知退市代码（`600001`、`600806`、
`002450` 等）在**全部 147 期**里都不存在，而 1995/2005/2010 期的代码 **100%** 在 2025 期
仍存在。故任何用到财务数据的回测都**不得声称已排除幸存者偏差**（README 的已知局限有记）。
"""

from __future__ import annotations

import datetime as dt
import struct
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

from .errors import MarketDataError

#: 头部大小（字节）。
HEADER_SIZE = 20

#: 索引项布局：6 字节代码 + 1 字节填充 + 4 字节数据块绝对偏移。
_INDEX = struct.Struct("<6s1c1L")

#: 本项目需要的字段：``名称 → float 索引``。
#:
#: 只取这几列，不把 584 列全物化——实测全物化要读 0.71 GB，而只取这些列约 17 MB。
#: 单位与口径见各自的 ``_YTD`` / ``_QUARTER`` / 时点之分，**不要**按名字猜。
FIELDS = {
    "eps_ytd": 0,  # 基本每股收益（元，累计）
    "bvps": 3,  # 每股净资产（元，时点）
    "revenue_ytd": 73,  # 其中：营业收入（元，累计）
    "net_profit_ytd": 95,  # 归属于母公司所有者的净利润（元，累计）
    "revenue_quarter": 229,  # 营业收入（元，单季）
    "net_profit_quarter": 231,  # 归属于母公司所有者的净利润（元，单季）
    # 通达信 `FINVALUE(n)` = 本表的**索引 n − 1**（在真实文件上核对过：
    # FINVALUE(238)=总股本 ↔ 索引 237、FINVALUE(276)=利润TTM ↔ 索引 275、
    # FINVALUE(184)=增历史 ↔ 索引 183）。下面三个就是「估值」那几个公式真正用的字段——
    # 它们是**现成的整列**，不必自己拼或年化（见 mbt.data.valuation 的说明）。
    "total_shares": 237,  # 总股本（股，时点）——FINVALUE(238)
    "profit_ttm": 275,  # 归母净利润 TTM（元）——FINVALUE(276)
    "growth_ytd": 183,  # 归母净利润累计同比（%）——FINVALUE(184)
    "equity": 270,  # 归属母公司股东权益（元，时点）——FINVALUE(271)
    "announcement_date": 313,  # 财报公告日期（YYMMDD / YYYYMMDD 整数，存在 float32 里）
}

#: 公告日可用的最早年份。在此之前字段是「等于报告期」的占位符（实测 1995 年报 670/673 条如此）。
MIN_ANNOUNCEMENT_YEAR = 2005


@dataclass(frozen=True)
class FinancialRecord:
    """一个标的**一期**财报。

    属性:
        symbol: 标的符号（形如 ``sh600000``），由报告期文件的 6 位代码经数据源名单反查得到。
        report_period: 报告期（会计区间终点），**不是**公告日。
        announcement_date: 公告日；判不出来时为 ``None``。
        values: 字段名 → 数值，键取自 :data:`FIELDS`（不含 ``announcement_date``）。
    """

    symbol: str
    report_period: dt.date
    announcement_date: dt.date | None
    values: dict[str, float]

    @property
    def usable(self) -> bool:
        """该期的公告日是否可信（见模块文档的「2005 年之前不可用」）。"""
        return (
            self.announcement_date is not None
            and self.announcement_date.year >= MIN_ANNOUNCEMENT_YEAR
            and self.announcement_date > self.report_period
        )


class CwDataSource:
    """`vipdoc/cw` 的按期财务数据源。

    参数:
        root: ``vipdoc/cw`` 目录（**不是** ``vipdoc`` 本身）。
    """

    def __init__(self, root):
        self._root = Path(root)

    @property
    def periods(self) -> tuple[dt.date, ...]:
        """目录里全部**非空**报告期，升序。"""
        return tuple(sorted({period for period, _ in _load(self._root)}))

    def records(self, symbol: str) -> tuple[FinancialRecord, ...]:
        """标的的**全部报告期**记录，按报告期升序（含尚未公告的）。

        它给的是「这个标的历年都报了什么」，**不保证时点正确**——按评估日取数请用
        :meth:`as_of`。两个入口分开，是为了让「我知道自己在取哪一期」变得显式。
        """
        code = _core_code(symbol)
        out = []
        for _, table in _load(self._root):
            row = table.get(code)
            if row is not None:
                out.append(row)
        return tuple(sorted(out, key=lambda record: record.report_period))

    def as_of(self, symbol: str, on: dt.date | dt.datetime | str) -> FinancialRecord | None:
        """评估日**已公告**的最近一期；没有则 ``None``。

        这是本模块的时点纪律所在（ADR-0006）：按**公告日**过滤，而不是按报告期。拿报告期
        对齐等于提前数月知道尚未公布的财报，而那种错误不会报错，只会让结论偏乐观。
        """
        stamp = _as_date(on)
        usable = [
            record
            for record in self.records(symbol)
            if record.usable and record.announcement_date <= stamp
        ]
        if not usable:
            return None
        return max(usable, key=lambda record: (record.announcement_date, record.report_period))


def load_financials(root) -> dict[str, tuple[FinancialRecord, ...]]:
    """一次读完全部报告期，返回 ``标的 → 记录`` 的映射。

    实测读完全部 147 个文件约 0.7 秒、只要 7 列约 17 MB，故「一次装进内存」是划算的。

    标的符号由**目录里 6 位代码**构造：`gpcw` 只有裸代码，没有市场前缀，而按前缀猜第二遍
    会与 `board_of` / `instrument_type` 里已有的两份前缀知识漂移（见 :func:`_symbol_of`）。
    """
    table = _load(Path(root))
    grouped: dict[str, list[FinancialRecord]] = {}
    for _, records in table:
        for record in records.values():
            grouped.setdefault(record.symbol, []).append(record)

    return {
        symbol: tuple(sorted(items, key=lambda record: record.report_period))
        for symbol, items in grouped.items()
    }


def non_loss_mask(
    financials: dict[str, tuple[FinancialRecord, ...]],
    index: pd.Index,
    columns: pd.Index,
) -> pd.DataFrame:
    """「非亏损」过滤的 boolean 标的宽表。

    口径：**累计归母净利润 > 0**（``net_profit_ytd``）。理由：这就是判断「是否亏损」的市场
    惯例口径，也正是 ST 认定所依据的那个数。用**单季**会把「前三季赚、四季度亏」判成亏损
    （反之亦错），故单季字段存在但不用于此。

    某标的在某评估日**没有可用财报**（未公告、或公告日在 2005 年前而不可信）→ 该格为
    ``False``（**排除**）。这是**入场过滤**，排除是安全方向——不报错，因为报错会让 2005 年
    前的任何回测直接崩。理由可经 :func:`mbt.universe.why_not_in_universe` 查（它会给出
    「未通过附加过滤（如『非亏损』）」）。

    参数:
        financials: :func:`load_financials` 的产物。
        index: 交易日索引（宽表的行）。
        columns: 标的（宽表的列）。
    """
    rows = []
    for symbol in columns:
        records = financials.get(symbol, ())
        if not records:
            # 没有财务数据 → 整列假，且只记一次原因。
            rows.append(pd.Series(False, index=index, name=symbol))
            continue
        rows.append(_column(records, index, symbol))

    if not rows:
        return pd.DataFrame(False, index=index, columns=columns)
    return pd.concat(rows, axis=1).astype(bool)


def _column(records, index: pd.Index, symbol: str) -> pd.Series:
    """一个标的在各评估日是否非亏损。

    **按公告日分组、逐段填充**，而不是对每个 (标的, 日期) 各扫一遍历史记录：后者是
    O(标的 × 天数 × 期数)，全市场（约 5,500 标的 × 2,500 天 × 147 期）要跑几分钟；这样是
    O(标的 × 期数 × log 天数)。
    """
    usable = sorted((r for r in records if r.usable), key=lambda record: record.announcement_date)
    result = pd.Series(False, index=index, name=symbol)
    if not usable:
        return result

    # 每期财报的生效区间：自其公告日起、到下一期公告日之前。
    bounds = [pd.Timestamp(record.announcement_date) for record in usable] + [None]
    for position, record in enumerate(usable):
        if record.values.get("net_profit_ytd", 0.0) <= 0:
            continue  # 亏损的段留假，无需写
        start = bounds[position]
        stop = bounds[position + 1]
        window = index >= start
        if stop is not None:
            window = window & (index < stop)
        result[window] = True
    return result


def _as_of(records, on: dt.date):
    """按公告日取该日已公告的最近一期。

    与 :meth:`CwDataSource.as_of` **共用同一套判据**（都走 :attr:`FinancialRecord.usable` 与
    「公告日 ≤ 该日」），故 ADR-0006 的纪律只有一处实现。
    """
    usable = [r for r in records if r.usable and r.announcement_date <= on]
    if not usable:
        return None
    return max(usable, key=lambda record: (record.announcement_date, record.report_period))


def _as_date(value) -> dt.date:
    if isinstance(value, dt.datetime):
        return value.date()
    if isinstance(value, dt.date):
        return value
    return pd.Timestamp(value).date()


# --- 解析 ---------------------------------------------------------------------


def _core_code(symbol: str) -> str:
    """从符号里取 6 位裸代码：``sh600000`` → ``600000``。

    ``gpcw`` 只存裸代码，故 join 时也要裸代码。**不**在这里校验前缀与代码是否匹配
    （``sh000001`` 是上证指数而 ``sz000001`` 是平安银行）——那是 :func:`_symbol_of` 的事。
    """
    return symbol[2:] if len(symbol) > 6 else symbol


def _symbol_of(code: str) -> str:
    """由 6 位裸代码给出符号。

    ``gpcw`` 没有市场前缀，而**代码首位本身决定市场**：``6`` 开头沪市、``0``/``3`` 开头深市、
    ``9`` 开头北交所（本机实测北交所用 ``920xxx``）。故这里由首位推断前缀，而**不是**再维护
    一份前缀表——那份知识在 `board_of` / `instrument_type` 里已有两份且互相校验着，加第三份
    就是漂移的种子。

    .. note::

        代码与交易所的对应关系在极少数历史区间可能例外；但本模块只用于**按标的 join 财务
        数据**，join 不上的结果是「该标的没有财务数据」（过滤时排除，安全方向），不会静默
        取到别人的数字。
    """
    if code.startswith("6"):
        return f"sh{code}"
    if code.startswith("9"):
        return f"bj{code}"
    return f"sz{code}"


@lru_cache(maxsize=4)
def _load_cached(root: Path, fingerprint: tuple):
    """解析目录下全部 ``gpcw*.dat``（结果缓存）。``fingerprint`` 只用来让缓存**随文件失效**。"""
    return _parse_all(root)


def _load(root: Path):
    """按**目录内容指纹**缓存的入口。

    缓存键必须包含各文件的 ``(mtime_ns, size)``：通达信客户端会**每期重写**这些文件
    （实测文件 mtime 都是近期），只按目录名缓存会让长驻进程一直用旧表。这与
    :mod:`mbt.data.gbbq` 的做法一致。
    """
    paths = sorted(Path(root).glob("gpcw*.dat"))
    fingerprint = tuple((path.name, path.stat().st_mtime_ns, path.stat().st_size) for path in paths)
    return _load_cached(Path(root), fingerprint)


def _parse_all(root: Path):
    paths = sorted(Path(root).glob("gpcw*.dat"))
    if not paths:
        raise MarketDataError(f"{root} 下没有 gpcw*.dat——请确认这是通达信的 vipdoc/cw 目录")

    out = []
    for path in paths:
        raw = path.read_bytes()
        period, count, index_size, block_size = _read_header(raw, path)
        if count == 0:
            continue

        columns = block_size // 4
        # 字段名 → 在 `picked` 里的**位置**（`picked` 已是挑出来的那几列，故用位置索引；
        # 用原始 float 索引的话第 313 个会越界——那是本模块第一版踩过的坑）。
        positions = {name: position for position, name in enumerate(FIELDS)}
        wanted = list(FIELDS.values())

        records: dict[str, FinancialRecord] = {}
        for i in range(count):
            code_bytes, _, offset = _INDEX.unpack_from(raw, HEADER_SIZE + i * index_size)
            code = code_bytes.decode("ascii").strip("\x00")
            if not code:
                continue

            block = np.frombuffer(raw, dtype="<f4", count=columns, offset=offset)
            picked = block[wanted]
            record = FinancialRecord(
                symbol=_symbol_of(code),
                report_period=_to_date(period),
                announcement_date=_decode_announcement(
                    float(picked[positions["announcement_date"]])
                ),
                values={
                    name: float(picked[position])
                    for name, position in positions.items()
                    if name != "announcement_date"
                },
            )
            # 同一文件里存在**字节完全相同**的重复代码（实测 300750 / 301192 / 301321 各两份），
            # 去重，免得同一期被算两次。
            records.setdefault(code, record)

        out.append((_to_date(period), records))

    return tuple(out)


def _read_header(raw: bytes, path: Path):
    """按**显式偏移**读头部并校验帧恒等式（见模块文档对 struct 串的告警）。"""
    version = struct.unpack_from("<H", raw, 0)[0]
    if version != 1:
        raise MarketDataError(f"{path.name} 的格式版本是 {version}，本模块只认 1")

    period = struct.unpack_from("<I", raw, 2)[0]
    count = struct.unpack_from("<H", raw, 6)[0]
    index_size = struct.unpack_from("<H", raw, 10)[0]
    block_size = struct.unpack_from("<I", raw, 12)[0]

    if count:
        expected = HEADER_SIZE + count * index_size + count * block_size
        if len(raw) != expected:
            raise MarketDataError(
                f"{path.name} 的帧与头部自述不符：文件 {len(raw)} 字节，"
                f"而头部给出 {count} 条 ×（索引 {index_size} + 块 {block_size}）"
                f"+ 头 {HEADER_SIZE} = {expected} 字节。格式假设已失效。"
            )
    elif len(raw) != HEADER_SIZE:
        raise MarketDataError(f"{path.name} 自称 0 条，却有 {len(raw)} 字节（应为 20）")

    return period, count, index_size, block_size


def _decode_announcement(value: float) -> dt.date | None:
    """把 ``财报公告日期`` 的 float32 解成日期。

    **只认 6 位 ``YYMMDD``，世纪按 70 分界。** 这看起来像是少支持了一种编码，实则不然：

    - 实测全部 **303,491** 条非零公告日里，**8 位的一条都没有**（6 位 259,954 条，其余 43,537
      条是残缺值，一律返回 ``None``——把它们当日期会得到看似正常实则错误的年份）。
    - 更根本的是**float32 装不下 8 位整数**：``float32(20260331)`` 是 ``20260332``、
      ``float32(20260417)`` 是 ``20260416``。即便通达信真写 ``YYYYMMDD``，读到这里也已经被
      舍入改掉了。故「支持 8 位」是一条**做不到**的承诺，写在那里只会让人以为它被覆盖过。

    世纪分界取 70：``69`` → 2069、``70`` → 1970。财务数据始于 1988，故这个分界在本项目
    的数据范围内没有歧义。
    """
    number = int(round(value))
    if number <= 0:
        return None
    text = str(number)
    if len(text) != 6:
        return None
    try:
        yy = int(text[:2])
        return dt.date(2000 + yy if yy < 70 else 1900 + yy, int(text[2:4]), int(text[4:]))
    except ValueError:
        return None


def _to_date(period: int) -> dt.date:
    text = str(period)
    return dt.date(int(text[:4]), int(text[4:6]), int(text[6:]))
