"""证券主表：`T0002/hq_cache/base.dbf`，给出**真实上市日**与总股本（票据 #25）。

用上通达信自带的主表，把股票池的次新股门槛从「本地行情不足 N 根」的**近似**换成
「自**上市日**起不足 N 个交易日」的**真实口径**（见 :func:`mbt.universe.build_universe`）。

## 事实（2026-09-12 在本机逐字节核实）

**dBase III 布局**，三项自洽性全部成立：

```
头部 32 字节：版本 0x03 | 记录数 u32 | 头长 u16 | 记录长 u16
字段描述符 32 字节一个，以 0x0D 结束（本机结束于偏移 1312，共 40 个字段）
记录：1 字节删除标记 + 各字段定长文本
文件末尾 0x1A（EOF）
本机：8,217 条 × 481 字节，头长 1,313，文件长 3,953,691
     → 1313 + 8217×481 = 3,953,690 = 文件长 − 1 ✓
```

**`SSDATE` 是真实上市日**，且有 300 个样本的硬证据：窗口内上市（首根 ``.day`` 晚于 2015-06）的
300 只股票，**首根与上市日相差 0 天，无一例外**——即 ``.day`` 从上市日开始，而 ``SSDATE`` 与它逐日
吻合。另有 `ZGB`（总股本，万股）等。

## 两处必须说清的局限

**一、`SSDATE` 有 5.2% 的股票为空或字符串 ``'0'``，且那 304 只 100% 是已退市的。**
实测：304 只中「末根仍到全市场最新交易日」的有 **0** 只（样例 `sh600001` 末根 2009-12-15、
`sh600002` 2006-04-06）。这与 `gpcw` 同源——**主表按当前证券主表重新生成**，退市标的从中消失。
故上市日未知时**回退到「本地行情根数」规则**，而那对它们本来就准（其本地首根就是上市日）。

注意判据是**字面 ``'0'``**，不是票据原先记的「1 位占位」——实测 129 条非空短值全是 ``'0'``，
且全是基金（如 `160605`）。

**二、`ZGB` 是当前快照，不是逐日历史。** 它带 `GXRQ`（更新日期）但只有一组值，故**不能回溯**
历史市值。:meth:`SecurityMasterDataSource.total_shares` 刻意**不收日期参数**，就是为了让这个局限
在签名上就能看出来——给一个 ``total_shares(symbol, on)`` 会诱导读者以为它是时点正确的（ADR-0006）。

## 刻意**不**用 `SC` 字段

`SC` 看起来是市场代码（0=深、1=沪、2=北），但实测与 ``.day`` 目录前缀有 **219 处不一致**
（174 个指数 + **16 只股票**，如 `sz000003` 的 SC=1）。故**以 ``.day`` 目录为准**，`SC` 不作判据。
"""

from __future__ import annotations

import datetime as dt
import struct
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from .errors import MarketDataError

#: dBase III 的头部大小。
HEADER_SIZE = 32

#: 字段描述符的字节数。
DESCRIPTOR_SIZE = 32

#: 字段描述符区的结束标记。
DESCRIPTOR_TERMINATOR = 0x0D

#: 文件末尾的 EOF 标记。
EOF_MARKER = 0x1A

#: 本项目需要的字段。
LISTING_DATE_FIELD = "SSDATE"
TOTAL_SHARES_FIELD = "ZGB"
SYMBOL_FIELD = "GPDM"


@dataclass(frozen=True)
class SecurityInfo:
    """一个标的的主表信息。

    属性:
        symbol: 标的符号（形如 ``sh600000``），由 ``.day`` 目录的符号反查得到。
        listing_date: 真实上市日；主表里为空或 ``'0'`` 时为 ``None``（**未知**）。
        total_shares: 总股本（单位：万股）；取不到时为 ``None``。
            .. warning:: **这是当前快照，不是逐日历史**——见模块文档。
    """

    symbol: str
    listing_date: dt.date | None
    total_shares: float | None


class SecurityMasterDataSource:
    """`base.dbf` 的读取器。

    参数:
        path: ``base.dbf`` 的路径。它**不在** ``vipdoc`` 之内（通常在
            ``<安装目录>/T0002/hq_cache/``），故与 :class:`~mbt.data.gbbq.GbbqDataSource` 一样
            直接收文件路径。
        symbols: 可选的**符号名单**，用来把主表里的 6 位裸代码对回 ``sh600000`` 这种形式。
            主表没有市场前缀，而 `SC` 字段不可靠（见模块文档），故**以 ``.day`` 目录为准**。
            不给时 ``symbol`` 字段会是裸代码。
    """

    def __init__(self, path, symbols=None):
        self._path = Path(path)
        self._symbols = None if symbols is None else _bare_to_symbol(symbols)

    @property
    def count(self) -> int:
        """主表的记录数。"""
        return _load(self._path)[0]

    def info(self, symbol: str) -> SecurityInfo | None:
        """取一个标的的主表信息；主表里没有它就返回 ``None``。

        传 ``sh600000`` 或裸代码 ``600000`` 都可以——映射在**实例层**做（缓存的表按裸代码存，
        因为缓存不该依赖某个实例的符号名单）。
        """
        item = _load(self._path)[1].get(_bare(symbol))
        if item is None:
            return None
        return SecurityInfo(
            symbol=self._to_symbol(_bare(symbol)),
            listing_date=item.listing_date,
            total_shares=item.total_shares,
        )

    def listing_dates(self) -> dict[str, dt.date]:
        """**已知**上市日的标的 → 上市日。未知的不出现在结果里。"""
        return {
            self._to_symbol(code): item.listing_date
            for code, item in _load(self._path)[1].items()
            if item.listing_date is not None
        }

    def total_shares(self, symbol: str) -> float | None:
        """总股本（万股）；主表里没有或取不到时为 ``None``。

        .. warning::

            **它是当前快照，不能回溯历史。** 刻意不收日期参数——收了会让人以为它是时点正确的
            （ADR-0006 关心的正是这种静默失真），而这张表只有一组值。
        """
        item = _load(self._path)[1].get(_bare(symbol))
        return None if item is None else item.total_shares

    def _to_symbol(self, code: str) -> str:
        if self._symbols is None:
            return code
        return self._symbols.get(code, code)


def load_listing_dates(path, symbols=None) -> dict[str, dt.date]:
    """便利函数：直接取「已知上市日」的映射。"""
    return SecurityMasterDataSource(path, symbols=symbols).listing_dates()


def _bare_to_symbol(symbols) -> dict[str, str]:
    """6 位裸代码 → 符号。``.day`` 目录是市场归属的**唯一**依据（见模块文档）。"""
    return {symbol[2:]: symbol for symbol in symbols}


def _bare(symbol: str) -> str:
    """``sh600000`` 与 ``600000`` 都归一成 ``600000``。"""
    return symbol[2:] if len(symbol) > 6 else symbol


def _decode_date(text: str) -> dt.date | None:
    """``SSDATE`` 形如 ``YYYYMMDD``；为空或字面 ``'0'`` 即**未知**。

    实测那 129 条非空短值全是 ``'0'``（全是基金）。**不**把短值当月日补零——那会造出一个
    看似正常实则错误的年份，而这类静默错答正是本项目反复防的。
    """
    if not text or text == "0" or len(text) != 8:
        return None
    try:
        return dt.date(int(text[:4]), int(text[4:6]), int(text[6:]))
    except ValueError:
        return None


def _decode_number(text: str) -> float | None:
    if not text:
        return None
    try:
        return float(text)
    except ValueError:
        return None


def _read_layout(raw: bytes, path: Path):
    """读出头部与字段描述符，并做三项自洽性校验。

    **三项都校验**的理由：它们在本机全部成立，一旦不成立说明格式变了——而静默按错误的布局
    解析会给出**看似正常的错数字**（本项目反复防的就是这一类）。
    """
    if len(raw) < HEADER_SIZE:
        raise MarketDataError(f"{path.name} 不足 {HEADER_SIZE} 字节，不是 dBase III 文件")

    version = raw[0]
    if version != 0x03:
        raise MarketDataError(
            f"{path.name} 的版本字节是 {version:#04x}，本模块只认 dBase III（0x03）"
        )

    count = struct.unpack_from("<I", raw, 4)[0]
    header_length = struct.unpack_from("<H", raw, 8)[0]
    record_length = struct.unpack_from("<H", raw, 10)[0]

    fields = []
    offset = HEADER_SIZE
    while offset < len(raw) and raw[offset] != DESCRIPTOR_TERMINATOR:
        name = raw[offset : offset + 11].split(b"\x00")[0].decode("ascii", "replace")
        width = raw[offset + 16]
        fields.append((name, width))
        offset += DESCRIPTOR_SIZE

    if offset >= len(raw) or raw[offset] != DESCRIPTOR_TERMINATOR:
        raise MarketDataError(f"{path.name} 的字段描述符区没有 0x0D 结束标记，格式不符")
    if offset + 1 != header_length:
        raise MarketDataError(
            f"{path.name} 的描述符区结束于 {offset + 1}，而头部自称头长 {header_length}"
        )

    widths = sum(width for _, width in fields)
    if widths != record_length - 1:
        raise MarketDataError(
            f"{path.name} 的字段宽度合计 {widths} ≠ 记录长 {record_length} − 1（删除标记）"
        )

    expected = header_length + count * record_length
    if len(raw) - 1 != expected:
        raise MarketDataError(
            f"{path.name} 的帧与头部自述不符：文件 {len(raw)} 字节，"
            f"而头部给出 {count} 条 × {record_length} + 头 {header_length} = {expected}"
            f"（末字节应为 EOF 标记）。格式假设已失效。"
        )
    if raw[-1] != EOF_MARKER:
        raise MarketDataError(f"{path.name} 的末字节不是 EOF 标记 {EOF_MARKER:#04x}")

    return count, header_length, record_length, fields


@lru_cache(maxsize=4)
def _load_cached(path: Path, fingerprint: tuple):
    return _parse(path)


def _load(path: Path):
    """按**文件指纹**缓存（客户端会重写这份文件——实测 mtime 是近期）。"""
    stat = Path(path).stat()
    return _load_cached(Path(path), (stat.st_mtime_ns, stat.st_size))


def _parse(path: Path):
    raw = path.read_bytes()
    count, header_length, record_length, fields = _read_layout(raw, path)

    positions = {}
    position = 0
    for name, width in fields:
        positions[name] = (position, width)
        position += width

    for needed in (SYMBOL_FIELD, LISTING_DATE_FIELD, TOTAL_SHARES_FIELD):
        if needed not in positions:
            raise MarketDataError(f"{path.name} 缺少字段 {needed!r}——本模块只认这份主表的布局")

    table: dict[str, SecurityInfo] = {}
    for index in range(count):
        start = header_length + index * record_length
        chunk = raw[start + 1 : start + record_length]  # 跳过删除标记
        row = {}
        for name, (offset, width) in positions.items():
            row[name] = chunk[offset : offset + width].decode("gbk", "replace").strip()

        code = row[SYMBOL_FIELD]
        if not code:
            continue
        table[code] = SecurityInfo(
            symbol=code,  # 由调用方在构造数据源时换成带前缀的符号
            listing_date=_decode_date(row[LISTING_DATE_FIELD]),
            total_shares=_decode_number(row[TOTAL_SHARES_FIELD]),
        )

    return count, table
