"""`vipdoc/cw/gpXX*.dat`：每标的一份的 13 字节定长记录（票据 #50）。

这个文件族占了 `vipdoc/cw` 的 62%（本机 8,976 个文件、约 1.5 GB、1.257 亿条记录），
ADR-0008 一期**刻意没有逆向**，理由是「tag 语义未确认，且全网无开源解析器」。本模块做的是
它留的那一步：**把布局钉住，把语义留给对账**。

## 文件布局（已在真实文件上逐字节复核）

```
每标的一份：vipdoc/cw/gp<符号>.dat（gpsh600000.dat / gpsz000001.dat / gpbj920819.dat）
全部记录等长 13 字节；长度**全部**是 13 的整数倍（实测 8,976 个文件无一例外）
```

===========  ======  ==================================================
偏移         类型    含义
===========  ======  ==================================================
0            u8     **id**（字段码，实测出现 1..52 与 99）
1            u32    日期，``YYYYMMDD`` 十进制整数（**不是** float32）
5            f32    ``f1``——多数 id 的主值
9            f32    ``f2``——许多 id 恒为 0，但也确有非零值，故不是填充
===========  ======  ==================================================

记录**按 id 分段**：一个 id 一段连续记录，段内按日期升序；不同 id 的段首尾相接。
（实测全部 8,976 个文件共 245,470 段：同一个 id 出现在两段的情况 **0 例**；段内日期非升序
168 例，而它们**全部**由日期字段本身残缺引起——值形如 ``201408`` / ``201203``，即**缺了
「日」的 ``YYYYMM``**，会被 :func:`_to_date` 丢掉，故经本模块读出的序列是严格升序的。）

## 已知与未知，分得很清

**已知**：id 是**跨标的稳定的字段码**——同一个 id 在不同票上是同一个量。故这张编号表一旦
拿到，全市场通用。

**未知**：id → 字段名的编号表在通达信服务端，**本机没有**。`TCalc.dll` 里只有公式帮助原文
（``GPONEDAT(ID),ID为数据编号``，归为「股票的单个数据（非序列）」），没有任何编号清单；
`cw/*.txt` 是下载清单（文件名 + MD5 + 字节数）。故编号语义**不能靠猜**（ADR-0008 的
自证要求），只能对账。

**已对上账的一个**：``id = 30`` 是**除权除息日**。它与 `gbbq` 逐日对上——`sh600000`
的 26 个日期**全部**落在 gbbq 的除权除息事件里，两边**只差 2006-05-12 那一天**，而那天是
股改对价送股（不除权，见 `mbt.data.dilution`）。两个解析器读两族不同的文件、走两套解码，
故这个交集不是同义反复。

## 时点纪律在此**做不到**（ADR-0006 的登记缺口）

日频 id 只保留**最近 2000 条**——是一个**滚动窗口，旧记录被覆盖掉了**。证据不是「条数看着
像 2000」，而是**同一文件里多个 id 共用同一段起止日期**：

- `gpsh880411.dat` / `gpsh881111.dat` / `gpsh881416.dat` 各有 **14 个 id 全部正好 2000 条**，
  且这 14 个 id 的日期跨度**完全相同**（20180622~20260915）。条数是巧合，日期跨度一致不是。
- 全量 8,976 个文件里，**5,142 个（57.3%）**含至少一个正好 2000 条的 id。
- 抽样 44 个文件统计这些 id 的收尾日期，81 次落在 20260915、20 次落在 20260916——**都是最新
  交易日**，即窗口跟着最新数据往前滚。

后果必须写明白：对这些 id 而言，「评估日当时的值」**已经不在文件里**，故
:meth:`GpDataSource.records` 给出的是「现在这一版」，**不能**当作历史快照用。
这与 `gpcw` 的处境不同——那里的老报告期还在，只是公告日可能退化；这里是**整段历史被删**。
任何用到日频 id 的因子都**不能**声称时点正确，除非日频截断这件事另有解（目前没有）。

## 本模块刻意不做的事

不提供任何语义命名的取值口（如 ``pe()`` / ``dividend_dates()``）。编号语义未知时给出这样的
名字，等于把猜测写进接口，而调用方无从分辨哪个是真的。
"""

from __future__ import annotations

import datetime as dt
import struct
from dataclasses import dataclass
from pathlib import Path

from .errors import MarketDataError

#: 单条记录的字节数。三条字段的偏移都由此推出。
RECORD_SIZE = 13

#: 日期字段允许的年份范围。超出即视为**无日期**（见 :func:`_to_date`）。
MIN_YEAR = 1990
MAX_YEAR = 2100


@dataclass(frozen=True)
class GpRecord:
    """一个 id 下的一条记录。

    两个 float 槽**保留中性命名**（``f1`` / ``f2``，即 ADR-0008 写的 ``a`` / ``b``）：
    它们的含义随 id 而变，且编号语义尚未确认，故不冒充语义——与 :class:`mbt.data.gbbq.GbbqRecord`
    保留 ``f1..f4`` 是同一个理由。
    """

    date: dt.date
    f1: float
    f2: float


class GpDataSource:
    """`vipdoc/cw` 的 `gp*.dat` 数据源。

    参数:
        root: ``vipdoc/cw`` 目录（**不是** ``vipdoc`` 本身）。
    """

    def __init__(self, root):
        self._root = Path(root)

    def path_for(self, symbol: str) -> Path:
        """标的对应的文件路径：``sh600000`` → ``<root>/gpsh600000.dat``。"""
        return self._root / f"gp{symbol}.dat"

    def tables(self, symbol: str) -> dict[int, tuple[GpRecord, ...]]:
        """该标的**全部** id 的记录，一次读完。

        供批量调用方复用：每个文件是 300 KB 量级，逐个 id 反复调 :meth:`records` 会把同一份
        文件读上几十遍。
        """
        path = self.path_for(symbol)
        if not path.is_file():
            raise MarketDataError(f"未找到标的 {symbol} 的 gp 文件：{path}")
        return _parse(path.read_bytes(), path)

    def ids(self, symbol: str) -> tuple[int, ...]:
        """该标的的**全部** id，按在文件里出现的先后。"""
        return tuple(self.tables(symbol))

    def records(self, symbol: str, field_id: int) -> tuple[GpRecord, ...]:
        """某个 id 的全部记录，按日期升序。

        该 id 不在文件里是**正常状态**（缺口纪律：缺失跳过），返回空元组而不是报错。

        .. warning::

            日频 id 只有最近 2000 条（见模块文档的「时点纪律在此做不到」）。返回的是
            **现在这一版**，不是评估日当时的快照。
        """
        return self.tables(symbol).get(field_id, ())

    def latest(self, symbol: str, field_id: int) -> GpRecord | None:
        """某个 id 最新的一条记录；没有则 ``None``。

        通达信那个家族的公式是「单个数据（非序列）」，取的就是这一条。
        """
        records = self.records(symbol, field_id)
        if not records:
            return None
        return max(records, key=lambda record: record.date)


def _parse(raw: bytes, path: Path) -> dict[int, tuple[GpRecord, ...]]:
    """把整份文件切成 ``id → 记录``。

    **按 id 分组而不是假定「段」**：实测每段连续，但那是观察、不是格式契约，故实现不依赖
    它——哪天记录交错出现，这里仍能读对，而一个依赖连续性的实现会静默读错。
    """
    if len(raw) % RECORD_SIZE:
        raise MarketDataError(
            f"{path.name} 的长度 {len(raw)} 不是 {RECORD_SIZE} 的整数倍——"
            f"13 字节定长记录这个格式假设已失效。"
        )

    grouped: dict[int, list[GpRecord]] = {}
    for offset in range(0, len(raw), RECORD_SIZE):
        date = _to_date(struct.unpack_from("<I", raw, offset + 1)[0])
        if date is None:
            # 无日期的记录跳过：日期都读不出来的记录，其数值也无法归到任何时点。
            continue
        grouped.setdefault(raw[offset], []).append(
            GpRecord(
                date=date,
                f1=struct.unpack_from("<f", raw, offset + 5)[0],
                f2=struct.unpack_from("<f", raw, offset + 9)[0],
            )
        )
    return {field_id: tuple(records) for field_id, records in grouped.items()}


def _to_date(number: int) -> dt.date | None:
    """把 ``YYYYMMDD`` 十进制整数解成日期；解不出（0、残缺、不存在的日子）则为 ``None``。

    真实文件里确实有解不出的日期，但**极少**：全量 8,976 个文件、125,689,706 条记录里只有
    **695 条**（0.0006%）解不出，其中恰为 0 的 341 条，散布在 343 个文件里。故这里必须容错，
    不能抛错——抛错会让整份文件读不出来，而里头的记录几乎全是好的。
    """
    year, month, day = number // 10000, number // 100 % 100, number % 100
    if not (MIN_YEAR <= year <= MAX_YEAR):
        return None
    try:
        return dt.date(year, month, day)
    except ValueError:
        return None
