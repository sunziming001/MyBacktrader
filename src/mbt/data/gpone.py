"""`T0002/hq_cache/gpSHone.dat` / `gpSZone.dat`：``GPONEDAT(ID)`` 的取值来源（票据 #50）。

公式 ``GPONEDAT(ID)`` 读的是**每市场一个**的快照文件——文件名里的 ``one`` 正是
「单条数据、非序列」的意思（`TCalc.dll` 的公式帮助原文把 ``GPONEDAT`` 归为「股票的**单个
数据（非序列）**」）。这与 `mbt.data.fundamental` 读的 ``gpcw*.dat``（按期、序列）是两回事。

## 文件布局（逐字节复核）

```
每市场一份：vipdoc 之外，在 T0002/hq_cache/
    gpshone.dat   沪市（代码 600000..689009）
    gpszone.dat   深市（代码 1..302132，即 000001..302132）
```

===========  ======  ==================================================
偏移         类型    含义
===========  ======  ==================================================
0            u32    代码，**6 位裸代码的整数**（``sh600000`` → 600000，``sz000001`` → 1）
4            u16    **字段号**，取值 1..47（见 :data:`FIELDS`）
6            f32    值
===========  ======  ==================================================

记录等长 10 字节，长度恒为 10 的整数倍（实测 ``gpshone.dat`` 823,710 = 10×82,371、
``gpszone.dat`` 995,750 = 10×99,575）。文件是**字段主序**：先按字段号分段、段内按代码升序。

## 字段号 → 字段名 的来源

通达信自己的公式帮助原文（本机 ``D:\\tmp\\tdx_g.txt`` 抄件，标题行即
``GPONEDAT(ID),ID为数据编号``），逐条抄进 :data:`FIELDS`。**这不是猜的**：该表覆盖 1..47，
而真实文件里出现的字段号恰好就是 1..47（沪市一个不缺，深市缺 40、41）。

## 为什么可以断定「就是它」（四条独立对账）

1. **字段号集合**：``gpshone.dat`` 的字段号恰好是 1..47，与帮助原文一一对应。
2. **发行价 / 总发行数量**（字段 1、2）对上**公开 IPO 数据**：
   贵州茅台 31.39 元 / 7,150 万股、浦发银行 10.00 / 40,000、工商银行 3.12 / 1,495,000、
   招商银行 7.30 / 150,000、中国银行 3.08 / 649,350.625、中信证券 4.50 / 40,000。
3. **最新总股本**（字段 33，万股）对 `gpcw` 的 ``FINVALUE(238)``（股 ÷ 1e4）——**另一族文件、
   另一套解码**：沪市 2,222/2,311、深市 2,790/2,897 只落在 1% 以内。
4. **市盈率与每股收益自洽**：字段 23（一致预期T年PE）× 字段 5（一致预期T年EPS）≈ 当时股价
   （浦发 9.45 对 9.10、茅台 1,348.4 对 1,258、平安 11.04 对 11.70）。

## 不要与 `vipdoc/cw/gp*.dat` 搞混

``vipdoc/cw/gp{sh,sz,bj}*.dat``（13 字节/条、每标的一份，见 :mod:`mbt.data.gp`）是**另一族**文件，
其 id 编号**不服从**本模块这张表——实测 ``vipdoc/cw`` 那个族里 ``id=4`` 的 f1 精确等于当日
收盘价，而本表把 4 指向「一致预期T年度」。票据 #50 早期把两者混为一谈，已订正。

## 时点纪律：这族数据**做不到**（ADR-0006 的登记缺口）

文件是**快照**：每个 ``(代码, 字段)`` 只有**一条**记录，即「最新值」。文件里**没有日期字段**，
故「评估日当时的一致预期」**不在文件里**——机构上一次给出的预测已被下一次覆盖。

后果必须写明白：:meth:`GponeDataSource.value` 给出的是「**现在这一版**」，**不得**当作历史
快照用于回测。它可以用在「今天的池子/今天的读数」这类**非回溯**场景（例如盘后选股），
但任何声称时点正确的回测都不能用它。文件 mtime 是客户端**重写这个文件**的时刻，不是这些
数值各自的发布日，故不能拿它当数据日期。
"""

from __future__ import annotations

import datetime as dt
import struct
from functools import lru_cache
from pathlib import Path

from .errors import MarketDataError

#: 单条记录的字节数。三个字段的偏移都由此推出。
RECORD_SIZE = 10

#: 记录布局：
#: ``u32`` 裸代码、``u16`` 字段号、``f32`` 值。
_RECORD = struct.Struct("<IHf")

#: 字段号 → 字段名，逐条抄自通达信公式帮助原文（``GPONEDAT(ID),ID为数据编号``）。
#:
#: 单位随字段而定，见名字里的括注。**空值显示为 0**（原文：「所有的空值数据显示为0」），
#: 故 0 既可能是真 0，也可能是「没有这项数据」——调用方若关心这个区别，只能自己按字段判断
#: （例如字段 4 是年份，出现 0 就一定不是真年份）。
FIELDS: dict[int, str] = {
    1: "发行价(元)",
    2: "总发行数量(万股)",
    3: "一致预期目标价(元)",
    4: "一致预期T年度",
    5: "一致预期T年每股收益",
    6: "一致预期T+1年每股收益",
    7: "一致预期T+2年每股收益",
    8: "一致预期T年净利润(万元)",
    9: "一致预期T+1年净利润(万元)",
    10: "一致预期T+2年净利润(万元)",
    11: "一致预期T年营业收入(万元)",
    12: "一致预期T+1年营业收入(万元)",
    13: "一致预期T+2年营业收入(万元)",
    14: "一致预期T年营业利润(万元)",
    15: "一致预期T+1年营业利润(万元)",
    16: "一致预期T+2年营业利润(万元)",
    17: "一致预期T年每股净资产(元)",
    18: "一致预期T+1年每股净资产(元)",
    19: "一致预期T+2年每股净资产(元)",
    20: "一致预期T年净资产收益率(%)",
    21: "一致预期T+1年净资产收益率(%)",
    22: "一致预期T+2年净资产收益率(%)",
    23: "一致预期T年PE",
    24: "一致预期T+1年PE",
    25: "一致预期T+2年PE",
    26: "最新解禁日(YYMMDD格式)",
    27: "最新解禁数量(万股)",
    28: "下一报告期的预约披露时间",
    29: "最新持股机构家数",
    30: "最新机构持股总量(万股)",
    31: "最新持股基金家数",
    32: "最新基金持股量(万股)",
    33: "最新总股本(万股)",
    34: "最新实际流通A股(万股)",
    35: "最新业绩预告 报告期(YYMMDD格式)",
    36: "最新业绩预告 本期归母净利润下限(万元)",
    37: "最新业绩预告 本期归母净利润上限(万元)",
    38: "最新业绩预告 本期归母净利润预计同比增减幅下限%",
    39: "最新业绩预告 本期归母净利润预计同比增减幅上限%",
    40: "最新业绩快报 报告期",
    41: "最新业绩快报 归母净利润(万元)",
    42: "分红募资 派现总额(万元)",
    43: "分红募资 募资总额(万元)",
    44: "最新业绩预告 本期扣非净利润下限(万元)",
    45: "最新业绩预告 本期扣非净利润上限(万元)",
    46: "最新业绩预告 本期扣非净利润预计同比增减幅下限%",
    47: "最新业绩预告 本期扣非净利润预计同比增减幅上限%",
}

#: 字段名 → 字段号（:data:`FIELDS` 的反查表）。
FIELD_BY_NAME: dict[str, int] = {name: field for field, name in FIELDS.items()}

#: 市场前缀 → 文件名。**本机没有北交所那一份**（公式文档说该函数适用沪深京）。
_FILE_BY_MARKET = {"sh": "gpshone.dat", "sz": "gpszone.dat"}


class GponeDataSource:
    """`T0002/hq_cache` 下 gp*one.dat 的取值来源。

    参数:
        root: ``T0002/hq_cache`` 目录（**不是** ``vipdoc``，这两族文件不在一个地方）。
    """

    def __init__(self, root):
        self._root = Path(root)

    def covers(self, symbol: str) -> bool:
        """本数据源**有没有这个市场**的那份文件——只看市场，不看文件在不在。

        用途是把两种「读不到」分开，它们对用户的含义正相反：北交所（本机没有
        ``gpbjone.dat``）是**已知的覆盖缺口**，安静退回历史即可；而沪深那份文件不在，
        几乎总是 ``root`` **指错了目录**，必须喊出来。

        :meth:`path_for` 两种都抛错（对调用方来说都是「取不到」），故要分辨只能问这里。
        """
        return symbol[:2] in _FILE_BY_MARKET

    def path_for(self, symbol: str) -> Path:
        """符号对应的文件。北交所、或前缀不认识时抛 :class:`MarketDataError`。"""
        market = symbol[:2]
        name = _FILE_BY_MARKET.get(market)
        if name is None:
            raise MarketDataError(
                f"{symbol}: 本数据源只覆盖 {sorted(_FILE_BY_MARKET)}（文件为 "
                f"{sorted(_FILE_BY_MARKET.values())}）。公式文档说该函数适用沪深京，"
                f"但本机没有北交所那一份。"
            )
        path = self._root / name
        if not path.is_file():
            raise MarketDataError(f"{path} 不存在——请确认这是通达信的 T0002/hq_cache 目录")
        return path

    def snapshot(self, symbol: str) -> dict[int, float]:
        """该标的**全部**字段号 → 值的快照。没有的字段号不出现（不补 0）。"""
        return dict(_load(self.path_for(symbol)).get(_core_code(symbol), {}))

    def value(self, symbol: str, field: int) -> float | None:
        """某字段的**最新值**；没有则 ``None``。

        「最新」是文件自己的语义（每个字段只存一条），**不是**「评估日当时的值」——
        见模块文档的时点缺口一节。
        """
        return self.snapshot(symbol).get(field)

    def named(self, symbol: str) -> dict[str, float]:
        """字段名 → 值的快照（名字取自 :data:`FIELDS`）。便于按语义取数。"""
        return {
            FIELDS[field]: value
            for field, value in self.snapshot(symbol).items()
            if field in FIELDS
        }

    def fields_present(self, symbol: str) -> set[int]:
        """该符号**所在文件**里出现过的全部字段号——不限于这只标的。

        用来校验格式假设（「字段号是否就是帮助原文那 1..47」）。单只标的只会命中它有数据的
        那几个字段，故这类断言必须打在整个文件上。
        """
        return {field for row in _load(self.path_for(symbol)).values() for field in row}

    def updated_on(self, symbol: str) -> dt.date:
        """该符号所在文件**最后写入那天**——「我们什么时候拿到这份一致预期」的答案。

        这是前瞻可用性的判据（见 :mod:`mbt.data.forward`）：评估日早于这一天时，手上这份
        内容可能已经不是那天的内容，故那时不得使用。

        .. warning::

            它**不是**「机构发布这些预测的日子」，而是「通达信客户端写下这个文件的时刻」。
            对防前视而言前者才是要问的（我们何时**能知道**），但本地只有后者——故同一天下载
            的文件里可能混着几周前发布的预测。那是数据质量的账，不是时点的账。
        """
        return dt.date.fromtimestamp(self.path_for(symbol).stat().st_mtime)


def _load(path: Path) -> dict[int, dict[int, float]]:
    """按**文件指纹**缓存地解析。通达信会重写这些文件，故缓存键含 ``(mtime_ns, size)``。"""
    stat = path.stat()
    return _load_cached(str(path), stat.st_mtime_ns, stat.st_size)


@lru_cache(maxsize=4)
def _load_cached(path_text: str, mtime_ns: int, size: int) -> dict[int, dict[int, float]]:
    """解析结果缓存。``mtime_ns`` / ``size`` 只用来让缓存**随文件失效**，不参与解析。"""
    path = Path(path_text)
    return _parse(path.read_bytes(), path)


def _parse(raw: bytes, path: Path) -> dict[int, dict[int, float]]:
    """解析成 ``代码 → (字段号 → 值)``；长度不合即报错（ADR-0005 的缺口纪律）。"""
    count, rest = divmod(len(raw), RECORD_SIZE)
    if rest:
        raise MarketDataError(
            f"{path.name} 有 {len(raw)} 字节，不是 {RECORD_SIZE} 的整数倍"
            f"（余 {rest}）——单条记录 {RECORD_SIZE} 字节的假设已失效。"
        )

    out: dict[int, dict[int, float]] = {}
    for i in range(count):
        code, field, value = _RECORD.unpack_from(raw, i * RECORD_SIZE)
        out.setdefault(code, {})[field] = value
    return out


def _core_code(symbol: str) -> int:
    """``sh600000`` → 600000、``sz000001`` → 1：文件里存的是 6 位裸代码的**整数**。"""
    body = symbol[2:]
    if not body.isdigit():
        raise MarketDataError(f"{symbol}: 代码部分不是数字")
    return int(body)
