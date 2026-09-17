"""``GPONEDAT(ID)`` 的取值来源：解析与对账（票据 #50）。

fixture 是**真实文件的切片**（``tests/fixtures/gpone/gpshone.dat``，从本机 ``gpshone.dat``
按代码抽出 14 只标的、510 条记录、5,100 字节）。字节全来自真实文件，只是把「字段主序」
重排成了「代码主序」——造一份合成的 ``gpone`` 等于用实现验证实现。

本模块最要紧的一组是**身份**：字段号 → 字段名的那张表来自通达信自己的公式帮助原文
（``GPONEDAT(ID),ID为数据编号``）。所以下面的期望值刻意分成两类：

- **外部锚点**（公开的 IPO 数据、``gpcw`` 独立算出的总股本）：用来证明「这张表就是对的」；
- **布局常数**（直接从字节读出）：用来锁住 10 字节的切法与三个偏移。
"""

import datetime as dt
import struct
from pathlib import Path

import pytest

from mbt.data import GPONE_FIELDS, GponeDataSource
from mbt.data.errors import MarketDataError

FIXTURE = Path(__file__).parent / "fixtures" / "gpone" / "gpshone.dat"

#: 公开的 IPO 数据（元 / 万股）——**外部来源**，不是从文件里抄的。
#:
#: 这是本模块最硬的一条证据：若字段 1 / 2 不是「发行价 / 总发行数量」，六只标的的六对数字
#: 不可能同时对上。
KNOWN_IPO = {
    "sh600519": (31.39, 7150.0),
    "sh600000": (10.00, 40000.0),
    "sh601398": (3.12, 1495000.0),
    "sh600036": (7.30, 150000.0),
    "sh601988": (3.08, 649350.625),
    "sh600030": (4.50, 40000.0),
}


@pytest.fixture
def gpone_root():
    """fixture 所在的目录（数据源收的是 ``T0002/hq_cache`` 那一层）。"""
    return FIXTURE.parent


def test_the_fixture_is_a_whole_number_of_ten_byte_records():
    """10 字节定长——这是本模块对文件格式的全部主张，先把它钉住。"""
    assert FIXTURE.stat().st_size == 5100
    assert FIXTURE.stat().st_size % 10 == 0
    assert FIXTURE.stat().st_size // 10 == 510


def test_the_layout_is_code_then_field_then_one_float():
    """``[u32 代码][u16 字段号][f32 值]``：用真实字节锁住三个偏移。

    茅台那一条（代码 600519、字段 1）的值是 ``31.389999389648438``——float32 存 31.39 的
    结果。用 ``==`` 而不是近似比较：这条要锁的正是**逐字节**的读法。
    """
    raw = FIXTURE.read_bytes()
    code, field, value = struct.unpack_from("<IHf", raw, 0)

    assert code == 600000
    assert field == 1
    assert value == 10.0

    # 找到茅台的那一条，确认偏移读法与代码、字段号的对应关系
    records = [struct.unpack_from("<IHf", raw, i * 10) for i in range(len(raw) // 10)]
    hit = [item for item in records if item[0] == 600519 and item[1] == 1]
    assert hit == [(600519, 1, 31.389999389648438)]


def test_every_code_has_at_most_one_record_per_field():
    """快照语义：``(代码, 字段号)`` 在文件里**只出现一次**。

    这条是「时点纪律做不到」这个结论的结构来源——不是「历史不好查」，而是**根本没有历史**：
    机构上一回给的一致预期已经被这一回覆盖掉了。
    """
    raw = FIXTURE.read_bytes()
    seen = set()
    for i in range(len(raw) // 10):
        code, field, _ = struct.unpack_from("<IHf", raw, i * 10)
        assert (code, field) not in seen, f"({code}, {field}) 出现了两次，快照假设不成立"
        seen.add((code, field))


def test_the_field_table_is_contiguous_one_to_forty_seven():
    """字段表覆盖 1..47 且无缺号——与公式帮助原文的范围一致。"""
    assert sorted(GPONE_FIELDS) == list(range(1, 48))
    assert GPONE_FIELDS[4] == "一致预期T年度"
    assert GPONE_FIELDS[5] == "一致预期T年每股收益"
    assert GPONE_FIELDS[8] == "一致预期T年净利润(万元)"
    assert GPONE_FIELDS[23] == "一致预期T年PE"


def test_the_issue_price_and_size_match_the_public_ipo_records(gpone_root):
    """字段 1 / 2 对上**公开的 IPO 数据**——六只标的、六对数字同时吻合。

    这是「字段号 → 字段名」那张表的独立验证：本模块的字段表若错位一格，这里立刻垮。
    """
    source = GponeDataSource(gpone_root)

    for symbol, (price, amount) in KNOWN_IPO.items():
        assert source.value(symbol, 1) == pytest.approx(price, rel=1e-6), symbol
        assert source.value(symbol, 2) == pytest.approx(amount, rel=1e-6), symbol


def test_the_total_shares_agree_with_the_financial_files(gpone_root):
    """字段 33（最新总股本，万股）对上 ``gpcw`` **另一族文件**解出的总股本。

    浦发银行 2026 半年报的 FINVALUE(238) 是 3,330,584 万股（股数 ÷ 1e4）——两条互不相干的
    解析路径给出同一个数。
    """
    source = GponeDataSource(gpone_root)

    assert source.value("sh600000", 33) == pytest.approx(3330583.75)
    assert source.value("sh600519", 33) == pytest.approx(125008.15625)


def test_the_pe_times_the_eps_lands_on_the_price(gpone_root):
    """一致预期T年PE（23）× 一致预期T年EPS（5）≈ 当时的股价。

    这是**文件内部**的自洽检查（不需要外部数据）：若 23 不是 PE 或 5 不是 EPS，乘积不会落在
    股价的量级上。茅台这一对给出 1,348.4 元，而它当时在 1,258 元附近。
    """
    source = GponeDataSource(gpone_root)

    pe = source.value("sh600519", 23)
    eps = source.value("sh600519", 5)

    assert pe == pytest.approx(20.014, rel=1e-3)
    assert eps == pytest.approx(67.372, rel=1e-3)
    assert pe * eps == pytest.approx(1348.4, rel=1e-2)


def test_a_date_field_decodes_to_a_real_date(gpone_root):
    """字段 26 是 ``YYMMDD`` 的日期——``270628`` 解作 2027-06-28。

    一个「值长得像日期」的字段是字段表正确性的旁证：帮助原文说 26 是「最新解禁日」。
    """
    source = GponeDataSource(gpone_root)
    number = source.value("sh600004", 26)

    assert number == 270628.0
    yy, mm, dd = int(number) // 10000, int(number) // 100 % 100, int(number) % 100
    assert dt.date(2000 + yy, mm, dd) == dt.date(2027, 6, 28)


def test_named_lookup_uses_the_documented_field_names(gpone_root):
    """按**字段名**取数——名字来自 :data:`GPONE_FIELDS`，与帮助原文逐字一致。"""
    source = GponeDataSource(gpone_root)
    named = source.named("sh600519")

    assert named["发行价(元)"] == pytest.approx(31.39, rel=1e-6)
    assert named["一致预期T年度"] == 2026.0
    assert named["一致预期T年PE"] == pytest.approx(20.014, rel=1e-3)


def test_a_stock_without_consensus_reports_zeros_not_absences(gpone_root):
    """没有一致预期的标的，字段存在但值为 0（帮助原文：「所有的空值数据显示为0」）。

    故 0 是「没有」还是「真 0」，本模块**不替调用方判断**——字段 4 是年份，出现 0 就一定是
    「没有」。这一条把这个区别记下来，免得下游把 0 当年份用。
    """
    source = GponeDataSource(gpone_root)

    assert source.value("sh600006", 4) == 0.0
    assert source.value("sh600006", 5) == 0.0
    assert source.value("sh600006", 23) == 0.0
    # 同一只票的股本字段却是有值的——「没有一致预期」不等于「文件里没这只票」
    assert source.value("sh600006", 33) == pytest.approx(200000.0)


def test_the_snapshot_returns_every_field_the_stock_has(gpone_root):
    """快照给出该标的的全部字段号；没有的字段号**不出现**（不补 0）。

    茅台缺的是解禁（26/27）、业绩预告/快报（35..41、44..47）、营收/营业利润（11..16）
    里的几个——本机实测如此。这条锁的是「缺的字段不补 0」，不是具体缺哪几个。
    """
    snapshot = GponeDataSource(gpone_root).snapshot("sh600519")

    assert set(snapshot) == set(range(1, 26)) | {28, 29, 30, 31, 32, 33, 34, 42, 43}
    assert 26 not in snapshot, "缺的字段不该被补成 0"
    assert 35 not in snapshot
    assert snapshot[4] == 2026.0


def test_a_symbol_that_is_not_in_the_file_is_empty_not_an_error(gpone_root):
    """文件里查不到这个标的 → 空快照，不报错（与「文件本身有问题」是两回事）。"""
    assert GponeDataSource(gpone_root).snapshot("sh699999") == {}
    assert GponeDataSource(gpone_root).value("sh699999", 1) is None


def test_the_beijing_market_has_no_file_and_says_so(gpone_root):
    """本机只有沪深两份；北交所那一份不存在，而公式文档说该函数适用沪深京。

    报错要指出**是哪一种缺**——「数据源不覆盖这个市场」与「路径指错了」必须能分辨。
    """
    with pytest.raises(MarketDataError, match="只覆盖"):
        GponeDataSource(gpone_root).snapshot("bj920819")


def test_a_missing_file_says_what_is_wrong(tmp_path):
    """目录里没有 gpshone.dat 时，报错要说清该给的是哪个目录。"""
    with pytest.raises(MarketDataError, match="hq_cache"):
        GponeDataSource(tmp_path).snapshot("sh600000")


def test_a_file_whose_length_is_not_a_multiple_of_ten_is_rejected(tmp_path, gpone_root):
    """长度不是 10 的整数倍 → 报错。截断必须报错，否则会解出一串看起来正常的错数字。"""
    (tmp_path / "gpshone.dat").write_bytes(FIXTURE.read_bytes()[:-3])

    with pytest.raises(MarketDataError, match="不是 10 的整数倍"):
        GponeDataSource(tmp_path).snapshot("sh600000")
