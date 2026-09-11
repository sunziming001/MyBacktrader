"""验证 ``gbbq`` 权息文件的解析与解密（票据 #3 切片 1–2）。

测试缝 S3：密文取自仓库内的**真实切片** ``tests/fixtures/gbbq/gbbq``，不依赖本机通达信
目录。该 fixture 是 sh600000 与 sz000001 两段真实记录的逐字节拼接——两段在真实文件里
并不相邻，但记录之间没有链接模式，故格式上仍是合法 gbbq（见 ``mbt.data.gbbq`` 模块文档）。
"""

import struct
from collections import Counter

import numpy as np
import pytest

from mbt.data import AdjustmentEvent, GbbqDataSource, GbbqError
from mbt.data.gbbq import ENCRYPTED_SIZE, HEADER_SIZE, RECORD_SIZE, _validate

SH = "sh600000"
SZ = "sz000001"

#: fixture 的记录总数（sh600000 88 条 + sz000001 80 条）。
TOTAL_RECORDS = 168


@pytest.fixture
def source(gbbq_file):
    return GbbqDataSource(gbbq_file)


def _float32_bits(value: float) -> bytes:
    """float32 的精确比特——对照必须精确，不能靠浮点近似。"""
    return struct.pack("<f", value)


def _read_oracle(path):
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.startswith("#") or line.startswith("symbol,"):
            continue
        symbol, date, category, f1, f2, f3, f4 = line.split(",")
        rows.append((symbol, int(date), int(category), float(f1), float(f2), float(f3), float(f4)))
    return sorted(rows)


# --- 文件结构 ---


def test_header_count_matches_the_record_count(gbbq_file):
    """``4 字节计数 + N × 29 字节``——这条格式假设是整块解析的地基。"""
    raw = gbbq_file.read_bytes()
    (count,) = struct.unpack_from("<I", raw, 0)

    assert count == TOTAL_RECORDS
    assert len(raw) == HEADER_SIZE + count * RECORD_SIZE


def test_encrypted_prefix_is_24_bytes_of_each_29(gbbq_file):
    """每条记录 24 字节密文 + 5 字节明文，且明文尾部不含计数头。"""
    assert ENCRYPTED_SIZE == 24
    assert HEADER_SIZE + TOTAL_RECORDS * RECORD_SIZE == gbbq_file.stat().st_size


def test_total_records_counts_every_category(source):
    assert source.total_records == TOTAL_RECORDS


def test_a_file_with_zero_records_is_valid(tmp_path):
    """计数为 0、只含 4 字节头的文件不是错误，只是空表。"""
    path = tmp_path / "empty_gbbq"
    path.write_bytes(struct.pack("<I", 0))

    assert GbbqDataSource(path).total_records == 0
    assert GbbqDataSource(path).records(SH) == ()


# --- 解密正确性（与独立 oracle 对账） ---


def test_decryption_matches_the_independent_oracle(source, gbbq_oracle):
    """逐字段比对 oracle（pytdx）——解密错了这里必崩。

    对照覆盖 fixture 里的**全部 168 条**记录，含 9 个不同类别，故不只验证了类别 1。
    浮点按 float32 比特比较，不做近似。
    """
    mine = sorted(
        (row.symbol, int(row.date.strftime("%Y%m%d")), row.category, row.f1, row.f2, row.f3, row.f4)
        for symbol in (SH, SZ)
        for row in source.records(symbol)
    )
    oracle = _read_oracle(gbbq_oracle)

    assert len(mine) == len(oracle) == TOTAL_RECORDS
    for got, want in zip(mine, oracle, strict=True):
        assert got[:3] == want[:3], f"标识字段不符：{got[:3]} != {want[:3]}"
        assert [_float32_bits(v) for v in got[3:]] == [
            _float32_bits(v) for v in want[3:]
        ], f"{want[0]} {want[1]} 类别 {want[2]} 的浮点字段不符：{got[3:]} != {want[3:]}"


def test_validate_rejects_a_record_that_breaks_the_structure():
    """解密结果的自证：结构不合法即报错。

    这是密钥表正确性的**不依赖 oracle** 的那一半保障——密钥错一位，解出的
    market/code/date 就会越界，而不是安静地给出看似合理的数字。
    """

    def record(market=1, code=b"600000", date=20260716, category=1):
        payload = struct.pack(
            "<B7sIBffff", market, code + b"\x00", date, category, 0.0, 0.0, 0.0, 0.0
        )
        assert len(payload) == RECORD_SIZE
        return np.frombuffer(payload, dtype=np.uint8).reshape(1, RECORD_SIZE)

    _validate(record())  # 合法记录不得误报

    with pytest.raises(GbbqError, match="结构约束"):
        _validate(record(code=b"6000XX"))  # 代码含非数字
    with pytest.raises(GbbqError, match="结构约束"):
        _validate(record(date=20261332))  # 13 月 32 日
    with pytest.raises(GbbqError, match="结构约束"):
        _validate(record(market=9))  # 不存在的市场编号
    with pytest.raises(GbbqError, match="结构约束"):
        _validate(record(category=0))  # 类别必须 >= 1


# --- 坏文件必须报错（缺口纪律） ---


def test_missing_file_is_rejected(tmp_path):
    with pytest.raises(GbbqError, match="不可读"):
        GbbqDataSource(tmp_path / "nope").records(SH)


def test_file_shorter_than_the_count_header_is_rejected(tmp_path):
    path = tmp_path / "short"
    path.write_bytes(b"\x01\x02")

    with pytest.raises(GbbqError, match="不足一个计数头"):
        GbbqDataSource(path).records(SH)


def test_length_not_matching_the_count_is_rejected(tmp_path):
    """计数头说 5 条、实际只有 2 条——格式不符，必须报错而非按实际条数静默读完。"""
    path = tmp_path / "truncated"
    path.write_bytes(struct.pack("<I", 5) + b"\x00" * (2 * RECORD_SIZE))

    with pytest.raises(GbbqError, match="与计数头不符"):
        GbbqDataSource(path).records(SH)


# --- 标的定位 ---


def test_market_prefix_selects_the_market(source):
    """同一串数字在不同市场是不同标的——市场编号必须参与定位。"""
    assert len(source.records(SH)) == 88
    assert len(source.records(SZ)) == 80
    assert source.records("sz600000") == ()
    assert source.records("sh000001") == ()


def test_unknown_symbol_yields_no_records(source):
    assert source.records("sh999999") == ()


@pytest.mark.parametrize("symbol", ["600000", "sh60000", "sh6000007", "xx600000", "sh60a000"])
def test_malformed_symbol_is_rejected(source, symbol):
    with pytest.raises(ValueError, match="市场前缀"):
        source.records(symbol)


# --- 事件抽取：只有除权除息类参与 ---


def test_records_keep_every_category(source):
    """原始记录不丢类别——将来的估值因子要用类别 5 的股本数据。"""
    categories = Counter(row.category for row in source.records(SH))

    assert categories == {1: 27, 2: 4, 3: 1, 5: 51, 6: 1, 8: 2, 9: 2}


def test_only_dividend_records_become_adjustment_events(source):
    """类别 2–15 的四个浮点数是股本数量，不是价格数据，绝不能进复权。"""
    records = source.records(SH)
    non_dividend = [row for row in records if row.category != 1]
    events = source.events(SH)

    assert non_dividend, "fixture 必须含非除权除息记录，否则这条测试无从判别"
    assert len(events) == 27
    assert all(isinstance(event, AdjustmentEvent) for event in events)
    # 事件日期与被排除的记录日期不重合，证明过滤真的按类别生效
    assert len(events) == sum(1 for row in records if row.category == 1)


def test_dividend_fields_map_to_the_event_semantics(source):
    """类别 1 的 f1..f4 = 每 10 股分红 / 配股价 / 每 10 股送转 / 每 10 股配股。"""
    by_date = {event.ex_date.strftime("%Y%m%d"): event for event in source.events(SH)}

    cash_and_bonus = by_date["20020822"]
    assert cash_and_bonus.cash_per_10 == pytest.approx(2.0)
    assert cash_and_bonus.bonus_per_10 == pytest.approx(5.0)
    assert cash_and_bonus.rights_per_10 == 0.0

    rights = {event.ex_date.strftime("%Y%m%d"): event for event in source.events(SZ)}["20001106"]
    assert rights.cash_per_10 == 0.0
    assert rights.rights_price == pytest.approx(8.0)
    assert rights.rights_per_10 == pytest.approx(3.0)


def test_events_are_sorted_by_ex_date(source):
    for symbol in (SH, SZ):
        dates = [event.ex_date for event in source.events(symbol)]
        assert dates == sorted(dates)


def test_records_are_sorted_by_date(source):
    for symbol in (SH, SZ):
        dates = [row.date for row in source.records(symbol)]
        assert dates == sorted(dates)
