"""`vipdoc/cw/gpXX*.dat`：13 字节定长记录的解码（票据 #50）。

fixture 是**真实文件的切片**（从 `gpsh600000.dat` 上按 id 块边界切下的 637 字节，
含 id 29 与 30 两块），故「13 字节」「偏移 1 是日期」「偏移 5/9 是两个 float32」这些
格式假设被一起锁住。

本模块**只解布局，不解语义**：id 到字段名的编号表在通达信服务端，本机没有（见 ADR-0008
的修订）。唯一的例外是 id 30 —— 它被**独立的第二个解析器**（`gbbq`）钉住了，故本文件里
最重要的一条就是拿它去对账。这正是 ADR-0008 说的「从猜测变为对账」。
"""

from __future__ import annotations

import datetime as dt
import struct
from pathlib import Path

import pytest

from mbt.data.errors import MarketDataError
from mbt.data.gbbq import GbbqDataSource
from mbt.data.gp import RECORD_SIZE, GpDataSource

FIXTURE = Path(__file__).parent / "fixtures" / "cw" / "gpsh600000.dat"
GBBQ_FIXTURE = Path(__file__).parent / "fixtures" / "gbbq" / "gbbq"

#: 本文件里被 `gbbq` 钉住的那个 id：实测它的日期**恰好**是除权除息日。
DIVIDEND_ID = 30

#: 由独立手段（按偏移直接读字节）解出的期望值——**不是照实现抄的**。
EXPECTED = {
    "records": 49,
    "ids": (29, 30),
    "first": {"id": 29, "date": dt.date(2005, 4, 28), "f1": 1.0, "f2": 0.0},
    # 这条的 f2 非零，故「第二格也是 float32」不是空断言。
    "one_with_two_values": {"id": 30, "date": dt.date(2002, 8, 22), "f1": 48200.0, "f2": 120500.0},
    "id_29": {"count": 23, "first": dt.date(2005, 4, 28), "last": dt.date(2016, 3, 11)},
    "id_30": {"count": 26, "first": dt.date(2000, 7, 6), "last": dt.date(2026, 7, 16)},
}


@pytest.fixture
def cw_root():
    """fixture 所在的目录（数据源收的是 ``vipdoc/cw`` 目录）。"""
    return FIXTURE.parent


def _raw_records() -> list[tuple[int, int, float, float]]:
    """按显式偏移手工切一遍，供各条测试使用（不经过被测实现）。"""
    raw = FIXTURE.read_bytes()
    out = []
    for index in range(len(raw) // RECORD_SIZE):
        offset = index * RECORD_SIZE
        out.append(
            (
                raw[offset],
                struct.unpack_from("<i", raw, offset + 1)[0],
                struct.unpack_from("<f", raw, offset + 5)[0],
                struct.unpack_from("<f", raw, offset + 9)[0],
            )
        )
    return out


def test_the_fixture_is_a_whole_number_of_thirteen_byte_records():
    """帧：文件长度是 13 的整数倍。这是本模块对文件格式的全部主张。"""
    raw = FIXTURE.read_bytes()

    assert RECORD_SIZE == 13
    assert len(raw) % RECORD_SIZE == 0
    assert len(raw) // RECORD_SIZE == EXPECTED["records"]


def test_the_layout_is_id_then_date_then_two_floats():
    """`[u8 id][u32 date][f32 f1][f32 f2]`——ADR-0008 记的就是这个布局。

    按**显式偏移**读头一条，并与独立解出的常数比。日期按 ``u32`` 且是 ``YYYYMMDD``
    十进制整数（不是 float32：8 位整数在 float32 里存不住，见财务数据那处的教训）。
    """
    field_id, date, f1, f2 = _raw_records()[0]
    expected = EXPECTED["first"]

    assert field_id == expected["id"]
    assert date == 20050428
    assert dt.date(date // 10000, date // 100 % 100, date % 100) == expected["date"]
    assert f1 == pytest.approx(expected["f1"], abs=1e-6)
    assert f2 == pytest.approx(expected["f2"], abs=1e-6)


def test_the_second_float_slot_is_really_a_second_value():
    """``f2`` 不是填充：真实文件里有非零值。

    若把它当成对齐用的填充字节，遇到非零的记录就会静默丢掉一半数据。
    """
    row = next(r for r in _raw_records() if r[1] == 20020822)
    expected = EXPECTED["one_with_two_values"]

    assert row[0] == expected["id"]
    assert row[2] == pytest.approx(expected["f1"], abs=1e-6)
    assert row[3] == pytest.approx(expected["f2"], abs=1e-6)


def test_the_fixture_holds_one_contiguous_block_per_id(cw_root):
    """一个 id 一段连续记录，这一段里不掺别的 id——实测 8,969 个文件皆如此。"""
    runs = []
    for field_id, *_ in _raw_records():
        if not runs or runs[-1] != field_id:
            runs.append(field_id)

    assert tuple(runs) == EXPECTED["ids"]


def test_records_of_one_id_come_back_in_date_order(cw_root):
    """同一 id 的记录按日期升序（实测真实文件皆如此），故「最后一条」就是最新一条。"""
    source = GpDataSource(cw_root)

    for field_id, expected in ((29, EXPECTED["id_29"]), (DIVIDEND_ID, EXPECTED["id_30"])):
        records = source.records("sh600000", field_id)

        assert len(records) == expected["count"], field_id
        assert records[0].date == expected["first"], field_id
        assert records[-1].date == expected["last"], field_id
        assert [r.date for r in records] == sorted(r.date for r in records)


def test_the_ids_of_one_symbol_are_reported_in_file_order(cw_root):
    """``ids()`` 给出该标的全部 id，按在文件里出现的先后。"""
    assert GpDataSource(cw_root).ids("sh600000") == EXPECTED["ids"]


def test_latest_returns_the_newest_record(cw_root):
    """``latest`` 取日期最大的一条——「单个数据（非序列）」就是靠它从序列里取。"""
    source = GpDataSource(cw_root)

    assert source.latest("sh600000", DIVIDEND_ID).date == EXPECTED["id_30"]["last"]
    assert source.latest("sh600000", 29).date == EXPECTED["id_29"]["last"]


def test_an_absent_id_is_an_empty_series_not_an_error(cw_root):
    """没有这个 id 是正常状态（缺口纪律：缺失跳过），不是错误。"""
    source = GpDataSource(cw_root)

    assert source.records("sh600000", 41) == ()
    assert source.latest("sh600000", 41) is None
    assert 41 not in source.ids("sh600000")


# --- 与独立 oracle 对账（本文件的核心） ---------------------------------------


def test_the_dividend_id_matches_the_gbbq_oracle(cw_root):
    """id 30 的日期**恰好**是除权除息日——两个独立解析器给出同一个答案。

    这条把 id 30 从「猜它是分红」变成「它被另一份数据源确认」。`gbbq` 与本模块读的是
    完全不同的两族文件、走两套解码，故它们的交集不是同义反复。
    """
    gp_dates = {r.date for r in GpDataSource(cw_root).records("sh600000", DIVIDEND_ID)}
    oracle = {e.ex_date for e in GbbqDataSource(GBBQ_FIXTURE).events("sh600000")}

    assert gp_dates, "对账两边都得有东西，否则这条测不出来"
    assert gp_dates <= oracle, "gp 里出现了 gbbq 不认的除权日"


def test_the_one_date_the_oracle_has_and_the_dividend_id_does_not_is_the_share_reform(cw_root):
    """两边**只差一天**：2006-05-12 的股改送股。

    股改对价送股**不除权**（见 `test_real_share_reform_event_does_not_create_a_fake_jump`），
    故它不进「除权除息日」这一栏。差额恰好是这一天，说明 id 30 的口径是「除权除息」而不是
    「一切股本变动」——这个区分是有内容的，若哪天差额变了，说明口径变了。
    """
    gp_dates = {r.date for r in GpDataSource(cw_root).records("sh600000", DIVIDEND_ID)}
    oracle = {e.ex_date for e in GbbqDataSource(GBBQ_FIXTURE).events("sh600000")}

    assert oracle - gp_dates == {dt.date(2006, 5, 12)}
    # 同一段历史里，另一个 id 反而**含**这一天——旁证两者口径确实不同。
    assert dt.date(2006, 5, 12) in {r.date for r in GpDataSource(cw_root).records("sh600000", 29)}


# --- 解析的稳健性 -------------------------------------------------------------


def test_a_missing_symbol_says_what_is_wrong(cw_root):
    """找不到文件 → **报错**，并说出期望的路径。

    「这个标的没有 gp 文件」与「路径指错了」是两件事，后者静默成空序列会一直错下去。
    """
    with pytest.raises(MarketDataError, match="gpsh600001"):
        GpDataSource(cw_root).ids("sh600001")


def test_a_file_whose_length_is_not_a_multiple_of_thirteen_is_rejected(tmp_path):
    """长度对不上 13 的整数倍 → **报错**，不静默丢掉尾巴。

    丢尾巴会让「格式已经变了」看起来像「文件刚好短了几个字节」，而读出的日期会错位。
    """
    broken = tmp_path / "gpsh600000.dat"
    broken.write_bytes(FIXTURE.read_bytes()[:-3])

    with pytest.raises(MarketDataError, match="13"):
        GpDataSource(tmp_path).ids("sh600000")
