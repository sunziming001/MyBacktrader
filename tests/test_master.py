"""证券主表：真实上市日与总股本（票据 #25）。

fixture 是**真实 `base.dbf` 的字节切片**（4 条记录，3,238 字节），故 dBase III 的布局假设
（头部、字段描述符、记录、EOF）与字段索引被一起锁住——造一份合成 dbf 等于用实现验证实现。

四条记录覆盖三种情形：`600000`/`600519`（老股，窗口外上市）、`000001`（1991-04-03）、
`160605`（**上市日为字面 `'0'`**，锁「未知」这条路径）。
"""

from __future__ import annotations

import datetime as dt
import struct
from pathlib import Path

import pytest

from mbt.data.errors import MarketDataError
from mbt.data.master import (
    DESCRIPTOR_TERMINATOR,
    EOF_MARKER,
    HEADER_SIZE,
    SecurityMasterDataSource,
    load_listing_dates,
)

FIXTURE = Path(__file__).parent / "fixtures" / "master" / "base.dbf"


@pytest.fixture
def master():
    """带符号名单的数据源——`SC` 字段不可靠，故符号由 ``.day`` 目录反查（见模块文档）。"""
    return SecurityMasterDataSource(
        FIXTURE, symbols=["sh600000", "sh600519", "sz000001", "sz160605"]
    )


def test_the_fixture_satisfies_the_dbase_iii_frame_identity():
    """帧恒等式：``头长 + 记录数 × 记录长 + 1（EOF） == 文件长``。

    它把「布局假设」本身钉住——那是本模块对文件格式的全部主张，且在本机 3.95 MB 的原文件上成立。
    """
    raw = FIXTURE.read_bytes()
    count = struct.unpack_from("<I", raw, 4)[0]
    header_length = struct.unpack_from("<H", raw, 8)[0]
    record_length = struct.unpack_from("<H", raw, 10)[0]

    assert count == 4
    assert len(raw) == header_length + count * record_length + 1
    assert raw[-1] == EOF_MARKER


def test_the_header_and_descriptors_are_read_by_explicit_offsets():
    """头部按偏移读；字段描述符 32 字节一个、以 0x0D 结束。"""
    raw = FIXTURE.read_bytes()

    assert raw[0] == 0x03, "dBase III 的版本字节"
    assert struct.unpack_from("<H", raw, 8)[0] == 1313
    # 描述符区在 0x0D 处结束，且其长度与头部自称的头长一致
    offset = HEADER_SIZE
    while raw[offset] != DESCRIPTOR_TERMINATOR:
        offset += 32
    assert offset + 1 == 1313


# --- 上市日 -------------------------------------------------------------------


def test_real_listing_dates_match_independently_known_values(master):
    """上市日与**外部已知的真实日期**一致——不是照实现抄的。

    平安银行 1991-04-03、浦发银行 1999-11-10、贵州茅台 2001-08-27。
    """
    assert master.info("sz000001").listing_date == dt.date(1991, 4, 3)
    assert master.info("sh600000").listing_date == dt.date(1999, 11, 10)
    assert master.info("sh600519").listing_date == dt.date(2001, 8, 27)


def test_a_literal_zero_listing_date_is_unknown_not_a_date(master):
    """`SSDATE` 为字面 ``'0'`` 即**未知**，不当成某个日期。

    实测本机 129 条非空短值全是 ``'0'``（全是基金）。若按短值补零，会造出一个看似正常
    实则错误的年份——那正是本项目反复防的静默错答。
    """
    info = master.info("sz160605")

    assert info is not None, "这条记录在主表里，只是上市日不可用"
    assert info.listing_date is None


def test_listing_dates_excludes_the_unknown_ones(master):
    """``listing_dates()`` 只给**已知**的，未知的不出现（而不是给个假日期）。"""
    dates = master.listing_dates()

    assert set(dates) == {"sh600000", "sh600519", "sz000001"}
    assert "sz160605" not in dates


def test_symbols_without_a_market_prefix_are_translated_from_the_dot_day_listing(master):
    """主表只有 6 位裸代码，市场归属取自 ``.day`` 目录——**不用 `SC` 字段**。

    实测 `SC` 与目录前缀有 219 处不一致（174 个指数 + 16 只股票，如 `sz000003` 的 SC=1），
    故它不可作判据。这里确认两条：给了名单就翻成带前缀的符号；不给就保持裸代码。
    """
    bare = SecurityMasterDataSource(FIXTURE)  # 不给符号名单

    assert bare.info("600000").symbol == "600000", "不给名单就返回裸代码"
    assert master.info("sh600000").symbol == "sh600000", "给了名单就翻成带前缀的"


def test_an_unknown_symbol_yields_none(master):
    assert master.info("sh999999") is None
    assert master.total_shares("sh999999") is None


# --- 总股本 -------------------------------------------------------------------


def test_total_shares_is_the_real_share_count_in_ten_thousand_shares(master):
    """`ZGB` 单位是万股：浦发 3,330,583.75 万股 = 333.06 亿股。"""
    shares = master.total_shares("sh600000")

    assert shares == pytest.approx(3_330_583.75)
    assert shares / 10_000 == pytest.approx(333.06, abs=0.01), "折成亿股应约 333 亿"


def test_total_shares_takes_no_date_because_it_is_a_snapshot(master):
    """**它刻意不收日期参数。**

    这张表只有一组值（带 `GXRQ` 更新日期），**不是逐日历史**。给一个
    ``total_shares(symbol, on)`` 会诱导读者以为它是时点正确的（ADR-0006 关心的正是这种静默
    失真），而它做不到。故用签名把局限表达出来——这条测试就是那个签名的守卫。
    """
    import inspect

    parameters = list(inspect.signature(SecurityMasterDataSource.total_shares).parameters)

    assert parameters == ["self", "symbol"], "签名里不该出现日期参数"


# --- 校验与报错 ---------------------------------------------------------------


def test_a_missing_file_is_reported(tmp_path):
    with pytest.raises(FileNotFoundError):
        SecurityMasterDataSource(tmp_path / "nope.dbf").listing_dates()


def test_a_broken_frame_is_rejected(tmp_path):
    """帧与头部自述不符 → **报错**，不静默按头部截断（那会给出看似正常的错数字）。"""
    broken = tmp_path / "base.dbf"
    broken.write_bytes(FIXTURE.read_bytes()[:-10])

    with pytest.raises(MarketDataError, match="帧与头部自述不符"):
        SecurityMasterDataSource(broken).listing_dates()


def test_a_missing_eof_marker_is_rejected(tmp_path):
    mutated = tmp_path / "base.dbf"
    raw = bytearray(FIXTURE.read_bytes())
    raw[-1] = 0x00
    mutated.write_bytes(bytes(raw))

    with pytest.raises(MarketDataError, match="EOF"):
        SecurityMasterDataSource(mutated).listing_dates()


def test_an_unknown_version_byte_is_rejected(tmp_path):
    """版本字节变了说明格式变了——静默按 dBase III 解析会给出看似正常的错数字。"""
    mutated = tmp_path / "base.dbf"
    raw = bytearray(FIXTURE.read_bytes())
    raw[0] = 0x30
    mutated.write_bytes(bytes(raw))

    with pytest.raises(MarketDataError, match="版本字节"):
        SecurityMasterDataSource(mutated).listing_dates()


def test_a_record_length_inconsistent_with_the_field_widths_is_rejected(tmp_path):
    """记录长必须等于字段宽度合计 + 1（删除标记）。"""
    mutated = tmp_path / "base.dbf"
    raw = bytearray(FIXTURE.read_bytes())
    struct.pack_into("<H", raw, 10, 999)  # 改记录长
    mutated.write_bytes(bytes(raw))

    with pytest.raises(MarketDataError, match="字段宽度合计"):
        SecurityMasterDataSource(mutated).listing_dates()


def test_the_loader_helper_returns_listing_dates():
    dates = load_listing_dates(FIXTURE, symbols=["sh600000", "sz000001"])

    assert dates["sh600000"] == dt.date(1999, 11, 10)
    assert dates["sz000001"] == dt.date(1991, 4, 3)


def test_a_code_absent_from_the_symbol_list_keeps_its_bare_form():
    """名单里没有的代码**保持裸形式**，而不是被猜一个前缀。

    主表的 `SC` 字段不可靠（219 处不一致），故市场归属只能来自 ``.day`` 目录。名单不完整时
    猜前缀会**静默给错符号**，而保持裸形式至少是可辨认的——消费方会因对不上而发现。
    """
    dates = load_listing_dates(FIXTURE, symbols=["sh600000"])

    assert "600519" in dates, "不在名单里 → 保持裸代码"
    assert "sh600519" not in dates, "不该猜前缀"
    assert dates["600519"] == dt.date(2001, 8, 27)


def test_the_count_matches_the_header(master):
    assert master.count == 4
