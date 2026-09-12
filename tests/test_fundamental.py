"""按期财务数据：解析与时点纪律（票据 #8）。

fixture 是**真实文件的切片**（`tests/fixtures/cw/gpcw20251231.dat`，2025 年报里的三个标的，
7,061 字节），故「帧恒等式」「字段索引」这些假设被一起锁住——造一份合成的 `gpcw` 等于用实现
验证实现。

最要紧的一组是**时点纪律**：公告日按报告期对齐等于提前数月知道尚未公布的财报，而那种错误
不会报错，只会让结论偏乐观（ADR-0006）。
"""

from __future__ import annotations

import datetime as dt
import struct
from pathlib import Path

import pandas as pd
import pytest

from mbt.data.errors import MarketDataError
from mbt.data.fundamental import (
    HEADER_SIZE,
    MIN_ANNOUNCEMENT_YEAR,
    CwDataSource,
    FinancialRecord,
    load_financials,
    non_loss_mask,
)

FIXTURE = Path(__file__).parent / "fixtures" / "cw" / "gpcw20251231.dat"

#: 由独立手段（按研究文档的偏移表直接读字节）解出的期望值——**不是照实现抄的**。
#:
#: 公告日一律写成真实的 **6 位 YYMMDD**（`260321` 而非 `20260321`）：实测全部 303,491 条真实
#: 记录都是 6 位，且 float32 装不下 8 位整数。
EXPECTED = {
    "sh600000": {"announcement": 260331, "eps": 1.52, "bvps": 22.13},
    "sz000001": {"announcement": 260321, "eps": 2.07, "bvps": 23.25},
    "sh600519": {"announcement": 260417, "eps": 65.66, "bvps": 195.3554},
}


@pytest.fixture
def cw_root():
    """fixture 所在的目录（``load_financials`` / ``CwDataSource`` 收的是目录）。"""
    return FIXTURE.parent


def test_the_fixture_satisfies_the_frame_identity():
    """帧恒等式：``20 + nstk×索引 + nstk×块 == 文件长度``。

    这条把「布局假设」本身钉住——它是本模块对文件格式的全部主张，而它在**本机全部 147 个
    真实文件上成立**（见 `docs/research/tdx-cw-format.md`）。
    """
    raw = FIXTURE.read_bytes()
    index_size = struct.unpack_from("<H", raw, 10)[0]
    block_size = struct.unpack_from("<I", raw, 12)[0]
    count = struct.unpack_from("<H", raw, 6)[0]

    assert (index_size, block_size) == (11, 2336)
    assert len(raw) == HEADER_SIZE + count * index_size + count * block_size


def test_the_header_is_read_by_explicit_offsets_not_a_struct_string():
    """头部按偏移读：版本 1、报告期 20251231、索引 11、块 2336。

    研究文档曾把头部写成一个与它自己偏移表对不上的 struct 串（``'<1hI1H3L'`` 只出 6 个值，
    而偏移表有 7 个字段）。这条以真实字节锁住正确的读法。
    """
    raw = FIXTURE.read_bytes()

    assert struct.unpack_from("<H", raw, 0)[0] == 1
    assert struct.unpack_from("<I", raw, 2)[0] == 20251231
    assert struct.unpack_from("<H", raw, 6)[0] == 3
    assert struct.unpack_from("<H", raw, 10)[0] == 11
    assert struct.unpack_from("<I", raw, 12)[0] == 2336


def test_real_field_values_match_independently_decoded_constants(cw_root):
    """字段索引经**真实数据**验证：公告日、每股收益、每股净资产与独立解出的常数一致。

    这三个常数是按偏移表直接读字节得到的，不是照实现抄的。
    """
    source = CwDataSource(cw_root)

    for symbol, expected in EXPECTED.items():
        record = source.records(symbol)[0]

        assert record.report_period == dt.date(2025, 12, 31)
        assert record.values["eps_ytd"] == pytest.approx(expected["eps"], abs=1e-4), symbol
        assert record.values["bvps"] == pytest.approx(expected["bvps"], abs=1e-4), symbol


def test_the_announcement_date_is_decoded_from_six_digit_yymmdd(cw_root):
    """公告日是 **6 位 ``YYMMDD``**，世纪按 70 分界。

    fixture 里三条全是 6 位：``260321`` / ``260331`` / ``260417`` → 2026 年的三个三月/四月。
    """
    source = CwDataSource(cw_root)

    assert source.records("sz000001")[0].announcement_date == dt.date(2026, 3, 21)
    assert source.records("sh600000")[0].announcement_date == dt.date(2026, 3, 31)
    assert source.records("sh600519")[0].announcement_date == dt.date(2026, 4, 17)


def test_an_eight_digit_value_is_rejected_rather_than_silently_shifted():
    """8 位整数**不认**——不是偷懒，是它做不到。

    实测 ``float32(20260331)`` 是 ``20260332``、``float32(20260417)`` 是 ``20260416``：float32
    装不下 8 位整数。硬要按 ``YYYYMMDD`` 解，得到的日期会**错一天**（或直接抛错被我吞掉），
    而这种错误不会报错、只会让取到的财报错期。实测全部 303,491 条非零公告日里 8 位的**一条
    都没有**，故此路本就不通。
    """
    from mbt.data.fundamental import _decode_announcement

    assert _decode_announcement(0.0) is None
    assert _decode_announcement(41012.0) is None, "残缺位数不猜"
    assert _decode_announcement(20260331.0) is None, "8 位不认"


def test_a_six_digit_value_before_the_pivot_is_read_as_the_1900s():
    """世纪分界取 70：``891231`` → 1989-12-31（财务数据始于 1988，故本范围内无歧义）。"""
    from mbt.data.fundamental import _decode_announcement

    assert _decode_announcement(891231.0) == dt.date(1989, 12, 31)
    assert _decode_announcement(700101.0) == dt.date(1970, 1, 1)
    assert _decode_announcement(691231.0) == dt.date(2069, 12, 31)


def test_the_announcement_date_is_not_the_report_period(cw_root):
    """公告日**不等于**报告期——这正是本模块存在的理由。

    2025-12-31 的年报，公告日落在 2026 年 3–4 月。若按报告期对齐，等于在 2026-01-01 就
    知道了这份财报（提前约 3 个月）。
    """
    source = CwDataSource(cw_root)

    for symbol in EXPECTED:
        record = source.records(symbol)[0]
        assert record.announcement_date > record.report_period
        assert (record.announcement_date - record.report_period).days > 60


# --- 时点纪律（本票的核心） ---------------------------------------------------


def test_as_of_returns_none_before_the_announcement(cw_root):
    """公告日**之前**取不到这一期——这是 ADR-0006 的落点。

    若按报告期对齐，2026-01-15 就会拿到 2025 年报（实际 3 月才公布）。
    """
    source = CwDataSource(cw_root)

    assert source.as_of("sh600000", dt.date(2026, 3, 30)) is None
    assert source.as_of("sh600000", dt.date(2026, 3, 31)) is not None


def test_moving_the_announcement_date_by_one_day_changes_the_pick(cw_root):
    """把公告日**推迟一天**，评估日的取值必须随之改变——这条锁住「确实按公告日筛」。

    只断言「公告日前返回 None」是弱判据（按报告期筛也会在报告期前返回 None）。这里直接
    比较边界两侧的**同一天**：公告日当天有、前一天没有。
    """
    source = CwDataSource(cw_root)
    announcement = source.records("sz000001")[0].announcement_date

    assert source.as_of("sz000001", announcement) is not None
    assert source.as_of("sz000001", announcement - dt.timedelta(days=1)) is None


def test_a_record_whose_announcement_is_a_placeholder_is_not_usable(cw_root):
    """公告日等于报告期（占位符）→ 不可用。实测 1995 年报里 670/673 条如此。"""
    placeholder = FinancialRecord(
        symbol="sh600000",
        report_period=dt.date(1995, 12, 31),
        announcement_date=dt.date(1995, 12, 31),
        values={"net_profit_ytd": 1.0},
    )

    assert placeholder.usable is False


def test_a_record_before_the_usable_year_is_not_usable():
    """公告日年份早于 2005 → 不可用。实测 2001 年报有一批约 700 天的荒诞滞后。"""
    old = FinancialRecord(
        symbol="sh600000",
        report_period=dt.date(2003, 12, 31),
        announcement_date=dt.date(2004, 4, 1),
        values={"net_profit_ytd": 1.0},
    )

    assert old.announcement_date.year < MIN_ANNOUNCEMENT_YEAR
    assert old.usable is False


def test_a_missing_announcement_date_is_not_usable():
    """公告日为 0（未披露）→ 不可用，而不是当成「报告期当天公布的」。"""
    missing = FinancialRecord(
        symbol="sh600000",
        report_period=dt.date(2020, 12, 31),
        announcement_date=None,
        values={"net_profit_ytd": 1.0},
    )

    assert missing.usable is False


def test_symbols_are_recovered_from_the_bare_six_digit_codes(cw_root):
    """`gpcw` 只有裸代码，市场前缀由代码首位推出——本机实测北交所用 ``920xxx``。"""
    financials = load_financials(cw_root)

    assert set(financials) == {"sh600000", "sh600519", "sz000001"}


def test_the_prefix_rule_covers_the_three_markets():
    """前缀推断要覆盖沪、深、北三市，否则会静默漏掉整个市场。"""
    from mbt.data.fundamental import _symbol_of

    assert _symbol_of("600000") == "sh600000"
    assert _symbol_of("000001") == "sz000001"
    assert _symbol_of("300750") == "sz300750"
    assert _symbol_of("920819") == "bj920819"


# --- 「非亏损」过滤 -----------------------------------------------------------


def test_non_loss_mask_is_true_only_after_the_announcement(cw_root):
    """「非亏损」在公告日之前为假——过滤因此也是时点正确的。"""
    financials = load_financials(cw_root)
    index = pd.bdate_range("2026-01-02", "2026-05-29")
    columns = pd.Index(["sh600000", "sz000001", "sh600519"])

    mask = non_loss_mask(financials, index, columns)

    early = mask.loc[:"2026-03-20"].to_numpy()
    assert not early.any(), "公告日之前不该有任何一个标的通过"

    late = mask.loc["2026-04-20":]
    assert late.all().all(), "公告日之后三个标的都应通过（fixture 里都是盈利的）"


def test_non_loss_mask_excludes_a_symbol_without_financial_data(cw_root):
    """没有财务数据的标的**排除**（而不是放行）。

    这是入场过滤，排除是安全方向——报错会让 1990 年代的回测直接崩，而放行会让「对财务条件
    一无所知」的标的混进样本。
    """
    financials = load_financials(cw_root)
    index = pd.bdate_range("2026-04-20", periods=5)
    columns = pd.Index(["sh600000", "sh601999"])  # 后者 fixture 里没有

    mask = non_loss_mask(financials, index, columns)

    assert mask["sh600000"].all()
    assert not mask["sh601999"].any()


def test_non_loss_uses_the_cumulative_figure_not_the_quarter():
    """用**累计**归母净利润判非亏损，不是单季。

    构造一个「累计赚、单季亏」的标的：用单季判会把它当成亏损（实为盈利），反之亦错。
    这条直接钉住口径选择。
    """
    record = FinancialRecord(
        symbol="sh600000",
        report_period=dt.date(2024, 9, 30),
        announcement_date=dt.date(2024, 10, 20),
        values={"net_profit_ytd": 100.0, "net_profit_quarter": -500.0},
    )
    index = pd.bdate_range("2024-10-21", periods=3)

    mask = non_loss_mask({"sh600000": (record,)}, index, pd.Index(["sh600000"]))

    assert mask.all().all(), "累计为正即非亏损，单季为负不改变结论"


def test_non_loss_excludes_a_symbol_whose_cumulative_is_negative():
    record = FinancialRecord(
        symbol="sh600000",
        report_period=dt.date(2024, 12, 31),
        announcement_date=dt.date(2025, 4, 20),
        values={"net_profit_ytd": -1.0},
    )
    index = pd.bdate_range("2025-04-21", periods=3)

    mask = non_loss_mask({"sh600000": (record,)}, index, pd.Index(["sh600000"]))

    assert not mask.any().any()


# --- 解析的稳健性 -------------------------------------------------------------


def test_a_missing_directory_says_what_is_wrong(tmp_path):
    with pytest.raises(MarketDataError, match="没有 gpcw"):
        load_financials(tmp_path)


def test_a_file_whose_frame_does_not_match_its_header_is_rejected(tmp_path):
    """帧与头部自述不符 → **报错**，不静默按头部截断。

    截断会让「格式已经变了」看起来像「文件刚好短了一点」，而算出来的数字会是错的。
    """
    broken = tmp_path / "gpcw20251231.dat"
    raw = bytearray(FIXTURE.read_bytes())
    broken.write_bytes(bytes(raw[:-4]))  # 砍掉 4 字节

    with pytest.raises(MarketDataError, match="帧与头部自述不符"):
        load_financials(tmp_path)


def test_an_unknown_format_version_is_rejected(tmp_path):
    """版本号变了要报错——静默按 v1 解析会给出看似正常的错数字。"""
    mutated = tmp_path / "gpcw20251231.dat"
    raw = bytearray(FIXTURE.read_bytes())
    struct.pack_into("<H", raw, 0, 2)
    mutated.write_bytes(bytes(raw))

    with pytest.raises(MarketDataError, match="格式版本"):
        load_financials(tmp_path)


def test_an_empty_period_file_is_skipped_not_an_error(tmp_path):
    """空报告期（本机 26 个）是正常状态——2002 前的一/三季报不强制披露。

    真实空文件是**恰好 20 字节**的头部，且块大小字段被填成 ``0xFFFFFFFC``（nstk=0，故帧
    恒等式仍成立）。这里照真实形状构造，而不是随手拼一个短头。
    """
    empty = bytearray(HEADER_SIZE)
    struct.pack_into("<H", empty, 0, 1)
    struct.pack_into("<I", empty, 2, 19920331)
    struct.pack_into("<H", empty, 6, 0)
    struct.pack_into("<H", empty, 10, 11)
    struct.pack_into("<I", empty, 12, 0xFFFFFFFC)
    (tmp_path / "gpcw19920331.dat").write_bytes(bytes(empty))
    (tmp_path / "gpcw20251231.dat").write_bytes(FIXTURE.read_bytes())

    financials = load_financials(tmp_path)

    assert set(financials) == {"sh600000", "sh600519", "sz000001"}
    assert CwDataSource(tmp_path).periods == (dt.date(2025, 12, 31),)


def test_a_truncated_empty_header_is_rejected(tmp_path):
    """自称 0 条却不是 20 字节 → 报错。空文件也要合乎格式，否则是一份损坏的文件。"""
    truncated = bytearray(18)
    struct.pack_into("<H", truncated, 0, 1)  # 版本合法，使它走到长度检查
    (tmp_path / "gpcw19920331.dat").write_bytes(bytes(truncated))

    with pytest.raises(MarketDataError, match="应为 20"):
        load_financials(tmp_path)


def test_the_field_indices_are_the_documented_ones(cw_root):
    """字段索引是契约：改一个就会静默取到别的科目。

    光断言 ``FIELDS`` 的常量是**同义反复**（等于断言定义）。故这里拿**真实值**去对
    *研究文档核过的科目口径*：年报送出的累计营收必须远大于其中任何一个单季（单季是累计的
    一段），而归母净利同理。若 73 与 229、或 95 与 231 被写反，这条就会失败。
    """
    record = CwDataSource(cw_root).records("sh600000")[0]

    assert (
        record.values["revenue_ytd"] > record.values["revenue_quarter"] * 2
    ), "年度的累计营收应远大于单季——两者写反时这条会失败"
    assert record.values["net_profit_ytd"] > record.values["net_profit_quarter"] * 2
    # 每股收益 × 总股本应量级上接近归母净利（同一口径的两个侧面），用来交叉印证 0/95。
    assert record.values["eps_ytd"] > 0 and record.values["net_profit_ytd"] > 0


def test_cumulative_and_quarterly_fields_are_both_exposed_under_distinct_names(cw_root):
    """累计与单季**都在**，且名字自带区分——刻意不给一个含糊的 ``revenue``。"""
    record = CwDataSource(cw_root).records("sh600000")[0]

    assert "revenue_ytd" in record.values
    assert "revenue_quarter" in record.values
    assert "revenue" not in record.values, "含糊的名字会让累计/单季被混用"
    # 年报里累计 ≥ 单季（单季是其中一段）
    assert record.values["revenue_ytd"] > record.values["revenue_quarter"]
