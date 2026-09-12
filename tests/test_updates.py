"""更新检查：数据变了没有、变了要不要重跑（票据 #9）。

两种「变化」都用**真实 `.day` 字节**精确构造——造一个文件、建基线、改字节、再看判定。
这与解析层既有做法一致（fixture 是真实文件的切片），且两种情形都能精确复现。

最要紧的是**不能用 mtime 判变化**：通达信每天重写这些文件，而本项目的正门不落盘，故「变了没有」
必须从**内容**自证。有一条测试专门钉住这件事。
"""

from __future__ import annotations

import datetime as dt
import os
import struct
import time
from pathlib import Path

import pytest

from mbt.data.errors import MarketDataError
from mbt.data.updates import (
    APPENDED,
    INSIDE_GAP,
    NEW,
    REVISED,
    TAIL_BARS,
    TAIL_GAP,
    UNCHANGED,
    boundary_of,
    check_updates,
    compare,
    load_boundaries,
    save_boundaries,
)

FIXTURE = Path(__file__).parent / "fixtures" / "tdx" / "sh" / "lday" / "sh600000.day"

#: 一条 ``.day`` 记录：``(date, open, high, low, close, amount, volume, reserved)``，
#: 价格为「元 × 100」。**8 个字段**——与 :data:`mbt.data.tdx.DAY_DTYPE` 的布局一致。
RECORD = struct.Struct("<IIIIIfII")


def read_records(path: Path):
    raw = path.read_bytes()
    return [RECORD.unpack_from(raw, offset) for offset in range(0, len(raw), RECORD.size)]


def write_records(path: Path, records) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"".join(RECORD.pack(*record) for record in records))


def with_price(record, price: int):
    """把一条记录的**四个价格**都换成 ``price``，其余字段原样。

    四个价格必须**一起**改：只改收盘价会造出「收在最高价之上」的不自洽 K 线，被解析层
    :meth:`TdxDataSource._reject_anomalies` 拒绝，那个标的就会被整个跳过——测试也就测不到
    想测的东西了（本模块第一版就这么栽过）。
    """
    date, _, _, _, _, amount, volume, reserved = record
    return (date, price, price, price, price, amount, volume, reserved)


@pytest.fixture
def data_root(tmp_path):
    """一个只含 ``sh600000`` 的数据源，内容取**真实 fixture 的全部记录**。

    用真实字节而非合成：这样「追加一根 K 线」「改一根 K 线的收盘价」都作用于真实记录，
    而不是我编出来的形状。
    """
    records = read_records(FIXTURE)
    root = tmp_path / "vipdoc"
    write_records(root / "sh" / "lday" / "sh600000.day", records)
    return root


def day_path(root: Path, symbol="sh600000") -> Path:
    return root / symbol[:2] / "lday" / f"{symbol}.day"


def next_trading_day(records):
    """造一条在原末根之后约一周的新记录（价格与末根一致，故末段之外）。"""
    date, o, h, low, close, amount, volume, reserved = records[-1]
    when = dt.datetime.strptime(str(date), "%Y%m%d").date() + dt.timedelta(days=7)
    return (int(f"{when:%Y%m%d}"), close, close, close, close, 0.0, 1000, 0)


# --- 边界与摘要 ---------------------------------------------------------------


def test_boundary_carries_first_last_bars_and_a_tail_digest(data_root):
    from mbt.data import TdxDataSource

    prices = TdxDataSource(data_root).daily("sh600000")
    boundary = boundary_of("sh600000", prices)

    assert boundary.symbol == "sh600000"
    assert boundary.first_date == prices.index[0].date()
    assert boundary.last_date == prices.index[-1].date()
    assert boundary.bars == len(prices)
    assert boundary.tail_digest and len(boundary.tail_digest) == 16
    assert boundary.bars == len(read_records(FIXTURE))


def test_the_tail_digest_covers_only_the_last_few_bars(wide=None):
    """摘要只看末段——**很久以前**的历史变化不该被算成「回补」。

    取 5 根：改动第 3 根（远早于末段）不改摘要，改动倒数第 2 根则改。
    """
    import pandas as pd

    index = pd.bdate_range("2024-01-02", periods=20)

    def frame(closes):
        return pd.DataFrame(
            {"open": closes, "high": closes, "low": closes, "close": closes, "volume": [1] * 20},
            index=index,
        )

    base = [10.0] * 20
    early_changed = list(base)
    early_changed[2] = 99.0  # 第 3 根，远早于末段
    late_changed = list(base)
    late_changed[-2] = 99.0  # 倒数第 2 根，在末段里

    assert (
        boundary_of("x", frame(base)).tail_digest
        == boundary_of("x", frame(early_changed)).tail_digest
    ), "末段之外的改动不该改变摘要"
    assert (
        boundary_of("x", frame(base)).tail_digest
        != boundary_of("x", frame(late_changed)).tail_digest
    ), "末段之内的改动必须改变摘要"
    assert TAIL_BARS == 5


def test_an_empty_frame_has_no_boundary():
    import pandas as pd

    with pytest.raises(MarketDataError, match="没有 K 线"):
        boundary_of("sh600000", pd.DataFrame())


# --- 变化归类（compare：拿当前价格表回到上次的窗口去比） ---------------------


def test_a_symbol_not_in_the_previous_baseline_is_new():
    """首次见到即「新标的」——``check_updates`` 直接判，不经 ``compare``。"""
    report = _one_symbol_report(_frame([10.0] * 20), previous=None)

    assert report.by_kind == {NEW: ["sh600000"]}


def test_the_same_data_is_unchanged():
    frame = _frame([10.0] * 20)
    before = boundary_of("sh600000", frame)

    assert compare(before, frame) == UNCHANGED


def test_appending_bars_is_pure_append_repeatedly():
    """**这条钉住一个真 bug**：连续追加必须**每次都**判「纯追加」。

    早期版本让 ``boundary_of`` 收一个「锚」（把摘要算在上次的窗口上），却把**新的**末根日期
    记进去——于是边界自相矛盾，下一轮的锚又跟着移，**从第二次追加起全被误判成「回补/修正」**，
    退出码恒为 1，这个命令的核心用途从第二天起就废了。

    只测一轮追加是发现不了的，故这里连着测三轮。
    """
    frame = _frame([10.0] * 20)
    before = boundary_of("sh600000", frame)

    for extra in range(1, 4):
        grown = _frame([10.0] * (20 + extra))
        assert compare(before, grown) == APPENDED, f"第 {extra} 轮追加"
        # 与真实用法一致：每轮都把新边界当作下一轮的基线（关键，正是这里出的错）
        before = boundary_of("sh600000", grown)


def test_a_changed_value_inside_the_tail_window_is_a_revision():
    frame = _frame([10.0] * 20)
    before = boundary_of("sh600000", frame)
    changed = _frame([10.0] * 19 + [11.0])

    assert compare(before, changed) == REVISED


def test_a_changed_value_deep_in_history_is_a_revision_if_the_count_changes():
    """更早的**数值**改动测不出来（TAIL_BARS 的既定代价），但**增删**一定测得出。

    这条把代价写清楚：删掉一根最老的（根数变少）→ 判「回补/修正」。
    """
    frame = _frame([10.0] * 20)
    before = boundary_of("sh600000", frame)

    dropped = _frame([10.0] * 19)
    assert compare(before, dropped) == REVISED


def test_a_middle_deletion_hidden_by_an_append_is_a_revision():
    """删中间一根、末尾补一根（**总根数不变**）也必须判出修正。

    只看「末段摘要 + 总根数」的实现会把它判成「无变化」：末段未动、总数也没变。
    而截到**上次末根**之后的根数少了一根，故这条能测出来。
    """
    import pandas as pd

    days = pd.bdate_range("2024-01-02", periods=20)
    original = _frame_at(days, [10.0] * 20)
    before = boundary_of("sh600000", original)

    # 去掉第 5 根，末尾补一根——总根数仍是 20
    kept = list(days[:4]) + list(days[5:])
    altered_index = pd.DatetimeIndex([*kept, days[-1] + pd.Timedelta(days=1)])
    altered = _frame_at(altered_index, [10.0] * 20)

    assert len(altered) == len(original) == 20, "总根数必须相同，否则测的不是这条"
    assert compare(before, altered) == REVISED


def test_a_value_change_outside_the_tail_window_is_the_documented_blind_spot():
    """**把代价钉住**：末 5 根之外的**数值**改动测不出来，判「无变化」。

    要覆盖它就得对整段历史做内容摘要，那会随历史增长而变慢。这里显式记录该取舍，
    免得日后有人以为「无变化」等于「数据一定没变」。**增删**仍能测出（见相邻两条）。
    """
    frame = _frame([10.0] * 20)
    before = boundary_of("sh600000", frame)

    altered = _frame([10.0] * 2 + [99.0] + [10.0] * 17)  # 第 3 根，远早于末 5 根

    assert compare(before, altered) == UNCHANGED, "这是已知取舍，不是回归"
    assert TAIL_BARS == 5


def test_backfilling_older_history_is_a_revision_not_a_footnote():
    """在最前面补全更早的历史 → **回补/修正**，旧结果作废。

    这不是过慎：后复权以**序列首根**为基准（ADR-0003），故补全更早的历史会改变**整条复权
    序列的尺度**——旧结果与新结果不在同一个刻度上，不可比。
    """
    import pandas as pd

    days = pd.bdate_range("2024-06-03", periods=20)
    before = boundary_of("sh600000", _frame_at(days, [10.0] * 20))

    older = pd.bdate_range("2024-01-02", periods=100)
    backfilled = _frame_at(older, [10.0] * 100)

    assert compare(before, backfilled) == REVISED


def _frame(closes):
    import pandas as pd

    return _frame_at(pd.bdate_range("2024-01-02", periods=len(closes)), closes)


def _frame_at(index, closes):
    import pandas as pd

    return pd.DataFrame(
        {
            "open": closes,
            "high": closes,
            "low": closes,
            "close": closes,
            "volume": [1] * len(closes),
        },
        index=index,
    )


def _one_symbol_report(frame, previous):
    """造一个单标的的假数据源，跑一次 check_updates。"""
    import tempfile
    from pathlib import Path

    with tempfile.TemporaryDirectory() as raw:
        root = Path(raw) / "vipdoc"
        target = root / "sh" / "lday"
        target.mkdir(parents=True)
        payload = bytearray()
        for stamp, close in zip(frame.index, frame["close"], strict=True):
            price = int(round(close * 100))
            payload += RECORD.pack(int(f"{stamp:%Y%m%d}"), price, price, price, price, 0.0, 1000, 0)
        (target / "sh600000.day").write_bytes(bytes(payload))
        return check_updates(["sh600000"], tdx_root=root, previous=previous)


def _boundary_summary(report):
    return {symbol: (item.bars, item.last_date) for symbol, item in report.boundaries.items()}


# --- 端到端：纯追加与回补 ----------------------------------------------------


def test_appending_a_bar_is_detected_as_a_pure_append(data_root):
    first = check_updates(["sh600000"], tdx_root=data_root)
    assert first.by_kind == {NEW: ["sh600000"]}
    assert first.compared_with is None, "首次建立基线"

    records = read_records(day_path(data_root))
    write_records(day_path(data_root), [*records, next_trading_day(records)])

    second = check_updates(["sh600000"], tdx_root=data_root, previous=first.boundaries)

    assert second.by_kind == {APPENDED: ["sh600000"]}
    assert second.needs_rerun is False, "纯追加不让旧结果作废"


def test_rewriting_an_old_bar_is_detected_as_a_revision(data_root):
    """**本票的核心**：把倒数第 2 根的收盘价改掉，必须判为「回补/修正」。"""
    first = check_updates(["sh600000"], tdx_root=data_root)

    records = read_records(day_path(data_root))
    index = len(records) - 2
    records[index] = with_price(records[index], int(records[index][4] * 1.1))
    write_records(day_path(data_root), records)

    second = check_updates(["sh600000"], tdx_root=data_root, previous=first.boundaries)

    assert second.by_kind == {REVISED: ["sh600000"]}
    assert second.needs_rerun is True
    assert second.revised == ["sh600000"]


def test_an_untouched_file_is_unchanged(data_root):
    first = check_updates(["sh600000"], tdx_root=data_root)
    second = check_updates(["sh600000"], tdx_root=data_root, previous=first.boundaries)

    assert second.by_kind == {UNCHANGED: ["sh600000"]}
    assert second.needs_rerun is False


def test_a_rewritten_file_with_identical_content_is_not_a_change(data_root):
    """**这条钉住「不用 mtime 判变化」。**

    把文件原样重写一遍（内容一模一样，但 mtime 变了）——判据必须仍是「无变化」。若实现用了
    ``mtime``，通达信每天重写文件就会让它**每天报「全部标的都变了」**，功能等于没有。
    """
    first = check_updates(["sh600000"], tdx_root=data_root)

    path = day_path(data_root)
    payload = path.read_bytes()
    before_mtime = path.stat().st_mtime_ns
    time.sleep(0.01)
    path.write_bytes(payload)  # 内容相同、mtime 变新
    os.utime(path, ns=(path.stat().st_atime_ns, before_mtime + 10**9))

    assert path.stat().st_mtime_ns != before_mtime, "mtime 确实变了，否则这条测试无从判别"

    second = check_updates(["sh600000"], tdx_root=data_root, previous=first.boundaries)
    assert second.by_kind == {UNCHANGED: ["sh600000"]}


# --- 缺口：能判的与判不了的 ---------------------------------------------------


def test_an_inside_gap_is_reported_with_its_resumption_day(data_root):
    """区间内停牌**能判**：挖掉中间若干根，报「缺了哪几天、哪天恢复」。

    交易日数取自**市场日历**（全市场日期并集），是数出来的，不是按日历天估算的。

    这里**必须放两只标的**：市场日历若只由一个标的的日期构成，它自己停牌的那些日子会连同
    从日历里消失，缺口就测不出来了（见另一条测试）。
    """
    records = read_records(day_path(data_root))
    # 另一只标的作为「市场照常开门」的见证者，日期全覆盖。
    write_records(day_path(data_root, "sz000001"), records)
    trimmed = [record for position, record in enumerate(records) if position not in (30, 31, 32)]
    write_records(day_path(data_root), trimmed)

    report = check_updates(["sh600000", "sz000001"], tdx_root=data_root)

    inside = report.gaps_by_kind(INSIDE_GAP)
    assert len(inside) == 1
    gap = inside[0]
    assert gap.symbol == "sh600000"
    assert gap.missing == 3, "挖掉 3 根，市场日历上也应缺 3 个交易日"
    assert gap.resumed is not None
    assert gap.undecidable is False


def test_a_single_symbol_cannot_reveal_its_own_inside_gaps(data_root):
    """**把上面那前提钉住**：只检查一个标的时，它自己的停牌日会从日历里消失。

    这不是实现缺陷，是「市场日历 = 全市场日期并集」这一做法的固有性质。故调用方应当传全市场
    （或至少一批互不相关的标的）；只传一个时，区间内缺口的判定能力是退化的。这条测试的作用是
    **让这个局限显式可见**，而不是让它悄悄存在。
    """
    records = read_records(day_path(data_root))
    trimmed = [record for position, record in enumerate(records) if position not in (30, 31, 32)]
    write_records(day_path(data_root), trimmed)

    report = check_updates(["sh600000"], tdx_root=data_root)

    assert report.gaps_by_kind(INSIDE_GAP) == [], "单标的时判不出自己的停牌——这是已知局限，不是回归"


def test_a_trailing_gap_is_reported_as_a_fact_and_flagged_undecidable(data_root):
    """尾部空缺**只报事实**，并显式标注不可区分——**不猜**是停牌还是退市。"""
    long_symbol = "sh600000"
    short_symbol = "sz000001"
    records = read_records(day_path(data_root))
    write_records(day_path(data_root), records)
    # 另一只只到一半就没了
    write_records(day_path(data_root, short_symbol), records[:30])

    report = check_updates([long_symbol, short_symbol], tdx_root=data_root)

    tails = report.gaps_by_kind(TAIL_GAP)
    assert len(tails) == 1
    tail = tails[0]
    assert tail.symbol == short_symbol
    assert tail.missing == len(records) - 30, "尾部缺的交易日数应等于被截掉的那些"
    assert tail.resumed is None
    assert tail.undecidable is True, "尾部空缺必须被标为不可区分"


def test_a_symbol_current_to_the_market_close_has_no_tail_gap(data_root):
    report = check_updates(["sh600000"], tdx_root=data_root)

    assert report.gaps_by_kind(TAIL_GAP) == []


def test_non_stocks_are_skipped_without_being_counted_as_changes(data_root):
    """指数/基金不参与——它们本就不该进这里，且跳过**不算变化**。"""
    records = read_records(day_path(data_root))
    write_records(day_path(data_root, "sh000001"), records)

    report = check_updates(["sh600000", "sh000001"], tdx_root=data_root)

    assert "sh000001" not in report.boundaries
    assert sum(report.counts().values()) == 1


# --- 清单的读写 ---------------------------------------------------------------


def test_boundaries_round_trip_through_csv(tmp_path, data_root):
    report = check_updates(["sh600000"], tdx_root=data_root)
    path = tmp_path / "boundaries.csv"

    save_boundaries(path, report.boundaries)
    loaded = load_boundaries(path)

    assert loaded.keys() == report.boundaries.keys()
    original = report.boundaries["sh600000"]
    assert loaded["sh600000"] == original


def test_the_boundary_file_is_human_readable_csv(tmp_path, data_root):
    """CSV 而非二进制：要能被人打开看、被 git 追踪。"""
    report = check_updates(["sh600000"], tdx_root=data_root)
    path = save_boundaries(tmp_path / "b.csv", report.boundaries)

    lines = path.read_text(encoding="utf-8").strip().splitlines()
    assert lines[0] == "symbol,first_date,last_date,bars,tail_digest"
    assert lines[1].startswith("sh600000,")


def test_a_missing_boundary_file_is_an_error_not_a_silent_first_run(tmp_path):
    """「文件不在」与「首次运行」是两件事——静默当成首次会把所有修正退化成「新标的」。"""
    with pytest.raises(MarketDataError, match="边界清单不存在"):
        load_boundaries(tmp_path / "nope.csv")


def test_a_boundary_file_with_the_wrong_columns_is_rejected(tmp_path):
    bad = tmp_path / "bad.csv"
    bad.write_text("symbol,bars\nsh600000,60\n", encoding="utf-8")

    with pytest.raises(MarketDataError, match="缺少列"):
        load_boundaries(bad)


def test_a_verified_baseline_enables_revision_detection_end_to_end(tmp_path, data_root):
    """走一遍真实流程：建基线 → 落盘 → 改字节 → 读回基线 → 判出修正。"""
    path = tmp_path / "boundaries.csv"
    save_boundaries(path, check_updates(["sh600000"], tdx_root=data_root).boundaries)

    records = read_records(day_path(data_root))
    records[-1] = with_price(records[-1], records[-1][4] + 1)
    write_records(day_path(data_root), records)

    report = check_updates(["sh600000"], tdx_root=data_root, previous=load_boundaries(path))

    assert report.needs_rerun is True
    assert report.compared_with == 1
