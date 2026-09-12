"""回测产物的落盘（票据 #10，AC 3–6）。

产物是**证据**，故测试的重点不只是「文件生成了」，还有「不能悄悄覆盖」「记不下的条件要
明说」「基准对不上要报错」。

两条与 AC 直接对应的检查：`test_run_metadata_records_what_it_cannot_reproduce`
（复现的边界如实登记）与 `test_the_svgs_are_well_formed_xml`（两张图可归档）。
"""

from __future__ import annotations

import dataclasses
import datetime as dt
import json
import xml.etree.ElementTree as ET

import pandas as pd
import pytest
from helpers import curve, trades_table

from mbt.backtest import BacktestResult
from mbt.report import (
    data_snapshot,
    describe_screen,
    describe_strategy,
    render_drawdown_svg,
    render_equity_svg,
    write_run_artifacts,
)
from mbt.screen import Screen
from mbt.signals import momentum


def make_result(values=(100.0, 110.0, 121.0), trades=None):
    equity = curve(list(values))
    return BacktestResult(
        equity_curve=equity,
        trades=trades
        if trades is not None
        else pd.DataFrame(columns=["date", "size", "price", "value", "commission"]),
        final_value=float(equity.iloc[-1]),
        rejected=pd.DataFrame(columns=["date", "symbol", "size", "status", "reason"]),
    )


class ExampleStrategy:
    """仅用于元数据测试的策略。"""


# --- 目录布局与覆盖保护 ------------------------------------------------------


def test_artifacts_land_in_one_directory_per_run(tmp_path):
    """一个 run 的产物是一个整体，落进各自的子目录（Q6 的既定决定）。"""
    run_dir = write_run_artifacts(make_result(), output_dir=tmp_path)

    assert run_dir.parent == tmp_path
    assert {p.name for p in run_dir.iterdir()} == {
        "run.json",
        "metrics.json",
        "equity.csv",
        "equity.svg",
        "drawdown.svg",
        "trades.csv",
        "rejected.csv",
    }


def test_the_rejection_details_and_count_are_persisted(tmp_path):
    """**拒单必须落盘**：明细进 `rejected.csv`，条数与拒单率进 `metrics.json`。

    这条针对的是一次实测教训：拒单只存在内存里的 `BacktestResult.rejected`，CLI 跑完就没了，
    于是「买单全被拒、零成交」这类事实在产物里**完全看不见**，只能靠临时加打印排查——而那种
    排查不会留下任何可供事后核对的东西。
    """
    rejected = pd.DataFrame(
        [
            {
                "date": pd.Timestamp("2024-01-03"),
                "symbol": "sh600000",
                "size": 100,
                "status": "Margin",
                "reason": "",
            },
            {
                "date": pd.Timestamp("2024-01-04"),
                "symbol": "sz000001",
                "size": 200,
                "status": "Rejected",
                "reason": "不在股票池",
            },
        ]
    )
    result = dataclasses.replace(make_result(), rejected=rejected)

    run_dir = write_run_artifacts(result, output_dir=tmp_path)

    saved = pd.read_csv(run_dir / "rejected.csv")
    assert list(saved["symbol"]) == ["sh600000", "sz000001"]
    assert list(saved["status"]) == ["Margin", "Rejected"]
    # CSV 往返会把空字段读成 NaN（标准行为），故这里作归一后再比。
    assert saved["reason"].fillna("").tolist() == ["", "不在股票池"]

    metrics = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
    assert metrics["rejected_orders"] == 2
    assert metrics["rejection_rate"] == 1.0, "一笔成交都没有，故拒单率是 100%"


def test_the_rejection_rate_is_zero_when_nothing_was_rejected(tmp_path):
    run_dir = write_run_artifacts(make_result(), output_dir=tmp_path)

    metrics = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
    assert metrics["rejected_orders"] == 0
    assert metrics["rejection_rate"] == 0.0


def test_the_trade_details_are_persisted(tmp_path):
    """成交明细也落盘——否则「有没有真的成交、按什么价」只能靠摘要里那个笔数。"""
    trades = pd.DataFrame(
        [
            {
                "date": pd.Timestamp("2024-01-03"),
                "size": 100,
                "price": 10.0,
                "value": 1000.0,
                "commission": 0.5,
            }
        ]
    )
    run_dir = write_run_artifacts(make_result(trades=trades), output_dir=tmp_path)

    saved = pd.read_csv(run_dir / "trades.csv")
    assert list(saved["size"]) == [100]
    assert list(saved["price"]) == [10.0]


def test_the_output_directory_is_created_but_a_collision_is_refused(tmp_path):
    """父目录不存在则创建；同名 run 目录已存在则**报错**——产物是证据，不静默覆盖。"""
    target = tmp_path / "nested" / "deeper"
    created_at = dt.datetime(2026, 1, 2, 3, 4, 5)

    first = write_run_artifacts(make_result(), output_dir=target, created_at=created_at)
    assert first.is_dir()

    with pytest.raises(ValueError, match="产物目录已存在"):
        write_run_artifacts(make_result(), output_dir=target, created_at=created_at)


def test_the_run_id_carries_the_timestamp_and_a_short_digest(tmp_path):
    now = dt.datetime(2026, 1, 2, 3, 4, 5)

    run_dir = write_run_artifacts(make_result(), output_dir=tmp_path, created_at=now)

    assert run_dir.name.startswith("20260102-030405-")
    assert len(run_dir.name.split("-")[-1]) == 6


def test_two_runs_on_the_same_data_get_the_same_identity_code(tmp_path):
    """同一份数据 + 同一组策略 → 摘要码相同，故 run_id 只差时间戳：可看出它们同源。"""
    now = dt.datetime(2026, 1, 2, 3, 4, 5)
    later = dt.datetime(2026, 1, 2, 4, 5, 6)

    first = write_run_artifacts(
        make_result(), output_dir=tmp_path, strategy=ExampleStrategy, created_at=now
    )
    second = write_run_artifacts(
        make_result(), output_dir=tmp_path, strategy=ExampleStrategy, created_at=later
    )

    assert first.name.split("-")[-1] == second.name.split("-")[-1]
    assert first.name != second.name


# --- 元数据（AC 3） ----------------------------------------------------------


def test_run_metadata_records_the_strategy_interval_and_data_snapshot(tmp_path):
    """AC 3：策略名、参数、区间、**数据快照版本**。"""
    data_file = tmp_path / "sh600000.day"
    data_file.write_bytes(b"12345678")

    run_dir = write_run_artifacts(
        make_result(),
        output_dir=tmp_path / "out",
        strategy=ExampleStrategy,
        strategy_params={"fast": 5, "slow": 20},
        cash=100_000.0,
        max_positions=3,
        snapshot_paths=[data_file],
        benchmark_prices=curve([3000.0, 3100.0, 3200.0]),
    )

    meta = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))

    assert meta["strategy"]["path"].endswith("ExampleStrategy")
    assert meta["strategy"]["params"] == {"fast": 5, "slow": 20}
    assert meta["period"]["start"] == "2024-01-02"
    assert meta["period"]["trading_days"] == 3
    assert meta["cash"] == 100_000.0
    assert meta["max_positions"] == 3
    assert meta["data_snapshot"]["files"] == 1
    assert len(meta["data_snapshot"]["digest"]) == 16
    assert meta["benchmark"]["symbol"] == "sh000300"
    assert "software" in meta and "python" in meta["software"]


def test_run_metadata_records_what_it_cannot_reproduce(tmp_path):
    """选股规则含匿名函数，**不能**据此逐字复现——这一点必须写在元数据里。

    与其做一个看着能复现、实则悄悄丢条件的机制，不如把缺口写明（Q4 的既定决定）。
    """
    run_dir = write_run_artifacts(
        make_result(),
        output_dir=tmp_path,
        strategy=ExampleStrategy,
        screen=Screen(factor=lambda p: momentum(p["close"], 1), top_n=2),
        screen_label="动量前二",
    )

    screen = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))["screen"]

    assert screen["reproducible"] is False
    assert screen["top_n"] == 2
    assert screen["label"] == "动量前二"
    assert "不能" in screen["note"]


def test_a_non_serializable_strategy_parameter_is_rejected():
    """静默丢掉一个参数会让元数据看着完整而实则缺条件——那比不记更糟。"""
    with pytest.raises(ValueError, match="必须可序列化为 JSON"):
        describe_strategy(ExampleStrategy, {"callback": lambda: None})


def test_describe_screen_returns_none_when_there_is_no_screen():
    assert describe_screen(None) is None


def test_metrics_file_uses_null_instead_of_the_invalid_json_token_nan(tmp_path):
    """``NaN`` **不是合法 JSON**——`jq` / JavaScript / Rust 的解析器都会拒收整个文件。

    Python 的 ``json.dumps`` 默认会写出字面量 ``NaN``，而指标的缺失是**正常状态**
    （无波动、无平仓交易），故必须落成 ``null``。这条用**严格解析器**来验：Python 自己
    的 ``json.loads`` 默认接受 ``NaN``，所以它验不出来。
    """
    run_dir = write_run_artifacts(make_result(values=(100.0, 100.0, 100.0)), output_dir=tmp_path)

    raw = (run_dir / "metrics.json").read_text(encoding="utf-8")

    assert "NaN" not in raw and "Infinity" not in raw
    parsed = json.loads(raw, parse_constant=_reject_constant)
    assert parsed["sharpe"] is None, "无波动时夏普应落成 null"


def _reject_constant(name):
    raise AssertionError(f"metrics.json 里出现了非法 JSON 字面量 {name}")


def test_metrics_file_is_readable_by_a_strict_json_parser(tmp_path):
    """整份文件都要能被严格解析——不只是那个缺失的字段。"""
    run_dir = write_run_artifacts(
        make_result(values=(100.0, 120.0, 90.0, 130.0)), output_dir=tmp_path
    )

    json.loads(
        (run_dir / "metrics.json").read_text(encoding="utf-8"), parse_constant=_reject_constant
    )


def test_the_metrics_file_carries_the_six_metrics_the_ac_names(tmp_path):
    """AC 列的六项**一个都不能少**——`metrics.json` 里必须有这六个键。

    值是否可算则是另一回事：缺失是**正常状态**（无波动、无平仓交易），落成 ``null``。
    故这里断言「键在」而不是「值都是数」——后者会把合法的缺失也判为失败。
    """
    run_dir = write_run_artifacts(
        make_result(
            values=(100.0, 120.0, 90.0, 130.0),
            trades=trades_table([(0, 100, 10.0, 0.0), (1, -100, 12.0, 0.0)]),
        ),
        output_dir=tmp_path,
    )

    metrics = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))

    for name in (
        "annual_return",
        "sharpe",
        "max_drawdown",
        "win_rate",
        "payoff_ratio",
        "turnover",
    ):
        assert name in metrics, name
    # 这一组在这份数据上算得出来，故必须是数值——键在但值错也要拦住。
    for name in ("annual_return", "max_drawdown", "win_rate", "turnover"):
        assert isinstance(metrics[name], int | float), name


def test_the_metrics_file_matches_the_computed_metrics(tmp_path):
    run_dir = write_run_artifacts(
        make_result(values=(100.0, 120.0, 90.0, 130.0)), output_dir=tmp_path
    )

    metrics = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))

    assert metrics["max_drawdown"] == pytest.approx(0.25)
    assert metrics["trading_days"] == 4
    assert isinstance(metrics["notes"], list)


# --- 净值数据 ----------------------------------------------------------------


def test_the_equity_csv_holds_the_curve_and_the_aligned_benchmark(tmp_path):
    """基准列与策略净值**逐行对齐**，否则两条线不可比。"""
    benchmark = curve([3000.0, 3100.0, 3200.0])

    run_dir = write_run_artifacts(make_result(), output_dir=tmp_path, benchmark_prices=benchmark)

    frame = pd.read_csv(run_dir / "equity.csv", index_col="date", parse_dates=True)

    assert list(frame.columns) == ["equity", "benchmark"]
    assert len(frame) == 3
    assert frame["benchmark"].iloc[0] == pytest.approx(3000.0)


def test_a_benchmark_covering_only_part_of_the_period_is_intersected_and_recorded(tmp_path):
    """基准只覆盖一部分时取**交集**，并把**实际用到**的区间记进元数据。

    不静默假装它覆盖了全程——否则「同期对比」这件事本身就是假的。
    """
    benchmark = curve([3000.0, 3100.0], start="2024-01-03")  # 只覆盖后两天

    run_dir = write_run_artifacts(make_result(), output_dir=tmp_path, benchmark_prices=benchmark)

    meta = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
    frame = pd.read_csv(run_dir / "equity.csv", index_col="date", parse_dates=True)

    assert meta["benchmark"]["used_start"] == "2024-01-03"
    assert meta["benchmark"]["used_end"] == "2024-01-04"
    assert meta["benchmark"]["bars"] == 2
    assert frame["benchmark"].notna().sum() == 2


def test_a_benchmark_that_is_a_field_frame_is_rejected_with_a_clear_message(tmp_path):
    """传整张**字段宽表**当基准要报错并说清该怎么办。

    实跑 CLI 时撞到过：`TdxDataSource.daily()` 返回六列，而基准要的是单列序列；不拦的话
    它会在算年化时以 pandas 的 "cannot convert the series to float" 收场，与真正的病因
    隔了三层。
    """
    frame = pd.DataFrame(
        {"open": [1.0, 1.0], "close": [1.0, 1.0]},
        index=pd.bdate_range("2024-01-02", periods=2),
    )

    with pytest.raises(ValueError, match=r"单列价格序列"):
        write_run_artifacts(
            make_result(values=(1.0, 1.0)), output_dir=tmp_path, benchmark_prices=frame
        )


def test_a_benchmark_with_no_overlap_is_rejected(tmp_path):
    """两段时期根本对不上时，对比没有意义——报错。"""
    benchmark = curve([3000.0, 3100.0], start="2020-01-01")

    with pytest.raises(ValueError, match="没有足够的交集"):
        write_run_artifacts(make_result(), output_dir=tmp_path, benchmark_prices=benchmark)


def test_without_a_benchmark_the_metadata_says_so_and_the_csv_has_one_column(tmp_path):
    run_dir = write_run_artifacts(make_result(), output_dir=tmp_path)

    meta = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
    frame = pd.read_csv(run_dir / "equity.csv", index_col="date")

    assert meta["benchmark"] is None
    assert list(frame.columns) == ["equity"]


# --- 两张图（AC 5） ----------------------------------------------------------


def test_the_svgs_are_well_formed_xml(tmp_path):
    """可归档的最低要求：是合法的 XML 文档，文件名固定，可批量生成。"""
    run_dir = write_run_artifacts(
        make_result(), output_dir=tmp_path, benchmark_prices=curve([3000.0, 3100.0, 3200.0])
    )

    for name in ("equity.svg", "drawdown.svg"):
        root = ET.parse(run_dir / name).getroot()
        assert root.tag.endswith("svg")


def test_the_equity_chart_draws_one_line_per_series():
    """两条序列（策略与基准）→ 两条折线；没有基准时只有一条。"""
    equity = curve([100.0, 110.0, 120.0])
    benchmark = curve([3000.0, 3100.0, 3200.0])

    with_benchmark = render_equity_svg(equity, benchmark)
    without = render_equity_svg(equity)

    assert with_benchmark.count("<polyline") == 2
    assert without.count("<polyline") == 1


def test_the_drawdown_chart_plots_nothing_positive():
    """回撤图画**负值**：``Metrics.max_drawdown`` 报正值，只有图上取负。

    断言的是**方向**而不是「有高有低」：净值 [100, 120, 90] 的回撤序列是 ``0, 0, −25%``，
    而 SVG 的 y 向下增大，故最后一个点必须**最低**（y 最大）。若有人改成画正回撤，
    这个顺序会反过来，本断言即失败——「有高低差」那种断言是拦不住它的。
    """
    root = ET.fromstring(render_drawdown_svg(curve([100.0, 120.0, 90.0])))
    polyline = [el for el in root.iter() if el.tag.endswith("polyline")][0]
    ys = [float(pair.split(",")[1]) for pair in polyline.get("points").split()]

    assert len(ys) == 3
    assert ys[0] == pytest.approx(ys[1]), "前两点回撤都是 0，应等高"
    assert ys[2] > ys[0], "最深回撤没有画在更低处——方向反了（那会把正回撤画成回撤）"


def test_a_partially_covering_benchmark_is_not_stretched_across_the_chart():
    """横轴**按日期对齐**：只覆盖后两天的基准只占住那一段，不铺满整幅宽度。

    铺满会让人以为「全程都有基准」——而本模块在元数据里刻意不假装这件事。基准的线若被
    拉伸，它的第一个点会落在左边缘（x=64）；正确时它落在第 2 个刻度（x≈424）。
    """
    equity = curve([100.0, 110.0, 120.0])
    benchmark = curve([3000.0, 3100.0], start="2024-01-03")  # 只覆盖后两天

    root = ET.fromstring(render_equity_svg(equity, benchmark))
    lines = [el for el in root.iter() if el.tag.endswith("polyline")]

    assert len(lines) == 2
    benchmark_points = lines[1].get("points").split()
    assert len(benchmark_points) == 2

    first_x = float(benchmark_points[0].split(",")[0])
    assert first_x > 100, "基准被拉伸到整幅宽度了——它只覆盖了后两天"


def test_a_benchmark_covering_the_whole_period_spans_the_whole_width():
    """与上一条互补：全程覆盖时它确实从左边开始，故上一条测的是对齐而非「基准总是缩着」。"""
    equity = curve([100.0, 110.0, 120.0])

    root = ET.fromstring(render_equity_svg(equity, curve([3000.0, 3100.0, 3200.0])))
    lines = [el for el in root.iter() if el.tag.endswith("polyline")]
    points = lines[1].get("points").split()

    assert len(points) == 3
    assert float(points[0].split(",")[0]) == pytest.approx(64.0), "全程覆盖应从左边缘起画"


def test_chart_labels_with_xml_metacharacters_are_escaped():
    """SVG 是 XML：标题里的 ``&`` 不转义就不是合法文档，`ET.parse` 会当场失败。"""
    svg = render_equity_svg(curve([100.0, 110.0]))

    assert "&" not in svg.split("<text")[1].split("</text>")[0].replace("&amp;", "")


def test_a_flat_series_does_not_collapse_the_chart():
    """净值恒定（无波动）时坐标跨度为 0——必须撑开，否则所有点挤在一条线上且除零。"""
    svg = render_equity_svg(curve([100.0, 100.0, 100.0]))

    root = ET.fromstring(svg)
    assert root.tag.endswith("svg")


# --- 数据快照（AC 3 的「数据快照版本」） ------------------------------------


def test_the_snapshot_digest_changes_when_a_file_changes(tmp_path):
    """这是它存在的理由：通达信回补或修正历史后，摘要必须变，否则「复现」是假的。"""
    data_file = tmp_path / "sh600000.day"
    data_file.write_bytes(b"12345678")
    before = data_snapshot([data_file])["digest"]

    data_file.write_bytes(b"123456789")  # 回补了一条记录

    assert data_snapshot([data_file])["digest"] != before


def test_the_snapshot_digest_is_stable_and_order_independent(tmp_path):
    """同一批文件重复计算必须相同，且与传入顺序无关（摘要按路径排序）。"""
    first = tmp_path / "a.day"
    second = tmp_path / "b.day"
    first.write_bytes(b"a")
    second.write_bytes(b"b")

    one = data_snapshot([first, second])
    other = data_snapshot([second, first])

    assert one["digest"] == other["digest"]
    assert one["files"] == 2


def test_the_snapshot_manifest_is_readable_so_a_change_can_be_located(tmp_path):
    """摘要对不上时，得能看出是**哪个文件**变了——故清单本身要落进元数据。"""
    data_file = tmp_path / "sh600000.day"
    data_file.write_bytes(b"12345678")

    snapshot = data_snapshot([data_file])

    assert snapshot["manifest"][0]["path"].endswith("sh600000.day")
    assert snapshot["manifest"][0]["size"] == 8


def test_an_empty_snapshot_is_allowed_and_has_a_digest(tmp_path):
    """不传数据文件是合法的（例如用内存里的价格表），摘要仍有确定值。"""
    snapshot = data_snapshot([])

    assert snapshot["files"] == 0
    assert snapshot == data_snapshot([])


# --- 依据元数据复现（AC 4） ---------------------------------------------------


def test_a_run_can_be_replayed_from_its_metadata(tmp_path):
    """AC 4：从 ``run.json`` 取回策略类与关键参数，补上行情即可重跑。

    策略是可导入对象，故用它的路径就能取回；输入条件（资金、最大持仓、费用）原样翻回
    关键字参数。**不含** ``markets``（要重新取数）与 ``screen``（无法逐字恢复）。
    """
    from mbt.report import import_strategy, replay_arguments

    run_dir = write_run_artifacts(
        make_result(),
        output_dir=tmp_path,
        strategy=ExampleStrategy,
        strategy_params={"fast": 5},
        cash=100_000.0,
        max_positions=3,
        costs={"commission": 0.0003, "commission_mode": "all_in", "slippage": 0.002},
    )
    meta = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))

    assert import_strategy(meta) is ExampleStrategy
    assert meta["strategy"]["params"] == {"fast": 5}
    assert replay_arguments(meta) == {
        "cash": 100_000.0,
        "max_positions": 3,
        "commission": 0.0003,
        "commission_mode": "all_in",
        "slippage": 0.002,
    }


def test_replaying_reports_a_strategy_that_no_longer_exists():
    """策略被改名或删除时必须**报错**并指出是哪个路径，而不是返回空值让人继续跑。"""
    from mbt.report import import_strategy

    with pytest.raises(ValueError, match="没有 'NoSuchStrategy'"):
        import_strategy({"strategy": {"path": "tests.test_report.NoSuchStrategy"}})


def test_replaying_says_so_when_the_metadata_has_no_strategy():
    from mbt.report import import_strategy

    with pytest.raises(ValueError, match="没有策略导入路径"):
        import_strategy({})


def test_verifying_a_snapshot_detects_data_that_changed_since_the_run(tmp_path):
    """参数一样但数据变了，重跑出来的就不是当初那次回测——故摘要变了必须能发现。"""
    from mbt.report import verify_snapshot

    data_file = tmp_path / "sh600000.day"
    data_file.write_bytes(b"12345678")
    run_dir = write_run_artifacts(
        make_result(), output_dir=tmp_path / "out", snapshot_paths=[data_file]
    )
    meta = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))

    assert verify_snapshot(meta, [data_file]) == []

    data_file.write_bytes(b"1234567890")  # 通达信回补了一条记录

    problems = verify_snapshot(meta, [data_file])
    assert len(problems) == 1
    assert "数据快照已变化" in problems[0]


def test_verifying_without_a_digest_reports_rather_than_passing_silently(tmp_path):
    """没有摘要时不能当作「一致」——那会让人以为复现成功。"""
    from mbt.report import verify_snapshot

    assert verify_snapshot({}, []) != []
