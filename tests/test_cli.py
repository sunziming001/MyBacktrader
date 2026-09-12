"""CLI：两条命令的**薄映射**与退出码（票据 #11）。

AC 明令「测试可以直接调库而不靠 shell 命令硬凑」，故这里**全部直接调
`run_backtest_command` / `run_screen_command`**——它们是纯函数，不收 `sys.argv`、不调
`sys.exit`、输出由参数注入。只有一条测试碰真正的进程入口，为的是验证退出码确实被传出去。

两条默认值是本票「一跑就对」的关键，各有一条专门钉住：`--start` 默认
`2015-08-01`（费用口径全可查的最早日期）、`--output-dir` **必填**。
"""

from __future__ import annotations

import io
import struct
from pathlib import Path

import pandas as pd
import pytest
from conftest import GBBQ_FIXTURE

from mbt.cli import (
    DEFAULT_START,
    MAX_FAILURE_RATE,
    build_parser,
    parse_params,
    run_backtest_command,
    run_screen_command,
)

EXAMPLE_STRATEGY = "examples.strategies:BuyAndHold"


def capture():
    """两个 ``StringIO`` 当作 stdout / stderr 注入。"""
    return io.StringIO(), io.StringIO()


def write_day(path: Path, records) -> None:
    """写一个 ``.day`` 文件；``records`` 是 ``(date, o, h, l, c, amount, volume)``。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    blob = bytearray()
    for date, o, h, low, c, amount, volume in records:
        blob += struct.pack("<IIIIIfII", date, o, h, low, c, amount, volume, 0)
    path.write_bytes(bytes(blob))


def make_dataroot(tmp_path, symbol="sh600000", periods=80, start_price=1000, closes=None):
    """搭一个只含一个标的的最小数据源；价格恒定，故不会撞上涨跌停判定。

    日期用**交易日**（``bdate_range``）而非日历日——否则周六周日也成了交易日，
    「评估日不是交易日要报错」这条就测不出来了。

    权息文件直接**用仓库里的真实夹具**：它的字节是真的，故取数路径上一路走通（稀释判定、
    越界检查都要读它）。

    ``closes`` 给出时用它当收盘价（单位：**分**，解析时会除以 ``PRICE_SCALE``），长度须与
    ``periods`` 一致。**恒定价是默认值，而恒定价下 ``close > SMA`` 恒为假**——任何靠穿越均线
    入场的策略都会一笔不成交。故要测那类策略必须传 ``closes``，否则测了等于没测。
    """
    import pandas as pd

    root = tmp_path / "vipdoc"
    index = pd.bdate_range("2024-01-02", periods=periods)
    if closes is None:
        closes = [start_price] * periods
    assert len(closes) == periods, f"closes（{len(closes)} 个）须与 periods（{periods}）一致"

    # 每根 K 线都写成 o = h = l = c：既自洽（不撞「开盘/收盘越出高低区间」的质检），
    # 又让收盘价序列就是我们要的那条路径。
    records = [
        (int(f"{stamp:%Y%m%d}"), price, price, price, price, 0.0, 1000)
        for stamp, price in zip(index, closes, strict=False)
    ]
    write_day(root / symbol[:2] / "lday" / f"{symbol}.day", records)

    gbbq = tmp_path / "gbbq"
    gbbq.write_bytes(GBBQ_FIXTURE.read_bytes())
    return root, gbbq


def make_args(**overrides):
    """一个能跑通的参数集合，测试只覆盖关心的那几项。"""
    base = dict(
        command="backtest",
        strategy=EXAMPLE_STRATEGY,
        start=DEFAULT_START,
        end=None,
        cash=100_000.0,
        max_positions=None,
        commission=0.0,
        commission_mode=None,
        commission_min=0.0,
        slippage=0.0,
        benchmark="sh000300",
        tdx_root=None,
        gbbq=None,
        cw_root=None,
        non_loss=False,
        master=None,
        output_dir=None,
        limit=None,
        symbols_file=None,
        param=[],
    )
    base.update(overrides)
    return type("Args", (), base)


def screen_args(**overrides):
    """选股命令的参数集合；``as_of`` 默认取数据里的一个交易日。"""
    base = dict(
        command="screen",
        as_of="2024-03-01",  # 数据里确有的一个交易日（60 个交易日约到 3 月下旬）
        top_n=2,
        tdx_root=None,
        gbbq=None,
        cw_root=None,
        non_loss=False,
        master=None,
        output_dir=None,
        limit=None,
        symbols_file=None,
    )
    base.update(overrides)
    return type("Args", (), base)


# --- 参数解析 -----------------------------------------------------------------


def test_the_parser_has_three_subcommands():
    """三条命令：回测、选股、更新检查。"""
    parser = build_parser()
    assert set(parser._subparsers._group_actions[0].choices) == {"backtest", "screen", "update"}


def test_start_defaults_to_the_fully_priceable_window():
    """**本票最关键的一条默认值**：默认区间取「跑得对」的那个窗口。

    实测 37% 的股票数据起于 1997–2014，那段区间涨跌幅查得到、**费用查不到**，一成交就报
    RuleTableError。故默认不能是「全历史」。
    """
    args = build_parser().parse_args(
        [
            "backtest",
            "--strategy",
            EXAMPLE_STRATEGY,
            "--tdx-root",
            "R",
            "--gbbq",
            "G",
            "--output-dir",
            "O",
        ]
    )

    assert args.start == "2015-08-01"
    assert args.start == DEFAULT_START


def test_output_dir_is_required():
    """不给默认落盘位置——CLI 可能在任意工作目录下被执行，偷偷建目录不合适。"""
    with pytest.raises(SystemExit) as info:
        build_parser().parse_args(
            ["backtest", "--strategy", EXAMPLE_STRATEGY, "--tdx-root", "R", "--gbbq", "G"]
        )
    assert info.value.code == 2, "argparse 的用法错误退出码是 2"


def test_a_missing_required_argument_exits_with_two():
    with pytest.raises(SystemExit) as info:
        build_parser().parse_args(["backtest"])
    assert info.value.code == 2


# --- --param 的字面量解析 ------------------------------------------------------


def test_params_are_parsed_as_literals():
    got = parse_params(["fast=5", "slow=20.5", "flag=true", "name=ma"])

    assert got == {"fast": 5, "slow": 20.5, "flag": True, "name": "ma"}


def test_a_malformed_param_names_the_offending_item():
    """策略参数错一个，结果就完全是另一回事——报错必须指出是哪一个。"""
    with pytest.raises(ValueError, match="slow"):
        parse_params(["fast=5", "slow"])


# --- 策略的导入路径 ------------------------------------------------------------


def test_an_unimportable_strategy_module_is_reported():
    args = make_args(strategy="no.such.module:Thing")
    out, err = capture()

    code = run_backtest_command(args, stdout=out, stderr=err)

    assert code == 1
    assert "no.such.module" in err.getvalue()


def test_a_missing_class_in_an_importable_module_is_reported():
    args = make_args(strategy="examples.strategies:NotAStrategy")
    out, err = capture()

    code = run_backtest_command(args, stdout=out, stderr=err)

    assert code == 1
    assert "NotAStrategy" in err.getvalue()


def test_a_strategy_path_without_a_colon_is_reported():
    args = make_args(strategy="examples.strategies.BuyAndHold")
    out, err = capture()

    assert run_backtest_command(args, stdout=out, stderr=err) == 1
    assert "mypkg.strategies:BuyAndHold" in err.getvalue()


# --- 端到端：回测 --------------------------------------------------------------


def test_a_backtest_run_writes_artifacts_and_prints_a_summary(tmp_path):
    root, gbbq = make_dataroot(tmp_path)
    args = make_args(tdx_root=str(root), gbbq=str(gbbq), output_dir=str(tmp_path / "runs"))
    out, err = capture()

    code = run_backtest_command(args, stdout=out, stderr=err)

    assert code == 0, err.getvalue()
    assert "结果：" in out.getvalue()
    assert "最大回撤" in out.getvalue()

    run_dirs = list((tmp_path / "runs").iterdir())
    assert len(run_dirs) == 1
    assert {p.name for p in run_dirs[0].iterdir()} >= {
        "run.json",
        "metrics.json",
        "equity.csv",
        "equity.svg",
        "drawdown.svg",
    }


def test_the_backtest_artifacts_record_how_to_replay_it(tmp_path):
    """命令行指名的策略与产物里记的策略必须**同一种写法**，否则复现时要再翻译一次。"""
    import json

    root, gbbq = make_dataroot(tmp_path)
    args = make_args(
        tdx_root=str(root),
        gbbq=str(gbbq),
        output_dir=str(tmp_path / "runs"),
        param=["size=200"],
    )
    out, err = capture()

    assert run_backtest_command(args, stdout=out, stderr=err) == 0, err.getvalue()

    run_dir = next((tmp_path / "runs").iterdir())
    meta = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
    assert meta["strategy"]["path"].endswith("BuyAndHold")
    assert meta["strategy"]["params"] == {"size": 200}


def test_symbols_that_cannot_be_loaded_are_skipped_with_a_reason(tmp_path):
    """跳过必须被**说出来**，且汇总里能按原因分类——静默跳过会让「跳过了什么」变成谜。"""
    root, gbbq = make_dataroot(tmp_path, symbol="sh600000")
    # 再加一个「其他」品种的文件（001xxx 之类无法判定，会被品种过滤挡下）
    write_day(
        root / "sh" / "lday" / "sh999999.day",
        [(20240102, 1000, 1000, 1000, 1000, 0.0, 1000)],
    )
    args = make_args(tdx_root=str(root), gbbq=str(gbbq), output_dir=str(tmp_path / "runs"))
    out, err = capture()

    code = run_backtest_command(args, stdout=out, stderr=err)

    assert code == 0, err.getvalue()
    assert "跳过" in out.getvalue()


def test_a_run_where_everything_is_skipped_fails_rather_than_reporting_success(tmp_path):
    """「几乎全跳过」不该被脚本当成成功——否则 0 退出码会被读成「跑得好好的」。"""
    root = tmp_path / "vipdoc"
    root.mkdir(parents=True)
    (tmp_path / "gbbq").write_bytes(b"")
    args = make_args(
        tdx_root=str(root),
        gbbq=str(tmp_path / "gbbq"),
        output_dir=str(tmp_path / "runs"),
        symbols_file=None,
    )
    out, err = capture()

    code = run_backtest_command(args, stdout=out, stderr=err)

    assert code == 1
    assert "选出的标的为空" in err.getvalue()


def test_the_run_honours_the_requested_range(tmp_path):
    """``--start`` / ``--end`` 必须真的**生效**。

    这条钉的是一个真缺陷：两个参数原先被解析了却从未使用，回测因此永远跑全历史——而
    README 还把 ``--start`` 的默认值当作生效的行为写了理由。**文档描述不存在的行为是最糟的
    一类缺陷**，故这里直接断言净值曲线的区间。
    """
    root, gbbq = make_dataroot(tmp_path, periods=120)
    args = make_args(
        tdx_root=str(root),
        gbbq=str(gbbq),
        output_dir=str(tmp_path / "runs"),
        start="2024-02-01",
        end="2024-03-15",
    )
    out, err = capture()

    assert run_backtest_command(args, stdout=out, stderr=err) == 0, err.getvalue()

    import json

    run_dir = next((tmp_path / "runs").iterdir())
    period = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))["period"]
    assert period["start"] >= "2024-02-01"
    assert period["end"] <= "2024-03-15"
    assert period["trading_days"] < 120, "区间没有被截断——参数没生效"


def test_a_malformed_start_is_reported_not_ignored(tmp_path):
    """日期写错要**报错**，而不是被静默忽略后返回 0——脚本据此判断成败。"""
    root, gbbq = make_dataroot(tmp_path)
    args = make_args(
        tdx_root=str(root),
        gbbq=str(gbbq),
        output_dir=str(tmp_path / "runs"),
        start="not-a-date",
    )
    out, err = capture()

    assert run_backtest_command(args, stdout=out, stderr=err) == 1
    assert "不是合法日期" in err.getvalue()


def test_a_reversed_range_is_reported(tmp_path):
    """起始日晚于结束日要报错——区间为空不是「跑出空结果」，是参数写错了。"""
    root, gbbq = make_dataroot(tmp_path)
    args = make_args(
        tdx_root=str(root),
        gbbq=str(gbbq),
        output_dir=str(tmp_path / "runs"),
        start="2024-06-01",
        end="2024-01-01",
    )
    out, err = capture()

    assert run_backtest_command(args, stdout=out, stderr=err) == 1
    assert "区间为空" in err.getvalue()


def test_commission_without_a_mode_is_reported_by_the_portfolio_entry(tmp_path):
    """组合入口也要守卫 ``commission_mode``——原先它只在单标的入口里。

    实跑 CLI 撞到过：`--commission` 给了、`--commission-mode` 没给，成本口径被**静默**
    退回 ``all_in``，而两种口径算出的成本方向相反地错。
    """
    root, gbbq = make_dataroot(tmp_path)
    args = make_args(
        tdx_root=str(root),
        gbbq=str(gbbq),
        output_dir=str(tmp_path / "runs"),
        commission=0.0003,
        commission_mode=None,
    )
    out, err = capture()

    assert run_backtest_command(args, stdout=out, stderr=err) == 1
    assert "commission_mode" in err.getvalue()


def test_the_entry_point_makes_the_working_directory_importable(tmp_path, monkeypatch):
    """真正的进程入口要能把**当前工作目录**上的策略导入进来。

    安装后的入口脚本只会把「脚本所在目录」加进 ``sys.path``，不会加当前目录——于是
    `--strategy mypkg.strategies:BuyAndHold` 在用户自己的项目目录里反而导入不到，而
    「一条命令跑回测」这条 AC 就落不了地。

    这条是**唯一**碰真入口的测试：别的都直接调纯函数。
    """
    import sys

    from mbt.cli import main

    package = tmp_path / "myproj"
    package.mkdir()
    (package / "mystrat.py").write_text(
        "import backtrader as bt\n\n\nclass Noop(bt.Strategy):\n    pass\n",
        encoding="utf-8",
    )
    monkeypatch.chdir(package)
    monkeypatch.setattr(sys, "path", [p for p in sys.path if str(package) not in p])

    # 参数故意给不存在的路径：只要策略能**导入**，就不会在导入那一步失败——
    # 故这里断言的是「错误信息不是导入错误」。
    parser_args = [
        "backtest",
        "--strategy",
        "mystrat:Noop",
        "--tdx-root",
        str(tmp_path / "nope"),
        "--gbbq",
        str(tmp_path / "nope2"),
        "--output-dir",
        str(tmp_path / "out"),
    ]
    err = io.StringIO()
    import contextlib

    with contextlib.redirect_stderr(err):
        code = main(parser_args)

    assert code == 1
    assert "导入" not in err.getvalue(), f"策略应当可导入，却报了导入错误：{err.getvalue()}"


def test_the_failure_threshold_is_used(tmp_path):
    """阈值本身要是个明确常数，且与本测试一致——它是「还能不能采信」的分界。"""
    assert 0 < MAX_FAILURE_RATE < 1


def test_a_symbols_file_restricts_the_run(tmp_path):
    root, gbbq = make_dataroot(tmp_path, symbol="sh600000")
    listing = tmp_path / "list.txt"
    listing.write_text("# 我的自选\nsh600000\n", encoding="utf-8")
    args = make_args(
        tdx_root=str(root),
        gbbq=str(gbbq),
        output_dir=str(tmp_path / "runs"),
        symbols_file=str(listing),
    )
    out, err = capture()

    assert run_backtest_command(args, stdout=out, stderr=err) == 0, err.getvalue()
    assert "1 个股票候选" in out.getvalue()


def test_a_missing_symbols_file_is_reported(tmp_path):
    root, gbbq = make_dataroot(tmp_path)
    args = make_args(
        tdx_root=str(root),
        gbbq=str(gbbq),
        output_dir=str(tmp_path / "runs"),
        symbols_file=str(tmp_path / "nope.txt"),
    )
    out, err = capture()

    assert run_backtest_command(args, stdout=out, stderr=err) == 1
    assert "nope.txt" in err.getvalue()


def test_an_unreadable_benchmark_warns_but_does_not_fail_the_run(tmp_path):
    """没有基准的回测仍可解读，只是少了参照——故警告并继续，而不是中断。"""
    root, gbbq = make_dataroot(tmp_path)
    args = make_args(
        tdx_root=str(root),
        gbbq=str(gbbq),
        output_dir=str(tmp_path / "runs"),
        benchmark="sh000300",  # 数据源里没有它
    )
    out, err = capture()

    code = run_backtest_command(args, stdout=out, stderr=err)

    assert code == 0, err.getvalue()
    assert "警告" in err.getvalue()


# --- 端到端：选股 --------------------------------------------------------------


def test_a_screen_run_writes_a_ranked_candidate_list(tmp_path):
    root, gbbq = make_dataroot(tmp_path, periods=60)
    args = screen_args(
        tdx_root=str(root),
        gbbq=str(gbbq),
        output_dir=str(tmp_path / "screens"),
    )
    out, err = capture()

    code = run_screen_command(args, stdout=out, stderr=err)

    assert code == 0, err.getvalue()
    assert "候选集" in out.getvalue()
    run_dir = next((tmp_path / "screens").iterdir())
    assert (run_dir / "candidates.csv").is_file()
    assert (run_dir / "run.json").is_file()


def test_non_loss_requires_a_cw_root(tmp_path):
    """`--non-loss` 不给 `--cw-root` 要报错——而不是静默忽略那个开关。"""
    root, gbbq = make_dataroot(tmp_path)
    args = make_args(
        tdx_root=str(root),
        gbbq=str(gbbq),
        output_dir=str(tmp_path / "runs"),
        non_loss=True,
        cw_root=None,
    )
    out, err = capture()

    assert run_backtest_command(args, stdout=out, stderr=err) == 1
    assert "cw-root" in err.getvalue()


def test_non_loss_excludes_symbols_without_usable_financials(tmp_path):
    """启用「非亏损」后，fixture 里没有财务数据的标的应被**排除**，而不是照旧放行。

    这条同时验证「过滤真的接进了股票池」——若没接上，两只都会成交。
    """
    root, gbbq = make_dataroot(tmp_path, symbol="sh600000", periods=80)
    # 再加一只 fixture 里没有财务数据的标的
    write_day(
        root / "sz" / "lday" / "sz000001.day",
        [
            (int(f"{stamp:%Y%m%d}"), 1000, 1000, 1000, 1000, 0.0, 1000)
            for stamp in pd.bdate_range("2024-01-02", periods=80)
        ],
    )
    cw_root = Path(__file__).parent / "fixtures" / "cw"
    args = make_args(
        tdx_root=str(root),
        gbbq=str(gbbq),
        output_dir=str(tmp_path / "runs"),
        start=None,
        end=None,
        non_loss=True,
        cw_root=str(cw_root),
    )
    out, err = capture()

    assert run_backtest_command(args, stdout=out, stderr=err) == 0, err.getvalue()
    assert "基本面过滤" in out.getvalue()
    # fixture 里的 600000 有 2025 年报（公告日 2026-03-31），而数据是 2024 年 → 区间内没有
    # 可用财报，故两只都被排除；这正是「宁可少收」的方向。
    assert "0/2" in out.getvalue()


def test_the_extra_mask_is_loaded_from_the_cw_directory(tmp_path):
    """`--cw-root` 指错时要报错并指出问题，而不是静默当成「没有财务数据」。"""
    root, gbbq = make_dataroot(tmp_path)
    args = make_args(
        tdx_root=str(root),
        gbbq=str(gbbq),
        output_dir=str(tmp_path / "runs"),
        non_loss=True,
        cw_root=str(tmp_path / "没有这个目录"),
    )
    out, err = capture()

    assert run_backtest_command(args, stdout=out, stderr=err) == 1
    assert "财务数据读取失败" in err.getvalue()


# --- update 子命令 -----------------------------------------------------------


def update_args(**overrides):
    base = dict(
        command="update",
        tdx_root=None,
        baseline=None,
        limit=None,
        symbols_file=None,
        save_baseline=None,
    )
    base.update(overrides)
    return type("Args", (), base)


def test_the_update_command_builds_a_baseline_and_saves_it(tmp_path):
    """首次运行建立基线，退出码 `0`（**首次没有可比对象不是错误**）。"""
    from mbt.cli import run_update_command

    root, _ = make_dataroot(tmp_path)
    baseline = tmp_path / "boundaries.csv"
    args = update_args(tdx_root=str(root), save_baseline=str(baseline))
    out, err = capture()

    code = run_update_command(args, stdout=out, stderr=err)

    assert code == 0, err.getvalue()
    assert "首次运行" in out.getvalue()
    assert baseline.is_file()


def test_a_symbol_that_vanished_from_the_source_is_reported(tmp_path):
    """基线里有、这次取不到的标的 → **已消失**，并让退出码为 1。

    它同样让旧结果作废（样本少了一只），且必须报出来——否则会静默地从基线里消失。
    """
    from mbt.data.updates import MISSING, SymbolBoundary, check_updates

    root, _ = make_dataroot(tmp_path)
    ghost = SymbolBoundary(
        symbol="sz000999",
        first_date=pd.Timestamp("2024-01-02").date(),
        last_date=pd.Timestamp("2024-03-01").date(),
        bars=40,
        tail_digest="whatever",
    )

    report = check_updates(["sh600000"], tdx_root=root, previous={"sz000999": ghost})

    assert report.by_kind.get(MISSING) == ["sz000999"]
    assert report.needs_rerun is True
    assert "sz000999" in report.revised


def test_an_unreadable_symbol_is_reported_not_silently_dropped(tmp_path):
    """取不到数据 ≠ 没变化：**必须报出来**，否则「上次查过、这次没了」会无人知晓。"""
    from mbt.data.updates import check_updates

    root, _ = make_dataroot(tmp_path)
    # 造一个「文件存在但内容坏掉」的标的：长度不是记录长度的整数倍
    broken = root / "sz" / "lday" / "sz000002.day"
    broken.parent.mkdir(parents=True, exist_ok=True)
    broken.write_bytes(b"\x00" * 7)

    report = check_updates(["sh600000", "sz000002"], tdx_root=root)

    assert report.unreadable == ("sz000002",)
    assert "sz000002" not in report.boundaries


def test_the_update_command_reports_a_revision_with_exit_code_one(tmp_path):
    """**有回补/修正 → 退出码 1**：那意味着上次的回测结果作废了，脚本应当知道。"""
    from mbt.cli import run_update_command

    root, _ = make_dataroot(tmp_path)
    baseline = tmp_path / "boundaries.csv"
    run_update_command(
        update_args(tdx_root=str(root), save_baseline=str(baseline)),
        stdout=capture()[0],
        stderr=capture()[1],
    )

    # 改掉倒数第 2 根 K 线的四个价格（只改收盘会造出自洽性错误而被拒绝）
    path = root / "sh" / "lday" / "sh600000.day"
    raw = bytearray(path.read_bytes())
    import struct as _struct

    offset = len(raw) - 2 * 32
    date, o, h, low, close, amount, volume, reserved = _struct.unpack_from("<IIIIIfII", raw, offset)
    _struct.pack_into("<IIIIIfII", raw, offset, date, 999, 999, 999, 999, amount, volume, reserved)
    path.write_bytes(bytes(raw))

    args = update_args(tdx_root=str(root), baseline=str(baseline))
    out, err = capture()
    code = run_update_command(args, stdout=out, stderr=err)

    assert code == 1, err.getvalue()
    assert "回补/修正" in out.getvalue()
    assert "不再可信" in out.getvalue()


def test_the_update_command_reports_an_unchanged_baseline_with_exit_code_zero(tmp_path):
    from mbt.cli import run_update_command

    root, _ = make_dataroot(tmp_path)
    baseline = tmp_path / "boundaries.csv"
    run_update_command(
        update_args(tdx_root=str(root), save_baseline=str(baseline)),
        stdout=capture()[0],
        stderr=capture()[1],
    )

    out, err = capture()
    code = run_update_command(
        update_args(tdx_root=str(root), baseline=str(baseline)), stdout=out, stderr=err
    )

    assert code == 0, err.getvalue()
    assert "无变化" in out.getvalue()


def test_the_update_command_reports_tail_gaps_and_flags_them_undecidable(tmp_path):
    """尾部空缺要报出来，并**显式声明停牌与退市不可区分**——不猜。"""
    from mbt.cli import run_update_command

    root, gbbq = make_dataroot(tmp_path, symbol="sh600000", periods=80)
    shorter = pd.bdate_range("2024-01-02", periods=40)
    write_day(
        root / "sz" / "lday" / "sz000001.day",
        [(int(f"{stamp:%Y%m%d}"), 1000, 1000, 1000, 1000, 0.0, 1000) for stamp in shorter],
    )
    args = update_args(tdx_root=str(root))
    out, err = capture()

    code = run_update_command(args, stdout=out, stderr=err)

    assert code == 0, err.getvalue()
    assert "尾部空缺" in out.getvalue()
    assert "停牌还是退市" in out.getvalue(), "必须显式声明停牌与退市不可区分"


def test_all_user_facing_output_is_gbk_encodable(tmp_path):
    """**CLI 的输出必须能在中文 Windows 控制台上打印。**

    这条是实跑真实数据时撞出来的：输出里用了 ``⚠️``，而中文 Windows 控制台是 GBK，
    打印时直接抛 ``UnicodeEncodeError: 'gbk' codec can't encode character``——**命令整体失败**，
    而那与「数据有问题」完全无关，排查起来很费劲。

    判据是「把每一行输出按 GBK 编码时不抛错」。用真流做，比逐字符串断言更接近实际。
    """
    from mbt.cli import run_update_command

    root, _ = make_dataroot(tmp_path)
    args = update_args(tdx_root=str(root), save_baseline=str(tmp_path / "b.csv"))

    out, err = capture()
    code = run_update_command(args, stdout=out, stderr=err)
    assert code == 0, err.getvalue()

    for line in (out.getvalue() + err.getvalue()).splitlines():
        line.encode("gbk")  # 抛 UnicodeEncodeError 即失败


def test_the_update_command_rejects_a_missing_baseline_file(tmp_path):
    """给了 `--baseline` 却指了个不存在的文件要**报错**，不静默当成首次。"""
    from mbt.cli import run_update_command

    root, _ = make_dataroot(tmp_path)
    args = update_args(tdx_root=str(root), baseline=str(tmp_path / "nope.csv"))
    out, err = capture()

    assert run_update_command(args, stdout=out, stderr=err) == 1
    assert "边界清单不存在" in err.getvalue()


def test_a_degraded_calendar_is_warned_about(tmp_path):
    """`--limit` / `--symbols-file` 会削弱判据（市场日历不完整），故必须**警告**。

    不警告的话，「区间内缺口看不见、尾部空缺偏小」会被当成「数据没问题」。
    """
    from mbt.cli import run_update_command

    root, _ = make_dataroot(tmp_path)
    args = update_args(tdx_root=str(root), limit=1)
    out, err = capture()

    run_update_command(args, stdout=out, stderr=err)

    assert "警告" in err.getvalue()
    assert "市场日历" in err.getvalue()


def test_the_master_report_counts_only_the_symbols_in_this_run(tmp_path):
    """`--master` 的汇总只能数**本次用到**的标的。

    实跑撞到过：`load_listing_dates` 返回整表（本机 8,088 条），直接拿它报数会印出
    「8088/53 个标的有真实上市日」——一个无意义的数字，且会让人以为数据有问题。
    """
    from mbt.cli import run_backtest_command

    root, gbbq = make_dataroot(tmp_path, periods=80)
    master_fixture = Path(__file__).parent / "fixtures" / "master" / "base.dbf"
    args = make_args(
        tdx_root=str(root),
        gbbq=str(gbbq),
        output_dir=str(tmp_path / "runs"),
        master=str(master_fixture),
    )
    out, err = capture()

    assert run_backtest_command(args, stdout=out, stderr=err) == 0, err.getvalue()
    line = next(line for line in out.getvalue().splitlines() if "证券主表" in line)
    # fixture 里只有 000001/600000/600519/160605 四条；本次只有一个标的（sh600000）。
    assert "1/1" in line, f"应只数本次的标的，而实际是：{line}"


def test_without_a_master_the_cli_says_which_rule_it_used(tmp_path):
    """不给主表时**必须说明用的是近似口径**——否则「用的哪个口径」成了谜。"""
    from mbt.cli import run_backtest_command

    root, gbbq = make_dataroot(tmp_path, periods=80)
    args = make_args(tdx_root=str(root), gbbq=str(gbbq), output_dir=str(tmp_path / "runs"))
    out, err = capture()

    assert run_backtest_command(args, stdout=out, stderr=err) == 0, err.getvalue()
    assert "近似口径" in out.getvalue()


def test_the_screen_command_says_which_rule_it_used(tmp_path):
    """选股命令也要说明口径——否则「用的哪个口径」成了谜（原先只有 backtest 会打印）。"""
    from mbt.cli import run_screen_command

    root, gbbq = make_dataroot(tmp_path, periods=60)
    args = screen_args(tdx_root=str(root), gbbq=str(gbbq), output_dir=str(tmp_path / "screens"))
    out, err = capture()

    assert run_screen_command(args, stdout=out, stderr=err) == 0, err.getvalue()
    assert "近似口径" in out.getvalue()


def test_a_malformed_as_of_is_reported(tmp_path):
    root, gbbq = make_dataroot(tmp_path)
    args = screen_args(
        tdx_root=str(root), gbbq=str(gbbq), output_dir=str(tmp_path / "screens"), as_of="2026-13-99"
    )
    out, err = capture()

    assert run_screen_command(args, stdout=out, stderr=err) == 1
    assert "不是合法日期" in err.getvalue()


def test_an_as_of_that_is_not_a_trading_day_is_reported(tmp_path):
    """评估日必须是交易日——不替你取最近的交易日（与 Screen 的既定行为一致）。"""
    root, gbbq = make_dataroot(tmp_path, periods=60)
    args = screen_args(
        tdx_root=str(root),
        gbbq=str(gbbq),
        output_dir=str(tmp_path / "screens"),
        as_of="2024-03-31",  # 周日，不是交易日
    )
    out, err = capture()

    assert run_screen_command(args, stdout=out, stderr=err) == 1
    assert "交易日" in err.getvalue()


# --- 拒单的可见性 -------------------------------------------------------------


def test_the_cli_warns_when_orders_were_placed_but_none_filled(tmp_path):
    """有下单却**一笔都没成交**时必须告警，并把拒单明细落盘。

    这是本项目最防的那类静默失败：净值曲线是一条平线、指标全是零，看着像个「本来就没信号」
    的正常结果。构造：窗口只有 30 根，而股票池门槛是 60 个交易日 → **每一笔**买单都以
    「不在股票池」被拒 → 零成交。

    刻意**不**按「拒单率超阈值」告警——名额有限时高拒单率是**正常**的（34 个信号、10 个名额，
    24 笔被拒是预期行为），真正没有意义的是「零成交」。
    """
    import json

    root, gbbq = make_dataroot(tmp_path, periods=30)  # 不足门槛 60
    args = make_args(tdx_root=str(root), gbbq=str(gbbq), output_dir=str(tmp_path / "runs"))
    out, err = capture()

    assert run_backtest_command(args, stdout=out, stderr=err) == 0, err.getvalue()
    text = out.getvalue()
    assert "拒单" in text
    assert "一笔都没成交" in text, "有下单却零成交，必须告警"

    run_dir = next((tmp_path / "runs").iterdir())
    metrics = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
    assert metrics["rejected_orders"] > 0
    assert metrics["rejection_rate"] == 1.0, "一笔都没成交，故拒单率是 100%"

    rejected = pd.read_csv(run_dir / "rejected.csv")
    assert set(rejected["status"]) == {"Rejected"}
    assert rejected["reason"].str.contains("不在股票池").all(), "理由要能看出为什么被拒"


# --- 示例策略（examples/strategies.py） ---------------------------------------


def test_the_ma_cross_example_both_buys_and_sells(tmp_path):
    """均线上穿的示例必须**真的买、也真的卖**。

    这条不是形式主义。模板的第一版把「下穿卖出」写在了**无持仓**的分支里（前面有个
    `continue`），于是**永远不卖**——而产物只把它记成「没有平仓交易」，**不报错**。
    手工跑一遍才发现。这条测试就是那次手工发现的固化：买卖任一为 0 就失败。

    它一度标着 ``xfail(strict=True)``，因为那时它撞到了**库的**两个独立缺陷，而错都在库、
    不在示例：sizer 不给交易费用留余量（#29 修掉），以及定量用的是当根收盘价、而订单在
    **下一根**成交（#32 修掉）。两处修完后它转绿，标记随之摘除——这正是当初用
    ``strict=True`` 的目的：不修就一直是红的，修好了会 XPASS 主动提醒。
    """
    import json
    import math

    # ±10% 的正弦路径，周期 40 根：既会反复穿越均线（有买有卖），
    # 单日变动约 1.6%，远在涨跌停带内，不会撞上异常判定。
    closes = [int(1000 * (1 + 0.10 * math.sin(2 * math.pi * i / 40))) for i in range(200)]
    root, gbbq = make_dataroot(tmp_path, periods=len(closes), closes=closes)
    args = make_args(
        strategy="examples.strategies:CloseCrossesAboveMA",
        tdx_root=str(root),
        gbbq=str(gbbq),
        output_dir=str(tmp_path / "runs"),
        param=["n=5"],
    )
    out, err = capture()

    assert run_backtest_command(args, stdout=out, stderr=err) == 0, err.getvalue()

    run_dir = next((tmp_path / "runs").iterdir())
    metrics = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
    assert metrics["closed_trades"] > 0, "示例策略一笔都没平仓——卖出那一支多半又没走到"
    # 有平仓就意味着有过买入；若买入为 0 则不可能有平仓，故两者都已被覆盖。
    assert metrics["win_rate"] is not None and metrics["payoff_ratio"] is not None


def test_the_ma_cross_example_on_a_flat_price_series_never_trades(tmp_path):
    """反过来钉住夹具的**默认**价格是恒定的。

    恒定价下 ``close > SMA`` 恒为假（相等不满足严格大于），故穿越类策略一笔不成交。
    这条把这件事写成断言，免得后人以为「夹具能测策略」——那不成立，会白测一场。
    """
    import json

    root, gbbq = make_dataroot(tmp_path, periods=200)  # 不传 closes → 恒定 1000 分
    args = make_args(
        strategy="examples.strategies:CloseCrossesAboveMA",
        tdx_root=str(root),
        gbbq=str(gbbq),
        output_dir=str(tmp_path / "runs"),
        param=["n=5"],
    )
    out, err = capture()

    assert run_backtest_command(args, stdout=out, stderr=err) == 0, err.getvalue()

    run_dir = next((tmp_path / "runs").iterdir())
    metrics = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
    assert metrics["closed_trades"] == 0
    assert "没有平仓交易" in out.getvalue(), "产物与摘要都该如实说明没有平仓"
