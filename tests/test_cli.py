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


def make_dataroot(tmp_path, symbol="sh600000", periods=80, start_price=1000):
    """搭一个只含一个标的的最小数据源；价格恒定，故不会撞上涨跌停判定。

    日期用**交易日**（``bdate_range``）而非日历日——否则周六周日也成了交易日，
    「评估日不是交易日要报错」这条就测不出来了。

    权息文件直接**用仓库里的真实夹具**：它的字节是真的，故取数路径上一路走通（稀释判定、
    越界检查都要读它）。
    """
    import pandas as pd

    root = tmp_path / "vipdoc"
    index = pd.bdate_range("2024-01-02", periods=periods)
    records = [
        (int(f"{stamp:%Y%m%d}"), start_price, start_price, start_price, start_price, 0.0, 1000)
        for stamp in index
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
        output_dir=None,
        limit=None,
        symbols_file=None,
    )
    base.update(overrides)
    return type("Args", (), base)


# --- 参数解析 -----------------------------------------------------------------


def test_the_parser_has_two_subcommands():
    parser = build_parser()
    assert set(parser._subparsers._group_actions[0].choices) == {"backtest", "screen"}


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
