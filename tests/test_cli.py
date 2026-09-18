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
    ALL_CANDIDATES,
    DEFAULT_START,
    MAX_FAILURE_RATE,
    PANEL_WARMUP_BARS,
    _resolve_as_of,
    _write_watchlist,
    build_parser,
    parse_params,
    run_backtest_command,
    run_screen_command,
)
from mbt.data import Panel

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


def make_dataroot(
    tmp_path, symbol="sh600000", periods=80, start_price=1000, closes=None, start="2024-01-02"
):
    """搭一个只含一个标的的最小数据源；价格恒定，故不会撞上涨跌停判定。

    日期用**交易日**（``bdate_range``）而非日历日——否则周六周日也成了交易日，
    「评估日不是交易日要报错」这条就测不出来了。

    ``start`` 是首根 K 线的日期。需要把行情挪到**某个年份**时给（前瞻那条路要求评估日的年份
    与该标的的一致预期财年相同，见 :func:`mbt.data.forward.uses_forward`，而这里的一致预期
    夹具指的是 2026 年）。

    权息文件直接**用仓库里的真实夹具**：它的字节是真的，故取数路径上一路走通（稀释判定、
    越界检查都要读它）。

    ``closes`` 给出时用它当收盘价（单位：**分**，解析时会除以 ``PRICE_SCALE``），长度须与
    ``periods`` 一致。**恒定价是默认值，而恒定价下 ``close > SMA`` 恒为假**——任何靠穿越均线
    入场的策略都会一笔不成交。故要测那类策略必须传 ``closes``，否则测了等于没测。
    """
    import pandas as pd

    root = tmp_path / "vipdoc"
    index = pd.bdate_range(start, periods=periods)
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
        watchlist_dir=None,
        limit=None,
        symbols_file=None,
        forward_root=None,
    )
    base.update(overrides)
    return type("Args", (), base)


# --- 参数解析 -----------------------------------------------------------------


def test_the_parser_has_three_subcommands():
    """三条命令：回测、选股、更新检查。"""
    parser = build_parser()
    assert set(parser._subparsers._group_actions[0].choices) == {"backtest", "screen", "update"}


def test_both_backtest_and_screen_accept_the_same_screen_switch():
    """`backtest` 与 `screen` 都收 ``--screen``——否则「同一条件只写一遍」在选股侧不成立。

    此前 ``screen`` 写死 ``momentum_screen``，于是 ``backtest --screen valuation`` 用估值规则
    而 ``screen`` 仍按动量选，两边给出的候选完全不是一回事。这条把两者的**开关一致性**钉住。
    """
    parser = build_parser()
    for command in ("backtest", "screen"):
        args = parser.parse_args(
            [command, "--tdx-root", "x", "--gbbq", "y", "--output-dir", "z"]
            + (["--strategy", "m:S"] if command == "backtest" else ["--as-of", "2024-01-02"])
        )
        assert hasattr(args, "screen"), f"{command} 应当接受 --screen"
        assert hasattr(args, "boards"), f"{command} 应当接受 --boards"


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


def test_the_backtest_artifact_records_the_screen_it_used(tmp_path):
    """回测产物也要记下**入场闸门是哪条规则**。

    ``write_run_artifacts`` 一直收 ``screen`` / ``screen_label`` 两个参数，而回测那条调用
    没传——于是 ``run.json`` 里恒为 ``"screen": null``：八条过滤器、排序因子、权重一条都
    没留下，而提示语还写着「此处只记其名称」（那是在 ``describe_screen`` 里写的，从没被
    回测调用过）。
    """
    import json

    root, gbbq = make_dataroot(tmp_path, periods=80)
    args = make_args(
        screen="momentum",
        tdx_root=str(root),
        gbbq=str(gbbq),
        output_dir=str(tmp_path / "runs"),
    )
    out, err = capture()

    assert run_backtest_command(args, stdout=out, stderr=err) == 0, err.getvalue()

    run_dir = next((tmp_path / "runs").iterdir())
    screen = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))["screen"]

    assert screen["label"] == "momentum"
    assert screen["top_n"] == 5  # 没给 --top-n 时，候选数默认取「取前 N」的出厂值
    assert screen["filters"] == []


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


def test_the_universe_block_records_the_boards_that_were_actually_used(tmp_path):
    """股票池口径要按**实际用的**记，不能是写死的字符串。

    原先回测侧记 ``{"rule": "出厂设定"}``、选股侧记「排除次新股，纳入四个板块」——都与
    ``--boards`` 无关。产物要答的是「这一份结果是在哪个池子上算的」，而池子正是被这个开关
    改掉的。
    """
    import json

    root, gbbq = make_dataroot(tmp_path, periods=80)
    args = make_args(
        tdx_root=str(root),
        gbbq=str(gbbq),
        output_dir=str(tmp_path / "runs"),
        boards="主板,创业板",
    )
    out, err = capture()

    assert run_backtest_command(args, stdout=out, stderr=err) == 0, err.getvalue()
    run_dir = next((tmp_path / "runs").iterdir())
    universe = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))["universe"]

    assert universe["boards"] == ["主板", "创业板"]
    assert universe["considered"] == 1  # 进入取数的标的数（--boards 之后的那个池子）


def test_the_universe_block_tells_real_listing_dates_from_the_bars_approximation(tmp_path):
    """次新股门槛用的是哪种上市日，必须记下来——两个口径对窗口内上市的新股结论不同。

    有真实上市日时数的是「自上市日起的交易日」；没有则回退成「本地行情根数」，而后者会把
    「数据刚好从上市日开始」当成「上市很久了」而放行（ADR-0013）。产物若不说用的是哪种，
    事后无从判断。
    """
    import json

    root, gbbq = make_dataroot(tmp_path, periods=80)
    master = Path(__file__).parent / "fixtures" / "master" / "base.dbf"

    for extra, expected in (({"master": str(master)}, "listing_date"), ({}, "available_bars")):
        out_dir = tmp_path / f"runs-{expected}"
        args = make_args(tdx_root=str(root), gbbq=str(gbbq), output_dir=str(out_dir), **extra)
        out, err = capture()

        assert run_backtest_command(args, stdout=out, stderr=err) == 0, err.getvalue()
        run_dir = next(out_dir.iterdir())
        universe = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))["universe"]

        assert universe["recent_listing_rule"] == expected
        # 门槛本身也记：没有开关、恒为出厂值，记它是为了让上一条有主语（「按哪种口径数的 60 天」）。
        assert universe["min_trading_days"] == 60


def test_the_screen_command_says_which_rule_it_used(tmp_path):
    """选股命令也要说明口径——否则「用的哪个口径」成了谜（原先只有 backtest 会打印）。"""
    from mbt.cli import run_screen_command

    root, gbbq = make_dataroot(tmp_path, periods=60)
    args = screen_args(tdx_root=str(root), gbbq=str(gbbq), output_dir=str(tmp_path / "screens"))
    out, err = capture()

    assert run_screen_command(args, stdout=out, stderr=err) == 0, err.getvalue()
    assert "近似口径" in out.getvalue()


def test_the_screen_artifact_records_which_rule_it_ran(tmp_path):
    """``run.json`` 要记下**真正跑的那条规则**，而不是一个写死的常量。

    原先那两行是常量（``rule`` 恒为「按 20 日动量排序取前 N」），与 ``--screen`` 无关——
    于是 ``--screen b1`` 的产物也这么写。做 ADR-0014 的 A/B 时正是先读到那句话，才去核实
    到底跑没跑 b1。产物的用处就是事后回答「这一份候选是用哪条规则算出来的」。
    """
    import json

    from mbt.cli import run_screen_command

    root, gbbq = make_dataroot(tmp_path, periods=60)
    args = screen_args(
        screen="momentum",
        tdx_root=str(root),
        gbbq=str(gbbq),
        output_dir=str(tmp_path / "screens"),
    )
    out, err = capture()

    assert run_screen_command(args, stdout=out, stderr=err) == 0, err.getvalue()
    run_dir = next((tmp_path / "screens").iterdir())
    meta = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))

    assert meta["screen"]["label"] == "momentum"
    assert meta["screen"]["top_n"] == 2


def test_the_brick_screen_runs_without_a_cw_root(tmp_path):
    """砖型只读行情，故 ``--screen brick`` **不需要** ``--cw-root``。

    这与 b1 / undervalued_growth 不同（那两条要财务数据算 PE），也是这条规则更省事的地方：
    少一个必须配对的路径开关。少给一个它根本不用不着的东西，不该报错。
    """
    import json

    from mbt.cli import run_screen_command

    root, gbbq = make_dataroot(tmp_path, periods=80)
    args = screen_args(
        screen="brick",
        tdx_root=str(root),
        gbbq=str(gbbq),
        output_dir=str(tmp_path / "screens"),
    )
    out, err = capture()

    assert run_screen_command(args, stdout=out, stderr=err) == 0, err.getvalue()
    run_dir = next((tmp_path / "screens").iterdir())
    meta = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))

    assert meta["screen"]["label"] == "brick"


def test_the_brick_rule_restricts_the_universe_to_its_own_boards(tmp_path):
    """砖型的板块范围由**规则自己声明**，``--boards`` 没给时就用它：主板、创业板、科创板。

    这一条钉住的是「声明落在规则上、且真的被池子用上」——否则需求里那句「排除北交所」只会
    是一句注释。北交所那只**故意也放进数据源**：它若被收进池子，本断言就会在 ``boards``
    上露馅（``considered`` 也会从 1 变 2）。
    """
    import json

    from mbt.cli import run_screen_command

    root, gbbq = make_dataroot(tmp_path, periods=80)
    # 再摆一只北交所的标的：它的板块可判定（`bj920001`），只是不该被砖型收进来。
    make_dataroot(tmp_path, symbol="bj920001", periods=80, start_price=2000)

    args = screen_args(
        screen="brick",
        tdx_root=str(root),
        gbbq=str(gbbq),
        output_dir=str(tmp_path / "screens"),
    )
    out, err = capture()

    assert run_screen_command(args, stdout=out, stderr=err) == 0, err.getvalue()
    run_dir = next((tmp_path / "screens").iterdir())
    universe = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))["universe"]

    assert universe["boards"] == ["主板", "创业板", "科创板"]
    assert "北交所" not in universe["boards"]


def test_the_backtest_command_can_name_the_brick_rule(tmp_path):
    """回测那条路也能指名砖型，且同样不需要 ``--cw-root``。

    两条路（选股与回测）必须认同一批规则名——``--screen`` 的取值表在两处各有一份，
    漂开之后会出现「选股能跑、回测说没有这条规则」这种只在一条路上才发现的怪事。
    """
    import json

    from mbt.cli import run_backtest_command

    root, gbbq = make_dataroot(tmp_path, periods=80)
    args = make_args(
        screen="brick",
        tdx_root=str(root),
        gbbq=str(gbbq),
        output_dir=str(tmp_path / "runs"),
    )
    out, err = capture()

    code = run_backtest_command(args, stdout=out, stderr=err)

    assert code == 0, err.getvalue()
    run_dir = next((tmp_path / "runs").iterdir())
    meta = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
    assert meta["screen"]["label"] == "brick"
    # 板块范围跟着规则走，回测这条路上也一样（它建的是同一个 `UniverseRules`）。
    assert meta["universe"]["boards"] == ["主板", "创业板", "科创板"]


def test_an_explicit_boards_switch_overrides_the_rules_own_scope(tmp_path):
    """``--boards`` 是**调用方在那个当下的显式选择**，优先级高于规则的声明。

    两条路都留着才说得通：规则声明的是默认（不然每台机器都要手填），``--boards`` 是覆盖
    （不然想临时加回北交所就无路可走）。
    """
    import json

    from mbt.cli import run_screen_command

    root, gbbq = make_dataroot(tmp_path, periods=80)
    args = screen_args(
        screen="brick",
        tdx_root=str(root),
        gbbq=str(gbbq),
        output_dir=str(tmp_path / "screens"),
        boards="主板,创业板,科创板,北交所",
    )
    out, err = capture()

    assert run_screen_command(args, stdout=out, stderr=err) == 0, err.getvalue()
    run_dir = next((tmp_path / "screens").iterdir())
    universe = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))["universe"]

    assert universe["boards"] == ["主板", "创业板", "北交所", "科创板"]


def make_ex_dividend_dataroot(tmp_path, periods=140):
    """两只标的：只有 ``sh600000`` 在窗口内除权（2024-07-18 每 10 股派 3.21 元）。

    窗口 2024-03-01 起，故 ``--as-of`` 那天有 100 根以上的行情——股票池要「上市以来至少
    60 个交易日」，窗口太短会一只都留不下，那测的就不是选股了。

    ``sz000001`` 的 2024-06-14 除权虽然也在窗口内，但它在这段里**价格是平的**（造数据时
    没让它跌），故折算前后都是 0% 动量——它在这条测试里是不除权、也不涨跌的对照。

    价格造得让两条序列给出**相反的**动量排序（除权后第 19 个交易日，2024-08-14）：

    ================  ==================  ==================
    标的              原始价 20 日动量     后复权 20 日动量
    ================  ==================  ==================
    sh600000          −1.0%（除权假跌）    **+2.28%**
    sz000001          0%                  0%
    ================  ==================  ==================

    ``sh600000`` 在除权日从 10.00 落到 9.90，跌得比每股 0.321 元的分红**少**；后复权乘上
    **除权因子的倒数**（即 ``前收盘 ÷ 参考价``）把它折算回去，假跳空于是被抹平。取前 1 名
    时，「用哪条序列算」就决定了名单上是哪一只。

    差异只在除权日之后的 20 根内存在（再往后回看窗口整段落在除权之后，两条序列又都趋平），
    故两个测试都取 ``index[k + 19]``（除权后第 19 根）。返回 ``ex_date`` 是为了让测试自己
    推算日子，不在断言里写死哪一天。

    返回 ``(root, gbbq, index, ex_date)``。
    """
    index = pd.bdate_range("2024-03-01", periods=periods)
    ex_date = pd.Timestamp("2024-07-18")
    root = tmp_path / "vipdoc"

    write_day(
        root / "sh" / "lday" / "sh600000.day",
        [
            (int(f"{stamp:%Y%m%d}"), price, price, price, price, 0.0, 1000)
            for stamp, price in zip(
                index, [1000 if s < ex_date else 990 for s in index], strict=True
            )
        ],
    )
    write_day(
        root / "sz" / "lday" / "sz000001.day",
        [(int(f"{stamp:%Y%m%d}"), 1000, 1000, 1000, 1000, 0.0, 1000) for stamp in index],
    )
    gbbq = tmp_path / "gbbq"
    gbbq.write_bytes(GBBQ_FIXTURE.read_bytes())
    return root, gbbq, index, ex_date


def run_screen_and_read_candidates(tmp_path, root, gbbq, day):
    """跑一次 ``mbt screen``（动量、只取 1 只），返回候选名单与那次跑的产物目录。

    #73 的两条测试都要这一小段：一条看名单本身，一条拿名单去比回测**成交**的标的。
    """
    out, err = capture()
    args = screen_args(
        screen="momentum",
        top_n=1,
        as_of=f"{day:%Y-%m-%d}",
        tdx_root=str(root),
        gbbq=str(gbbq),
        output_dir=str(tmp_path / "screens"),
    )

    assert run_screen_command(args, stdout=out, stderr=err) == 0, err.getvalue()
    run_dir = next((tmp_path / "screens").iterdir())
    return pd.read_csv(run_dir / "candidates.csv")["symbol"].tolist(), run_dir


def test_the_screen_command_screens_on_backward_adjusted_prices(tmp_path):
    """选股落在**后复权**价上，与回测是同一条序列（票据 #73）。

    此前选股命令把原始价直接喂给 ``assemble_panel``，而回测喂的是后复权价——同一个
    ``Screen`` 于是在两条序列上被评估，每日选股的名单与回测所依据的规则给出的结果于是不一致。
    ``tests/test_screen.py`` 的 ``test_the_screen_sees_the_same_price_series_that_gets_traded``
    钉的是**引擎那半**；这条钉**选股命令这半**，合起来才是「同一条序列」。

    夹具上 ``sh600000`` 在原始价下是 −1%（除权假跌）、在后复权价下是 +2.28%，而
    ``sz000001`` 两条都是 0%。故名单上是 ``sh600000`` 才说明用的是后复权价。

    **为什么用 ``momentum`` 而不是票据里点名的 ``b1``**：这里要钉的是「用的是哪条序列」，
    而动量是**最小的判别器**——它的排序因子直接读价格，夹具上两条序列给出的排序正好相反。
    b1 的八道门在这样一份合成行情上（价格平淡、没有摆动点与量能结构）一只都选不出来，故回
    答不了这个问题；而「两条路同一条序列」是**规则无关**的（两边调同一个入口），动量这一条
    足以把它钉住，b1 的其余门由它自己的那批测试管。

    评估日取除权日**之后**第 19 根而非除权当日：假跳空会留在其后整个均线窗口里，这正是本票
    要修的那种持续失真（票据 #73 的「除权事件当日的假跳空不再影响选股结果」）。
    """
    import json

    root, gbbq, index, ex_date = make_ex_dividend_dataroot(tmp_path)
    day = index[index.get_loc(ex_date) + 19]
    picked, run_dir = run_screen_and_read_candidates(tmp_path, root, gbbq, day)

    assert picked == ["sh600000"], "候选是在原始价上算的——除权日的假跌把 sh600000 的动量压成了 −1%"
    assert json.loads((run_dir / "run.json").read_text(encoding="utf-8"))["candidates"] == 1


def test_screen_and_backtest_pick_the_same_symbol_on_the_same_day(tmp_path):
    """两条路在同一个评估日必须给出**同一个答案**（票据 #73 的验收）。

    回测那边一直是对的（``engine.py`` 自己套了后复权），错的是选股命令。故修之前这两条会
    分家：选股按原始价选 ``sz000001``，而回测按后复权价买 ``sh600000``。

    可观测的差别在产物上：``screen`` 给的是 ``candidates.csv``，``backtest`` 的选股结果只能
    从**成交**看出来（选股规则是入场闸门）。于是「名单」与「成交的标的」相等就是这条验收。

    回测区间取 ``[除权日 − 40 根, 除权日 + 40 根]``，评估日取 ``除权日 + 19``：

    - 长度必须 ≥ 60 根，否则**股票池**会把两只都当成次新股剔掉（引擎是在**已截断**的行情上
      建池子的），而「满 60 根」的那一根出现在区间靠后的位置；
    - 开头的 20 根给动量一个可用的回看窗口（引擎是在已截断的面板上算排序因子的，窗口太短
      会全是缺失值、一个都不选）；
    - 除权日之后的 20 根内两条序列才会分家，故评估日取在那段内；区间再往后留 20 根，好让
      订单有机会成交（末尾那根下的单没有下一根可撮合）。

    除权日之前两条序列都是平的，动量同分，按「代码升序」判给 ``sh600000``——那与后复权
    的答案一致，故不影响本条的结论。
    """
    root, gbbq, index, ex_date = make_ex_dividend_dataroot(tmp_path)
    k = index.get_loc(ex_date)
    day = index[k + 19]
    start, end = index[k - 40], index[k + 40]

    listed, _ = run_screen_and_read_candidates(tmp_path, root, gbbq, day)

    back_out, back_err = capture()
    assert (
        run_backtest_command(
            make_args(
                screen="momentum",
                screen_top_n=1,
                start=f"{start:%Y-%m-%d}",
                end=f"{end:%Y-%m-%d}",
                tdx_root=str(root),
                gbbq=str(gbbq),
                output_dir=str(tmp_path / "runs"),
            ),
            stdout=back_out,
            stderr=back_err,
        )
        == 0
    ), back_err.getvalue()
    trades = pd.read_csv(next((tmp_path / "runs").iterdir()) / "trades.csv")

    assert listed, "选股一个都没选出来，这条就白测了"
    assert sorted(trades["symbol"].unique()) == sorted(
        listed
    ), "选股名单与回测成交的标的不是同一批——两条路用的价格序列不同"


def test_the_screen_metadata_is_a_function_of_the_screen_switch(tmp_path):
    """两份产物**必须互不相同**——这条钉的是「元数据随 ``--screen`` 变化」本身。

    上面那两条各钉一条规则的内容，却都答不了「换规则时元数据会不会跟着换」：若有人把标签
    重新写死成一个常量，两条里各有一半仍可能对上。同一次比对直接把这条关系钉住。
    """
    import json

    from mbt.cli import run_screen_command

    root, gbbq = make_dataroot(tmp_path, periods=60)
    blocks = {}
    for name in ("momentum", "b1"):
        out_dir = tmp_path / f"screens-{name}"
        args = screen_args(
            screen=name,
            tdx_root=str(root),
            gbbq=str(gbbq),
            output_dir=str(out_dir),
            cw_root=str(Path(__file__).parent / "fixtures" / "cw"),
        )
        out, err = capture()

        assert run_screen_command(args, stdout=out, stderr=err) == 0, err.getvalue()
        run_dir = next(out_dir.iterdir())
        blocks[name] = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))["screen"]

    assert blocks["momentum"]["label"] == "momentum"
    assert blocks["b1"]["label"] == "b1"
    assert blocks["momentum"] != blocks["b1"]


def test_the_b1_artifact_names_all_eight_filters_and_the_three_pattern_components(tmp_path):
    """``--screen b1`` 的产物要能看出跑的是 b1：八条过滤器 + 三个形态分数分量 + 权重 + 秩归一。

    ADR-0012 的核心改动就是「排序因子从 ``j_oversold`` 换成形态分数」，而它在留痕里原先
    是 ``null``——产物于是答不出「这一份候选是用哪条规则算出来的」。
    """
    import json
    from pathlib import Path

    from mbt.cli import run_screen_command

    root, gbbq = make_dataroot(tmp_path, periods=60)
    args = screen_args(
        screen="b1",
        tdx_root=str(root),
        gbbq=str(gbbq),
        output_dir=str(tmp_path / "screens"),
        cw_root=str(Path(__file__).parent / "fixtures" / "cw"),
    )
    out, err = capture()

    assert run_screen_command(args, stdout=out, stderr=err) == 0, err.getvalue()
    run_dir = next((tmp_path / "screens").iterdir())
    screen = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))["screen"]

    assert screen["label"] == "b1"
    assert screen["filters"] == [
        "trend",
        "not_below_the_line",
        "position",
        "low_j",
        "volume",
        "no_flat_pullback",
        "profitable",
        "cheap",
    ]
    assert screen["factors"] == ["top_calm", "pullback_shrink", "top_shadow"]
    assert screen["weights"] == [1.0, 1.0, 1.0]
    assert screen["normalize"] == "rank"


def test_an_explicit_as_of_is_still_named_in_the_log(tmp_path):
    """显式给了 ``--as-of`` 也要印出评估日——日志该自证「这次按哪天算的」。

    缺省那条路径自己会印（连同被跳过的残缺日）。这条钉的是另一条：给了日期时别把
    「按哪天评估」弄丢——候选清单上只有代码，没有日期，从日志反推不出来。
    """
    root, gbbq = make_dataroot(tmp_path, periods=60)
    args = screen_args(tdx_root=str(root), gbbq=str(gbbq), output_dir=str(tmp_path / "screens"))
    out, err = capture()

    assert run_screen_command(args, stdout=out, stderr=err) == 0, err.getvalue()
    assert "评估日：2024-03-01" in out.getvalue()


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


def test_placing_no_order_at_all_is_reported_differently_from_all_rejected(tmp_path):
    """「一笔订单都没下过」与「下了单全被拒」要分开说——两者的排查方向完全不同。

    后者去 ``rejected.csv`` 看理由；前者是**什么都没发生**，通常是股票池或选股规则在整段
    期间没产出候选。实测撞到过前者：股票池排除了北交所，而 ``--limit`` 取到的全是北交所，
    于是回测零成交、退出码 0，看着像个「本来就没信号」的正常结果。
    """
    root, gbbq = make_dataroot(tmp_path, periods=30)  # 不足门槛 60 → 股票池为空
    args = make_args(
        strategy="examples.strategies:UndervaluedGrowth",
        tdx_root=str(root),
        gbbq=str(gbbq),
        output_dir=str(tmp_path / "runs"),
    )
    out, err = capture()

    assert run_backtest_command(args, stdout=out, stderr=err) == 0, err.getvalue()
    text = out.getvalue()
    assert "没有下达任何订单" in text
    assert "一笔都没成交" not in text, "没下过单就别说「有下单」"


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
    assert metrics["win_rate"] is not None, "有平仓交易就该有胜率"

    # 盈亏比在「**没有亏损笔**」时本就无定义（``mean(亏损)`` 的分母是空的），且原因会写进
    # ``notes``。这条断言把两种情况分开说清，而不是含糊地要求它非空——后者会逼出一个假数字。
    if metrics["payoff_ratio"] is None:
        assert any("没有亏损" in note for note in metrics["notes"]), "缺失必须给出理由"


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


def test_the_report_counts_the_skips_from_loading_not_just_from_slicing(tmp_path):
    """取数阶段的跳过必须**带进切片**，否则会从报告里消失（票据 #45）。

    实测一次 300 只的运行：取数阶段跳了 84 只，而回测摘要只报「跳过 2」——那 2 是切片阶段
    的，84 只无声无息。根因是 ``slice_markets(markets, ...)`` 只收 markets，上游的跳过被
    整段丢掉。这条钉住「两段跳过都要出现在报告里」。
    """
    import pandas as pd

    from mbt.data import UniverseLoad, slice_markets
    from mbt.data.loader import NOT_STOCK, SkippedSymbol
    from mbt.data.market import MarketData

    earlier = [SkippedSymbol("sh000001", NOT_STOCK, "品种不是股票")]
    index = pd.bdate_range("2024-01-02", periods=5)
    frame = pd.DataFrame(
        {
            "open": [10.0] * 5,
            "high": [10.0] * 5,
            "low": [10.0] * 5,
            "close": [10.0] * 5,
            "volume": [1000] * 5,
        },
        index=index,
    )
    markets = [MarketData(symbol="sh600000", prices=frame, events=())]

    sliced = slice_markets(markets, start="2024-01-02", end="2024-01-05", skipped=earlier)

    assert isinstance(sliced, UniverseLoad)
    assert [item.symbol for item in sliced.skipped] == ["sh000001"], "上游的跳过被丢掉了"
    assert len(sliced.markets) == 1


def test_the_panel_window_keeps_exactly_the_requested_number_of_rows():
    """见 `tests/test_panel.py`：库侧的口径测试在那边（这里只测 CLI 怎么用它）。"""


# --- 每日选股的面板窗口（``--panel-bars``）-----------------------------------
#
# 面板日历截断是**提升内存上限**的手段，不是一次口径变更，故这一组钉的是
# 「切了要说、切不动要说、切得太短要拒」，而不是「切完结果仍然一样」——
# 后者是信号层的契约（见 test_valuation 那条 1000/999 断崖）。


def test_panel_bars_is_a_screen_only_switch():
    """与 ``--forward-root`` 同理：回测那条路上**没有**这个开关。

    回测的区间由 ``--start`` / ``--end`` 给，而它是各自算区间、逐日推进的，不需要一个
    「末端窗口」；把这个开关也开给回测，只会让人以为它也有同样的内存效果。
    """
    parser = build_parser()

    screen = parser.parse_args(["screen", "--tdx-root", "x", "--gbbq", "y", "--output-dir", "z"])
    assert screen.panel_bars == 0, "默认必须是「不截断」"

    backtest = parser.parse_args(
        [
            "backtest",
            "--strategy",
            EXAMPLE_STRATEGY,
            "--tdx-root",
            "x",
            "--gbbq",
            "y",
            "--output-dir",
            "z",
        ]
    )
    assert not hasattr(backtest, "panel_bars")


def test_panel_bars_cuts_the_panel_calendar_to_the_tail(tmp_path):
    """切了要说（日志 + 产物）：同一份名单按 1301 根还是按 1000 根历史算出来的，看 CSV 分不出。"""
    import json

    root, gbbq = make_dataroot(tmp_path, periods=1200)
    args = screen_args(
        tdx_root=str(root),
        gbbq=str(gbbq),
        output_dir=str(tmp_path / "screens"),
        as_of=None,  # 由数据自己定评估日，故评估日就在末根
        panel_bars=1100,
    )
    out, err = capture()

    code = run_screen_command(args, stdout=out, stderr=err)

    assert code == 0, err.getvalue()
    assert "面板：日历 1200 行 → 1100 行" in out.getvalue()
    run_dir = next((tmp_path / "screens").iterdir())
    metadata = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
    assert metadata["panel_bars"] == 1100
    assert metadata["panel_rows"] == 1100


def test_panel_bars_below_the_deepest_lookback_refuses_to_run(tmp_path):
    """窗口短于规则的深回看 → **拒跑**，而不是给一个按更短历史算出来的答案。

    ``pe_percentile`` 对不足窗口的历史照常给值（``min_periods=1``），故这件事不会有别的
    地方发现：名单照旧生成，只是读数算在更短的窗口上（实测 999 行就有 81 只标的不同）。
    """
    root, gbbq = make_dataroot(tmp_path, periods=1200)
    args = screen_args(
        tdx_root=str(root),
        gbbq=str(gbbq),
        output_dir=str(tmp_path / "screens"),
        as_of=None,
        panel_bars=PANEL_WARMUP_BARS - 1,
    )
    out, err = capture()

    code = run_screen_command(args, stdout=out, stderr=err)

    assert code == 1
    assert str(PANEL_WARMUP_BARS) in err.getvalue()
    assert not (tmp_path / "screens").exists(), "拒绝运行就不该留下产物目录"


def test_the_gate_also_fires_when_the_data_is_too_short_to_cut(tmp_path):
    """**闸门看的是「实际会用到多少历史」，不是「请求了多少」。**

    这条是本票最容易漏的一格：请求 1301 根，而数据只有 900 根，于是**切不动**——此时若把闸门
    挂在「切没切」上，它就整个跳过了，而评估日身后只有 900 行，``pe_percentile`` 照样在一个
    比 1000 更短的历史上给值，名单照样生成。两种情形（请求太少、数据太短）后果相同，故都得拦。
    """
    root, gbbq = make_dataroot(tmp_path, periods=PANEL_WARMUP_BARS - 100)
    args = screen_args(
        tdx_root=str(root),
        gbbq=str(gbbq),
        output_dir=str(tmp_path / "screens"),
        as_of=None,
        panel_bars=1301,  # 比可得历史长，切不动
    )
    out, err = capture()

    code = run_screen_command(args, stdout=out, stderr=err)

    assert code == 1
    assert "900" in err.getvalue(), "报错要说清实际剩了多少行"
    assert not (tmp_path / "screens").exists()


def test_a_request_longer_than_the_data_but_still_deep_enough_proceeds(tmp_path):
    """切不动、但身后的历史**够深** → 不切，继续跑，并如实说「未截断」。

    这才是「历史本来就比 N 短」那句话的适用范围（ADR-0014 决策 9）：不是「短了就一律放行」，
    而是「短到切不动、但没短到影响读数」——两个条件都要满足。
    """
    root, gbbq = make_dataroot(tmp_path, periods=PANEL_WARMUP_BARS + 200)
    args = screen_args(
        tdx_root=str(root),
        gbbq=str(gbbq),
        output_dir=str(tmp_path / "screens"),
        as_of=None,
        panel_bars=PANEL_WARMUP_BARS + 500,  # 比可得历史长，切不动
    )
    out, err = capture()

    code = run_screen_command(args, stdout=out, stderr=err)

    assert code == 0, err.getvalue()
    assert f"不足请求的 {PANEL_WARMUP_BARS + 500} 根，未截断" in out.getvalue()


def test_a_negative_panel_bars_is_a_usage_error():
    """``--panel-bars -1`` 看着像「不截断」，实际是「比任何真实窗口都短」——故是用法错误。

    这一格若被当成 0，闸门就被绕过了（两个数在 ``min(bars, 历史行数)`` 里含义相反）。
    """
    with pytest.raises(SystemExit):
        build_parser().parse_args(["screen", "--panel-bars", "-1"])


# --- 每日选股：评估日、全池、自选股文件 ----------------------------------------
#
# 这一段钉的是「一个 .bat 每天盘后跑一次」所需的那些接口：喂不了日期参数、要全池而不是
# 前 10、产物是一个**固定名**的自选股文件。


def test_the_parser_takes_all_as_a_top_n_meaning_no_truncation():
    parser = build_parser()

    given = parser.parse_args(
        ["screen", "--tdx-root", "x", "--gbbq", "y", "--output-dir", "z", "--top-n", "all"]
    )
    omitted = parser.parse_args(["screen", "--tdx-root", "x", "--gbbq", "y", "--output-dir", "z"])

    assert given.top_n == ALL_CANDIDATES
    assert omitted.top_n == 10, "不给仍是原来的默认——这一条改的是新取值，不是默认值"


def test_a_top_n_that_is_neither_a_count_nor_all_names_what_it_wanted():
    parser = build_parser()

    with pytest.raises(SystemExit):
        parser.parse_args(
            ["screen", "--tdx-root", "x", "--gbbq", "y", "--output-dir", "z", "--top-n", "abc"]
        )


def test_all_reaches_the_rule_as_no_truncation_rather_than_falling_back():
    """``--top-n all`` 必须到得了规则那一层，不能被「取第一个给了的」的回退链吞掉。

    回退链的原样是 ``or``，而 ``all`` 映射成 ``None`` 之后在 ``or`` 里等于「没给」——
    于是「不截断」会静默变成「取 5」。这条是冲着那个静默去的。
    """
    from mbt.cli import _screen_and_signals

    out = io.StringIO()
    args = screen_args(screen="momentum", top_n=ALL_CANDIDATES)

    screen, _ = _screen_and_signals(args, None, None, out)

    assert screen.top_n is None, "「不截断」没到规则那一层"


def test_a_top_n_of_ten_still_reaches_the_rule_as_ten():
    """对照面：数字照旧传下去——否则上一条可以通过「一律不截断」而假绿。"""
    from mbt.cli import _screen_and_signals

    out = io.StringIO()
    args = screen_args(screen="momentum", top_n=10)

    screen, _ = _screen_and_signals(args, None, None, out)

    assert screen.top_n == 10


def test_the_b1_screen_needs_a_cw_root_because_two_of_its_filters_read_financials(tmp_path):
    """B1 的最后两条过滤（PE 为正、PE 百分位低）不在行情里，故缺 --cw-root 要报错而不是少两条。"""
    root, gbbq = make_dataroot(tmp_path, periods=60)
    args = screen_args(
        screen="b1",
        tdx_root=str(root),
        gbbq=str(gbbq),
        output_dir=str(tmp_path / "screens"),
        cw_root=None,
    )
    out, err = capture()

    assert run_screen_command(args, stdout=out, stderr=err) == 1
    assert "cw-root" in err.getvalue()


def test_forward_root_is_a_screen_only_switch(tmp_path):
    """``--forward-root`` **只在 ``screen`` 上**——回测侧的参数表里没有它。

    这一条与 ADR-0006 修订二同源：一致预期是快照、没有历史，回测的评估日全在过去，故它
    在回测里永远只是前视偏差。挡法不是「运行时判断一下」，而是**回测那条路根本没有这个开关**，
    于是也没有那段代码路径。哪天有人图省事把它加到 ``backtest`` 上，这条会红。
    """
    parser = build_parser()
    screen_choices = parser.parse_args(
        ["screen", "--tdx-root", "x", "--gbbq", "y", "--output-dir", "z"]
    )
    assert hasattr(screen_choices, "forward_root")

    backtest = parser.parse_args(
        [
            "backtest",
            "--strategy",
            "m:S",
            "--tdx-root",
            "x",
            "--gbbq",
            "y",
            "--output-dir",
            "z",
        ]
    )
    assert not hasattr(backtest, "forward_root"), "回测不该有前瞻数据入口"


def test_the_daily_screen_reports_the_forward_caliber_when_the_forward_data_is_there(tmp_path):
    """给了一致预期目录、且评估日不早于它的写入日 → 选股按「选用PE」跑，并**印出来**。

    行情用 ``sh600000``（那份夹具里它有一致预期：财年 2026、EPS_T 1.521），故它该走前瞻。
    三条条件都要凑齐，缺一条就退回历史：写入日早于评估日、**评估日的年份等于财年 2026**、
    行情取的是原始价。第二条件由 ``start="2026-06-01"`` 保证——原式的 ``K线年 = 财年T``。
    """
    from pathlib import Path

    from mbt.cli import run_screen_command

    root, gbbq = make_dataroot(tmp_path, periods=60, start="2026-06-01")
    forward_root = _forward_root_written_on(tmp_path, "2026-06-15")
    args = screen_args(
        screen="b1",
        as_of="2026-07-01",
        tdx_root=str(root),
        gbbq=str(gbbq),
        output_dir=str(tmp_path / "screens"),
        cw_root=str(Path(__file__).parent / "fixtures" / "cw"),
        forward_root=str(forward_root),
    )
    out, err = capture()

    assert run_screen_command(args, stdout=out, stderr=err) == 0, err.getvalue()
    assert "前瞻：1/1 个标的按「选用PE」评估" in out.getvalue()

    import json

    run_dir = next((tmp_path / "screens").iterdir())
    metadata = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
    assert metadata["valuation_caliber"]["forward_symbols"] == 1
    assert "选用PE" in metadata["valuation_caliber"]["caliber"]


def test_an_evaluation_day_before_the_forward_file_was_written_falls_back(tmp_path):
    """评估日**早于**写入日 → 一个标的都不走前瞻，退回历史口径。

    这是「回测里前瞻永不生效」在命令层的同一条性质：回测的评估日全部落在写入日之前。
    年份刻意与财年一致（都在 2026），好让红的是**日期那一关**，不是年份那一关。
    """
    from pathlib import Path

    from mbt.cli import run_screen_command

    root, gbbq = make_dataroot(tmp_path, periods=60, start="2026-06-01")
    forward_root = _forward_root_written_on(tmp_path, "2026-07-20")
    args = screen_args(
        screen="b1",
        as_of="2026-07-01",
        tdx_root=str(root),
        gbbq=str(gbbq),
        output_dir=str(tmp_path / "screens"),
        cw_root=str(Path(__file__).parent / "fixtures" / "cw"),
        forward_root=str(forward_root),
    )
    out, err = capture()

    assert run_screen_command(args, stdout=out, stderr=err) == 0, err.getvalue()
    assert "前瞻：0/1 个标的按「选用PE」评估" in out.getvalue()


def test_a_market_without_consensus_data_is_reported_without_crying_wolf(tmp_path):
    """北交所标的进池时，只报「这市场没有这份数据」，**不许**喊「--forward-root 指对了吗」。

    这一条防的是**噪声淹没警报**：本机实测 5,375 个标的里有 276 只北交所，而客户端没有
    ``gpbjone.dat``。若把它和「沪深文件读不到」混成一个数，日更任务每天都会喊一次「目录指错了
    吗」——真指错的那天就淹在噪声里了。故这里同时钉三件事：数量对（1/2 走前瞻）、
    话对（说明市场无数据）、**警报不响**。
    """
    from pathlib import Path

    from mbt.cli import run_screen_command

    root, gbbq = make_dataroot(tmp_path, periods=60, start="2026-06-01")
    # 再加一只北交所标的：它没有一致预期，因为**客户端没有那个文件**，与目录对不对无关。
    make_dataroot(tmp_path, symbol="bj920001", periods=60, start="2026-06-01")

    forward_root = _forward_root_written_on(tmp_path, "2026-06-15")
    args = screen_args(
        screen="b1",
        as_of="2026-07-01",
        tdx_root=str(root),
        gbbq=str(gbbq),
        output_dir=str(tmp_path / "screens"),
        cw_root=str(Path(__file__).parent / "fixtures" / "cw"),
        forward_root=str(forward_root),
    )
    out, err = capture()

    assert run_screen_command(args, stdout=out, stderr=err) == 0, err.getvalue()
    printed = out.getvalue()

    assert "前瞻：1/2 个标的按「选用PE」评估" in printed
    assert "bj920001" in printed, "要点名是哪个标的所在市场没有数据，免得看起来像漏掉了"
    assert "指对了吗" not in printed, "北交所是已知缺口，不是路径错误——不许报警"
    assert "注意：" not in printed

    import json

    run_dir = next((tmp_path / "screens").iterdir())
    metadata = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
    caliber = metadata["valuation_caliber"]
    assert caliber["unsupported"] == 1, "北交所那只记进 unsupported"
    assert caliber["unreadable"] == 0, "它不该被算成「读不到」"


def test_a_screen_other_than_b1_ignores_the_forward_root(tmp_path):
    """``--forward-root`` 只对 ``b1`` 有意义——别的规则不读 PE，这一步整步跳过。

    不跳过的话，产物里会留一份「按前瞻跑的」记录，而那份名单其实与前瞻无关。
    """
    from pathlib import Path

    from mbt.cli import run_screen_command

    root, gbbq = make_dataroot(tmp_path, periods=60)
    forward_root = _forward_root_written_on(tmp_path, "2024-02-01")
    args = screen_args(
        screen="momentum",
        as_of="2024-03-01",
        tdx_root=str(root),
        gbbq=str(gbbq),
        output_dir=str(tmp_path / "screens"),
        cw_root=str(Path(__file__).parent / "fixtures" / "cw"),
        forward_root=str(forward_root),
    )
    out, err = capture()

    assert run_screen_command(args, stdout=out, stderr=err) == 0, err.getvalue()
    assert "前瞻：" not in out.getvalue()

    import json

    run_dir = next((tmp_path / "screens").iterdir())
    metadata = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
    assert metadata["valuation_caliber"] is None


def test_the_growth_screen_restates_the_peg_and_says_how_many_it_dropped(tmp_path):
    """``undervalued_growth`` 按前瞻跑时，PEG 也要重述——并**印出**它摘掉了多少只。

    两件事一起钉：① 原式那条 ``财年T>0`` 的守卫真的生效了（``sh600006`` 在一致预期文件里但
    没有预期，故评估日那天它的 PEG 该是缺失）；② 这个数必须说出来——PEG 那道门这次只看得到
    池子的一部分，不印出来，读产物的人只会看到候选变少而不知道为什么。
    """
    from pathlib import Path

    from mbt.cli import run_screen_command

    root, gbbq = make_dataroot(tmp_path, periods=60, start="2026-06-01")
    # 再加一只**在一致预期文件里、但没有预期**的标的（财年T = 0），它就是被守卫摘掉的那个。
    make_dataroot(tmp_path, symbol="sh600006", periods=60, start="2026-06-01")
    forward_root = _forward_root_written_on(tmp_path, "2026-06-15")
    args = screen_args(
        screen="undervalued_growth",
        as_of="2026-07-01",
        tdx_root=str(root),
        gbbq=str(gbbq),
        output_dir=str(tmp_path / "screens"),
        cw_root=str(Path(__file__).parent / "fixtures" / "cw"),
        forward_root=str(forward_root),
    )
    out, err = capture()

    assert run_screen_command(args, stdout=out, stderr=err) == 0, err.getvalue()
    printed = out.getvalue()
    assert "前瞻：1/2 个标的按「选用PE」评估" in printed
    assert "PEG：1/2 个标的在评估日没有 PEG" in printed

    import json

    run_dir = next((tmp_path / "screens").iterdir())
    metadata = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
    caliber = metadata["valuation_caliber"]
    assert caliber["forward_symbols"] == 1
    assert caliber["peg_missing"] == 1, "摘掉了谁必须记进产物，不能只在屏幕上闪过"
    assert "选用增" in caliber["peg_caliber"]


def test_the_legacy_valuation_screen_keeps_the_historical_peg(tmp_path):
    """``valuation``（旧版，留给对照实验用）**不开**前瞻——它要是也换口径，对照就一起动了。

    与 ``momentum`` 那条的区别在于：``valuation`` 真的读 PE 与 PEG，所以「不开」是个取舍，
    不是「用不上」。故它单独测一遍，免得日后有人按「读估值的都该开」把它加进名单。
    """
    from pathlib import Path

    from mbt.cli import run_screen_command

    root, gbbq = make_dataroot(tmp_path, periods=60, start="2026-06-01")
    forward_root = _forward_root_written_on(tmp_path, "2026-06-15")
    args = screen_args(
        screen="valuation",
        as_of="2026-07-01",
        tdx_root=str(root),
        gbbq=str(gbbq),
        output_dir=str(tmp_path / "screens"),
        cw_root=str(Path(__file__).parent / "fixtures" / "cw"),
        forward_root=str(forward_root),
    )
    out, err = capture()

    assert run_screen_command(args, stdout=out, stderr=err) == 0, err.getvalue()
    assert "前瞻：" not in out.getvalue()

    import json

    run_dir = next((tmp_path / "screens").iterdir())
    metadata = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
    assert metadata["valuation_caliber"] is None


def test_the_data_snapshot_records_the_consensus_files_it_read(tmp_path):
    """那份 ``gp*one.dat`` 进了 ``run.json`` 的数据摘要（票据 #50 修订）。

    它从接上前瞻那天起就是选股的真实输入之一，却一直没进摘要——于是产物说的「这次读了哪些
    文件」是错的。这里同时钉两件事：**跑了前瞻就记**，**没跑前瞻就不记**（后者由
    ``momentum`` 那条守着，见上）。
    """
    import json
    from pathlib import Path

    from mbt.cli import run_screen_command

    root, gbbq = make_dataroot(tmp_path, periods=60, start="2026-06-01")
    forward_root = _forward_root_written_on(tmp_path, "2026-06-15")
    args = screen_args(
        screen="b1",
        as_of="2026-07-01",
        tdx_root=str(root),
        gbbq=str(gbbq),
        output_dir=str(tmp_path / "screens"),
        cw_root=str(Path(__file__).parent / "fixtures" / "cw"),
        forward_root=str(forward_root),
    )
    out, err = capture()

    assert run_screen_command(args, stdout=out, stderr=err) == 0, err.getvalue()

    run_dir = next((tmp_path / "screens").iterdir())
    metadata = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
    manifest = [entry["path"] for entry in metadata["data_snapshot"]["manifest"]]

    assert any(path.endswith("gpshone.dat") for path in manifest), manifest
    assert not any(path.endswith("gpbjone.dat") for path in manifest), "本机没有这份，别记假的"


def _forward_root_written_on(tmp_path, day: str):
    """把真实夹具拷成一份一致预期目录，并把**写入日**改成给定的一天。

    写入日是关键：闸门比的就是它（评估日不早于它才许用前瞻）。夹具在仓库里的写入日
    是它被创建那天，比这些合成行情的日期晚，故要显式改成我们要的那天。
    """
    import datetime as dt
    import os
    import shutil
    from pathlib import Path

    target = tmp_path / "hq_cache"
    target.mkdir()
    copied = target / "gpshone.dat"
    shutil.copy(Path(__file__).parent / "fixtures" / "gpone" / "gpshone.dat", copied)
    stamp = dt.datetime.fromisoformat(f"{day}T15:00:00").timestamp()
    os.utime(copied, (stamp, stamp))
    return target


def test_a_backtest_shaped_args_cannot_turn_the_forward_caliber_on(tmp_path):
    """就算有人把这一步挪到共用路径上，**回测那套 args 也点不亮它**。

    这是上面那条之外的**第二重保险**，因为两条防的是不同的事：那条防「参数表里多出开关」，
    这条防「调用点被挪进共用路径」。回测的 args 没有 ``forward_root``，取到 None 即整步跳过，
    ``signals`` 一格不动——于是回测拿到的 PE 与改这条分支之前逐格相同。

    不在这里跑真回测：要让它逐笔成交得起一段八条过滤器全过的行情，而那件事
    :mod:`tests.test_b1_screen` 已在规则层钉到具体某一根（134/135/136），CLI 层再照搬一遍
    只是把同一段行情写两处。
    """
    import datetime as dt

    from mbt.cli import _forward_caliber

    args = make_args(screen="b1", cw_root=str(Path(__file__).parent / "fixtures" / "cw"))
    signals = {"pe": pd.DataFrame({"sh600000": [10.0, 20.0]})}
    out = io.StringIO()

    before = signals["pe"].copy()
    caliber = _forward_caliber(
        args,
        signals=signals,
        markets=[],
        as_of=dt.date(2024, 3, 1),
        screen_label="b1",
        stdout=out,
    )

    assert caliber is None
    assert signals["pe"].equals(before), "回测那套 args 竟然动了 PE"


def test_backtest_can_also_use_the_b1_gate(tmp_path):
    """``backtest`` 的 ``--screen`` 也要收 ``b1``——否则搬进库的 ``b1_signals`` 没有调用者。

    两条命令共用同一个 :func:`_screen_and_signals`，里面那条 b1 分支本来就写得能从两边用
    （它用 ``getattr`` 兜 ``start`` / ``end``）。可只要回测侧的 ``choices`` 不收 ``b1``，
    那条分支在回测侧就**永远够不着**，而 ADR-0012 那批全市场结论正是用 b1 入口跑出来的——
    「搬进库」若只搬到一个够不着的角落，等于没搬。

    这里钉的不是参数层收不收，而是**它真的走到了 b1 分支**：缺 ``--cw-root`` 时给出的
    是 b1 自己那句错，而不是 ``choices`` 甩回来的「invalid choice」。
    """
    parser = build_parser()
    parsed = parser.parse_args(
        [
            "backtest",
            "--strategy",
            "m:S",
            "--tdx-root",
            "x",
            "--gbbq",
            "y",
            "--cw-root",
            "w",
            "--output-dir",
            "z",
            "--screen",
            "b1",
        ]
    )
    assert parsed.screen == "b1", "回测侧的 choices 不收 b1，这条分支就永远够不着"

    root, gbbq = make_dataroot(tmp_path, periods=60)
    args = make_args(
        screen="b1",
        tdx_root=str(root),
        gbbq=str(gbbq),
        output_dir=str(tmp_path / "runs"),
        cw_root=None,
    )
    out, err = capture()

    assert run_backtest_command(args, stdout=out, stderr=err) == 1
    assert "cw-root" in err.getvalue()


def test_omitting_as_of_takes_the_latest_day_in_the_data_and_says_which(tmp_path):
    """定时任务喂不了日期参数，故缺省要能自己定出评估日，并把它印出来。

    **印出来这一半和定出来那一半同样要紧**：数据新鲜度的闸门是刻意不要的，于是
    「跑的是哪一天」是唯一还能发现「数据没跟上」的地方。
    """
    root, gbbq = make_dataroot(tmp_path, periods=60)
    args = screen_args(
        tdx_root=str(root),
        gbbq=str(gbbq),
        output_dir=str(tmp_path / "screens"),
        as_of=None,
    )
    out, err = capture()

    assert run_screen_command(args, stdout=out, stderr=err) == 0, err.getvalue()
    assert "评估日：2024-03-25" in out.getvalue(), out.getvalue()
    assert "最近一个齐全的交易日" in out.getvalue()


def test_a_stub_tail_is_skipped_and_the_skipped_days_are_named_in_the_log():
    """末根残缺时要回退，且**把跳过的日子连同只数印出来**——否则回退是静默的。

    本机真实数据的末根就是这个样子（4,718 只 → 5 只），故这条不是假想。
    """

    dates = pd.bdate_range("2024-01-02", periods=22)
    width = 100
    data = [[1.0] * width for _ in range(21)] + [[1.0] * 5 + [float("nan")] * 95]
    panel = Panel(
        {
            "close": pd.DataFrame(
                data, index=dates, columns=[f"sh{600000 + i}" for i in range(width)]
            )
        }
    )
    out, err = capture()

    got = _resolve_as_of(panel, out, err)

    assert got == dates[20].date(), "应回退到残桩之前那个交易日"
    assert "跳过更晚的 1 个交易日" in out.getvalue()
    assert "只有 5 只" in out.getvalue()


def test_a_screen_whose_load_collapsed_fails_instead_of_writing_a_watchlist(tmp_path):
    """跳过率过高时选股必须以 ``1`` 收场——它**还会覆盖自选股**，这条尤其要紧。

    门与回测那道共用同一个常数（:data:`MAX_FAILURE_RATE`）。为什么选股也非要这道门：一张从
    四分之一市场上算出来的名单，与从满市场算出来的**长得一模一样**——候选、排序、CSV 的
    形状都不变，故它只能靠退出码说话。而定时任务没人在看，退出码是唯一能拦住「拿残缺名单
    覆盖通达信自选股」的东西（自选股那份是**破坏性**的，每天都盖掉前一天）。
    """
    from mbt.report import WATCHLIST_NAME

    root, gbbq = make_dataroot(tmp_path, symbol="sh600000")
    index = pd.bdate_range("2024-01-02", periods=80)
    for symbol in ("sh600001", "sh600002", "sh600003"):
        # 一天跳 +400%，越出任何板块的涨跌幅——真实数据里「数据不可信」那批就是这个形状。
        prices = [1000] * 40 + [5000] * 40
        write_day(
            root / symbol[:2] / "lday" / f"{symbol}.day",
            [
                (int(f"{stamp:%Y%m%d}"), price, price, price, price, 0.0, 1000)
                for stamp, price in zip(index, prices, strict=True)
            ],
        )
    watchlist_dir = tmp_path / "自选股"
    args = screen_args(
        tdx_root=str(root),
        gbbq=str(gbbq),
        output_dir=str(tmp_path / "screens"),
        watchlist_dir=str(watchlist_dir),
    )
    out, err = capture()

    code = run_screen_command(args, stdout=out, stderr=err)

    assert code == 1, f"四只里三只读不出来，不该以 0 收场：{out.getvalue()}"
    assert "跳过率" in err.getvalue()
    assert not (watchlist_dir / WATCHLIST_NAME).exists(), "残缺名单不该覆盖自选股"


def test_the_watchlist_export_writes_bare_codes_in_rank_order(tmp_path):
    """自选股文件的内容 = 候选清单的**裸代码**，顺序一致（从优到劣）。

    文件名来自 ``mbt.report.WATCHLIST_NAME``——它是产物契约的一部分，故这里也走那个常量，
    免得改个名字这条测试还绿着。
    """
    from mbt.report import WATCHLIST_NAME

    root, gbbq = make_dataroot(tmp_path, periods=60)
    watchlist_dir = tmp_path / "自选股"
    args = screen_args(
        tdx_root=str(root),
        gbbq=str(gbbq),
        output_dir=str(tmp_path / "screens"),
        watchlist_dir=str(watchlist_dir),
    )
    out, err = capture()

    assert run_screen_command(args, stdout=out, stderr=err) == 0, err.getvalue()

    run_dir = next((tmp_path / "screens").iterdir())
    ranked = pd.read_csv(run_dir / "candidates.csv")["symbol"].tolist()
    target = watchlist_dir / WATCHLIST_NAME
    assert target.read_text(encoding="utf-8").splitlines() == [name[2:] for name in ranked]
    assert str(target) in out.getvalue()


def test_an_empty_watchlist_is_written_but_the_loss_is_announced(tmp_path):
    """没有候选时文件照写，但**必须嚷**——导入一份空自选股会把通达信那边的清空。"""
    from mbt.report import WATCHLIST_NAME

    out, err = capture()

    _write_watchlist([], tmp_path, out, err)

    assert (tmp_path / WATCHLIST_NAME).read_text(encoding="utf-8") == ""
    assert "没有候选" in err.getvalue()
    assert "清空" in err.getvalue()
