"""命令行：把库能力包成两条日常命令（票据 #11）。

```
mbt backtest --strategy mypkg.strategies:BuyAndHold --start 2015-08-01 \
             --cash 100000 --commission 0.0003 --commission-mode all_in \
             --output-dir D:\\runs
mbt screen   --symbols-file my_universe.txt --as-of 2026-09-11 --output-dir D:\\screens
```

## 两条硬规矩

**一、CLI 是库入口的薄映射。** 全部逻辑在 :func:`run_backtest_command` /
:func:`run_screen_command` 这两个**纯函数**里：不收 ``sys.argv``、不调 ``sys.exit``、
不直接写标准输出（由参数注入）。``main`` 只做三件事——解析参数、转交、返回退出码。测试因此
可以直接调用它们，**不必靠 subprocess 硬凑**（AC 明令）。

**二、默认区间取「跑得对」的那个窗口。** ``--start`` 默认 **2015-08-01**，即**费用口径**
全可查的最早日期（沪主板过户费的覆盖起点，各板块里最晚的一个）。不默认「全历史」是因为实测
37% 的股票数据起于 1997–2014，那段区间**涨跌幅查得到、费用查不到**，一成交就报
``RuleTableError``。要更早的区间请显式传 ``--start``——那是你的选择，我们如实报错而不替你截断。

## 退出码

- ``0``：跑完了，且有可用的结果（**含「有个别标的被跳过」**）；
- ``1``：数据或运行期错误，或**跳过率超过 :data:`MAX_FAILURE_RATE`**——「几乎全跳过」不该被
  脚本当成成功；
- ``2``：参数用法错误（``argparse`` 的默认）。
"""

from __future__ import annotations

import argparse
import importlib
import sys
from pathlib import Path

import pandas as pd

from mbt.data import TdxDataSource, load_universe_data, slice_markets, stock_symbols
from mbt.report import DEFAULT_BENCHMARK_SYMBOL, write_run_artifacts
from mbt.screen import SCREEN_FIELDS, momentum_screen
from mbt.universe import UniverseRules

#: 费用口径全可查的最早日期（沪主板过户费自该日起按成交金额计）。见模块说明。
DEFAULT_START = "2015-08-01"

#: 跳过率超过它即判失败——「几乎全跳过」不该被脚本当成成功。
MAX_FAILURE_RATE = 0.30


def build_parser() -> argparse.ArgumentParser:
    """构造参数解析器。**独立成函数**，便于测试断言用法错误的退出码。"""
    parser = argparse.ArgumentParser(prog="mbt", description="A 股策略回测与选股工具")
    sub = parser.add_subparsers(dest="command", required=True)

    backtest = sub.add_parser("backtest", help="跑一次组合回测")
    backtest.add_argument(
        "--strategy",
        required=True,
        help='策略的导入路径，形如 "mypkg.strategies:BuyAndHold"',
    )
    backtest.add_argument("--start", default=DEFAULT_START, help=f"起始日（默认 {DEFAULT_START}）")
    backtest.add_argument("--end", default=None, help="结束日（默认到数据末根）")
    backtest.add_argument("--cash", type=float, default=100_000.0, help="期初资金")
    backtest.add_argument("--max-positions", type=int, default=None, help="最大持仓只数")
    backtest.add_argument("--commission", type=float, default=0.0, help="手续费率")
    backtest.add_argument(
        "--commission-mode",
        choices=("all_in", "net"),
        default=None,
        help="手续费口径：all_in（全佣）/ net（净佣）。费率 > 0 时必填",
    )
    backtest.add_argument("--commission-min", type=float, default=0.0, help="单笔最低手续费")
    backtest.add_argument("--slippage", type=float, default=0.0, help="滑点比例")
    backtest.add_argument("--benchmark", default=DEFAULT_BENCHMARK_SYMBOL, help="基准标的")
    backtest.add_argument("--tdx-root", required=True, help="通达信 vipdoc 根目录")
    backtest.add_argument("--gbbq", required=True, help="权息文件 gbbq 的路径")
    backtest.add_argument("--output-dir", required=True, help="产物落盘目录（必须显式给出）")
    backtest.add_argument("--limit", type=int, default=None, help="只取前 N 个标的（试跑用）")
    backtest.add_argument("--symbols-file", default=None, help="只跑文件里列出的标的（每行一个）")
    backtest.add_argument(
        "--param",
        action="append",
        default=[],
        metavar="NAME=VALUE",
        help="策略参数，可重复，如 --param fast=5",
    )

    screen = sub.add_parser("screen", help="在指定评估日跑一次选股")
    screen.add_argument("--as-of", required=True, help="评估日，必须是交易日")
    screen.add_argument("--top-n", type=int, default=10, help="取前 N 名")
    screen.add_argument("--tdx-root", required=True, help="通达信 vipdoc 根目录")
    screen.add_argument("--gbbq", required=True, help="权息文件 gbbq 的路径")
    screen.add_argument("--output-dir", required=True, help="产物落盘目录（必须显式给出）")
    screen.add_argument("--limit", type=int, default=None, help="只取前 N 个标的（试跑用）")
    screen.add_argument("--symbols-file", default=None, help="只跑文件里列出的标的（每行一个）")

    return parser


def run_backtest_command(args, *, stdout=sys.stdout, stderr=sys.stderr) -> int:
    """执行一次回测命令，返回退出码。**纯函数**：不收 ``sys.argv``、不调 ``sys.exit``。"""
    try:
        strategy = _import_strategy(args.strategy)
    except ValueError as exc:
        print(f"错误：{exc}", file=stderr)
        return 1

    try:
        params = parse_params(args.param)
    except ValueError as exc:
        print(f"错误：{exc}", file=stderr)
        return 1

    symbols = _select_symbols(args, stderr)
    if symbols is None:
        return 1
    if not symbols:
        print("错误：选出的标的为空——请检查 --symbols-file 或 --limit", file=stderr)
        return 1

    print(f"取数：{len(symbols)} 个股票候选（已剔除指数、基金、可转债）", file=stdout)
    loaded = load_universe_data(symbols, tdx_root=args.tdx_root, gbbq_path=args.gbbq, rules=None)

    # 区间由**此处**落实：`--start` / `--end` 原先被解析了却没接上，回测因此永远跑全历史
    # ——而 README 还把默认值当作生效的行为写了理由。切片放在库里（可测、可复用）。
    try:
        loaded = slice_markets(loaded.markets, args.start, args.end)
    except ValueError as exc:
        print(f"错误：{exc}", file=stderr)
        return 1

    _report_loading(loaded, stdout, stderr)

    if not loaded.markets:
        print("错误：没有任何标的可用，回测无从做起", file=stderr)
        return 1
    if loaded.failure_rate > MAX_FAILURE_RATE:
        print(
            f"错误：跳过率 {loaded.failure_rate:.0%} 超过阈值 {MAX_FAILURE_RATE:.0%}，"
            f"结果不足以采信",
            file=stderr,
        )
        return 1

    from mbt.backtest import run_portfolio_backtest  # 延迟导入：backtrader 只在真跑时才需要

    try:
        result = run_portfolio_backtest(
            list(loaded.markets),
            strategy,
            cash=args.cash,
            max_positions=args.max_positions,
            commission=args.commission,
            commission_min=args.commission_min,
            commission_mode=args.commission_mode,
            slippage=args.slippage,
            universe_rules=UniverseRules(),
            **params,
        )
    except Exception as exc:  # noqa: BLE001
        print(f"错误：回测失败——{type(exc).__name__}: {exc}", file=stderr)
        return 1

    benchmark = _load_benchmark(args, stderr)

    run_dir = write_run_artifacts(
        result,
        output_dir=args.output_dir,
        strategy=strategy,
        strategy_params=params,
        cash=args.cash,
        max_positions=args.max_positions,
        costs={
            "commission": args.commission,
            "commission_min": args.commission_min,
            "commission_mode": args.commission_mode,
            "slippage": args.slippage,
        },
        universe={"rule": "出厂设定", "candidates": len(symbols), "loaded": len(loaded.markets)},
        benchmark_prices=benchmark,
        benchmark_symbol=args.benchmark,
        snapshot_paths=_snapshot_paths(loaded, args),
    )

    _report_result(result, run_dir, stdout)
    return 0


def run_screen_command(args, *, stdout=sys.stdout, stderr=sys.stderr) -> int:
    """执行一次选股命令，返回退出码。**纯函数**，理由同上。"""
    import datetime as dt

    try:
        as_of = dt.date.fromisoformat(args.as_of)
    except ValueError:
        print(f"错误：--as-of {args.as_of!r} 不是合法日期（应形如 2026-09-11）", file=stderr)
        return 1

    symbols = _select_symbols(args, stderr)
    if symbols is None:
        return 1
    if not symbols:
        print("错误：选出的标的为空——请检查 --symbols-file 或 --limit", file=stderr)
        return 1

    print(f"取数：{len(symbols)} 个股票候选（已剔除指数、基金、可转债）", file=stdout)
    loaded = load_universe_data(symbols, tdx_root=args.tdx_root, gbbq_path=args.gbbq, rules=None)
    _report_loading(loaded, stdout, stderr)

    if not loaded.markets:
        print("错误：没有任何标的可用，选股无从做起", file=stderr)
        return 1

    from mbt.data import assemble_panel
    from mbt.universe import build_universe

    try:
        panel = assemble_panel(list(loaded.markets), SCREEN_FIELDS)
        pool = build_universe(list(loaded.markets), rules=UniverseRules())
        # 默认规则取自**库**（`momentum_screen`），CLI 只指名它——「不含独立业务逻辑」。
        result = momentum_screen(window=20, top_n=args.top_n).apply(
            panel, as_of=as_of, universe_mask=pool
        )
        candidates = result.candidates(as_of)
    except Exception as exc:  # noqa: BLE001
        print(f"错误：选股失败——{type(exc).__name__}: {exc}", file=stderr)
        return 1

    try:
        run_dir = _write_screen_artifacts(result, candidates, args, as_of, loaded, panel)
    except Exception as exc:  # noqa: BLE001
        # 落盘失败（如目录已存在）也必须以**退出码**收场，而不是把原始异常抛给调用者。
        print(f"错误：写入选股产物失败——{type(exc).__name__}: {exc}", file=stderr)
        return 1
    _report_candidates(candidates, as_of, run_dir, stdout)
    return 0


def main(argv=None) -> int:
    """入口：解析参数、转交、返回退出码。**这里不写业务逻辑。**

    **把当前工作目录放进 ``sys.path``**：安装后的入口脚本只会把「脚本所在目录」加进去，
    不会加当前目录——于是 `--strategy mypkg.strategies:BuyAndHold` 在你自己的项目目录里
    反而导入不到。用户是站在自己的策略目录里敲这条命令的，所以把当前目录加在前头，
    与 ``python -m``、``pytest`` 的行为一致。
    """
    here = str(Path.cwd())
    if here not in sys.path:
        sys.path.insert(0, here)

    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command == "backtest":
        return run_backtest_command(args)
    return run_screen_command(args)


def _entry_point() -> None:
    """``console_scripts`` 的落点：把返回值变成进程退出码。"""
    sys.exit(main())


# --- 参数与数据的搬运 ----------------------------------------------------------


def parse_params(items) -> dict:
    """把 ``--param name=value`` 解析成 dict。

    值按字面量解析（``int`` → ``float`` → ``bool`` → 字符串）；解析不了就报错**并指出是哪一个**
    ——策略参数错一个，回测结果就完全是另一回事，不能含糊过去。
    """
    params: dict = {}
    for item in items:
        name, _, raw = item.partition("=")
        name = name.strip()
        if not name or not _:
            raise ValueError(f"--param {item!r} 格式不对，应形如 fast=5")
        params[name] = _literal(raw.strip())
    return params


def _literal(raw: str):
    lowered = raw.lower()
    if lowered in ("true", "false"):
        return lowered == "true"
    for cast in (int, float):
        try:
            return cast(raw)
        except ValueError:
            continue
    return raw


def _import_strategy(path: str):
    """按 ``module:Class`` 取回策略类。

    与 #10 的元数据用**同一种写法**——`describe_strategy` 记的就是 ``module.ClassName``，
    故 CLI 指名的策略与产物里记下的策略天然一致，复现时不必再翻译一次。
    """
    module_name, separator, class_name = path.partition(":")
    if not separator:
        raise ValueError(f'--strategy {path!r} 应形如 "mypkg.strategies:BuyAndHold"')
    try:
        module = importlib.import_module(module_name)
    except ImportError as exc:
        raise ValueError(
            f"导入 {module_name!r} 失败（{exc}）。请确认它在当前环境里可导入——"
            f"通常意味着策略所在目录不在 PYTHONPATH 上。"
        ) from exc
    try:
        return getattr(module, class_name)
    except AttributeError as exc:
        raise ValueError(f"{module_name!r} 里没有 {class_name!r}") from exc


def _select_symbols(args, stderr) -> list[str] | None:
    """按 ``--symbols-file`` / ``--limit`` 选出要跑的股票。出错时返回 ``None``。"""
    if args.symbols_file:
        path = Path(args.symbols_file)
        if not path.is_file():
            print(f"错误：--symbols-file {path} 不存在", file=stderr)
            return None
        listed = [line.strip() for line in path.read_text(encoding="utf-8").splitlines()]
        symbols = [s for s in listed if s and not s.startswith("#")]
        if not symbols:
            print(f"错误：{path} 里没有可用的标的（空行与 # 注释会被跳过）", file=stderr)
            return None
        return symbols

    try:
        scanned = TdxDataSource(args.tdx_root).symbols()
    except Exception as exc:  # noqa: BLE001
        print(f"错误：扫描 {args.tdx_root} 失败——{type(exc).__name__}: {exc}", file=stderr)
        return None

    symbols = stock_symbols(scanned)
    return symbols[: args.limit] if args.limit else symbols


def _load_benchmark(args, stderr):
    """读基准的**收盘价序列**。读不到就**警告并继续**——没有基准的回测仍可解读，只是少了参照。

    注意取的是 ``["close"]``：``daily()`` 返回的是**字段宽表**（六列），而基准这里要的是
    一条价格序列。传整张表进去会让它在算年化时以一句难懂的 pandas 报错收场。
    """
    try:
        return TdxDataSource(args.tdx_root).daily(args.benchmark)["close"]
    except Exception as exc:  # noqa: BLE001
        print(
            f"警告：基准 {args.benchmark} 读不到（{type(exc).__name__}: {exc}），"
            f"本次不产出基准对比",
            file=stderr,
        )
        return None


def _snapshot_paths(loaded, args) -> list[Path]:
    """本次实际读到的文件：各标的的 ``.day`` 加上权息文件——数据快照摘要要覆盖全部。"""
    source = TdxDataSource(args.tdx_root)
    paths = [source.path_for(market.symbol) for market in loaded.markets]
    paths.append(Path(args.gbbq))
    return paths


# --- 输出 ---------------------------------------------------------------------


def _report_loading(loaded, stdout, stderr) -> None:
    """汇报取数结果。**跳过必须被说出来**，否则「跳过了什么」会变成谜（ADR-0005）。"""
    print(f"  成功 {len(loaded.markets)}，跳过 {len(loaded.skipped)}", file=stdout)
    for kind, count in sorted(loaded.count_by_kind().items()):
        print(f"    跳过·{kind}：{count}", file=stdout)
    if loaded.skipped:
        print("  被跳过的标的（最多列 10 个）：", file=stderr)
        for item in loaded.skipped[:10]:
            print(f"    {item.symbol}  [{item.kind}] {item.detail}", file=stderr)


def _report_result(result, run_dir, stdout) -> None:
    from mbt.metrics import compute_metrics

    metrics = compute_metrics(
        result.equity_curve,
        result.trades,
    )
    print("\n结果：", file=stdout)
    print(f"  期末总资产    {result.final_value:,.2f}", file=stdout)
    print(f"  区间总收益    {metrics.total_return:+.2%}", file=stdout)
    print(f"  年化收益      {metrics.annual_return:+.2%}", file=stdout)
    print(f"  夏普          {metrics.sharpe:.3f}", file=stdout)
    print(f"  最大回撤      {metrics.max_drawdown:.2%}", file=stdout)
    print(f"  成交笔数      {len(result.trades)}（平仓 {metrics.closed_trades}）", file=stdout)
    for note in metrics.notes:
        print(f"  注：{note}", file=stdout)
    print(f"\n产物：{run_dir}", file=stdout)


def _report_candidates(candidates, as_of, run_dir, stdout) -> None:
    print(f"\n{as_of} 的候选集（{len(candidates)} 个，从优到劣）：", file=stdout)
    for position, symbol in enumerate(candidates, start=1):
        print(f"  {position:>3}. {symbol}", file=stdout)
    print(f"\n产物：{run_dir}", file=stdout)


def _write_screen_artifacts(result, candidates, args, as_of, loaded, panel) -> Path:
    """选股产物：候选清单 + 元数据。**复用 #10 的目录约定**，不新造一套。"""
    import datetime as dt
    import json

    from mbt.report import data_snapshot, git_version

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    stamp = dt.datetime.now()
    run_dir = output_dir / f"{stamp:%Y%m%d}-{stamp:%H%M%S}-screen"
    if run_dir.exists():
        raise ValueError(f"产物目录已存在：{run_dir}——不静默覆盖，请换名字或清掉它")
    run_dir.mkdir(parents=True)

    frame = pd.DataFrame({"symbol": candidates, "rank": range(1, len(candidates) + 1)})
    frame.to_csv(run_dir / "candidates.csv", index=False)

    metadata = {
        "as_of": as_of.isoformat(),
        "candidates": len(candidates),
        "loaded": len(loaded.markets),
        "skipped": [{"symbol": item.symbol, "kind": item.kind} for item in loaded.skipped],
        "universe": "出厂设定（排除次新股，纳入四个板块；ST 排除目前不生效）",
        "rule": "按 20 日动量排序取前 N",
        "data_snapshot": data_snapshot(_snapshot_paths(loaded, args)),
        "git": git_version(),
    }
    (run_dir / "run.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False, sort_keys=True), encoding="utf-8"
    )
    return run_dir
