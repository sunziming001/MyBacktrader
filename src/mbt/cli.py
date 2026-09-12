"""命令行：把库能力包成三条日常命令（票据 #9、#11）。

```
mbt backtest --strategy mypkg.strategies:BuyAndHold --start 2015-08-01 \
             --cash 100000 --commission 0.0003 --commission-mode all_in \
             --output-dir D:\\runs
mbt screen   --symbols-file my_universe.txt --as-of 2026-09-11 --output-dir D:\\screens
mbt update   --tdx-root D:\\Tools\\tdx\\vipdoc [--baseline b.csv --save-baseline b.csv]
```

## 三条硬规矩

**一、CLI 是库入口的薄映射。** 全部逻辑在 :func:`run_backtest_command` /
:func:`run_screen_command` / :func:`run_update_command` 这三个**纯函数**里：不收 ``sys.argv``、
不调 ``sys.exit``、不直接写标准输出（由参数注入）。``main`` 只做三件事——解析参数、转交、
返回退出码。测试因此可以直接调用它们，**不必靠 subprocess 硬凑**（AC 明令）。

**二、默认区间取「跑得对」的那个窗口。** ``--start`` 默认 **2015-08-01**，即**费用口径**
全可查的最早日期（沪主板过户费的覆盖起点，各板块里最晚的一个）。不默认「全历史」是因为实测
37% 的股票数据起于 1997–2014，那段区间**涨跌幅查得到、费用查不到**，一成交就报
``RuleTableError``。要更早的区间请显式传 ``--start``——那是你的选择，我们如实报错而不替你截断。

**三、输出必须能在中文 Windows 控制台上打印。** 控制台是 GBK，故**不输出 emoji 等非 GBK 字符**
（实跑时 ``⚠️`` 曾让整个命令崩在 ``UnicodeEncodeError`` 上，而那与数据无关）。

## 退出码

- ``0``：跑完了，且有可用的结果（**含「有个别标的被跳过」**）；
- ``1``：数据或运行期错误，或**跳过率超过 :data:`MAX_FAILURE_RATE`**——「几乎全跳过」不该被
  脚本当成成功；``update`` 子命令在**有回补/修正**时也返回 ``1``（那意味着上次的回测结果作废）；
- ``2``：参数用法错误（``argparse`` 的默认）。
"""

from __future__ import annotations

import argparse
import importlib
import sys
from pathlib import Path

import pandas as pd

from mbt.data import (
    TdxDataSource,
    load_financials,
    load_listing_dates,
    load_universe_data,
    non_loss_mask,
    slice_markets,
    stock_symbols,
)
from mbt.data.errors import MarketDataError
from mbt.report import DEFAULT_BENCHMARK_SYMBOL, write_run_artifacts
from mbt.screen import SCREEN_FIELDS, momentum_screen
from mbt.universe import UniverseRules, combine_masks

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
    backtest.add_argument(
        "--cw-root",
        default=None,
        help="财务数据目录（vipdoc/cw）。用 --non-loss 时必填",
    )
    backtest.add_argument(
        "--non-loss",
        action="store_true",
        help="启用「非亏损」过滤（按公告日的累计归母净利润；缺财报的标的排除）",
    )
    backtest.add_argument("--output-dir", required=True, help="产物落盘目录（必须显式给出）")
    backtest.add_argument(
        "--master",
        default=None,
        help="证券主表 base.dbf 的路径（提供则次新股门槛按**真实上市日**算）",
    )
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
    screen.add_argument(
        "--cw-root",
        default=None,
        help="财务数据目录（vipdoc/cw）。用 --non-loss 时必填",
    )
    screen.add_argument(
        "--non-loss",
        action="store_true",
        help="启用「非亏损」过滤（按公告日的累计归母净利润；缺财报的标的排除）",
    )
    screen.add_argument("--output-dir", required=True, help="产物落盘目录（必须显式给出）")
    screen.add_argument(
        "--master",
        default=None,
        help="证券主表 base.dbf 的路径（提供则次新股门槛按**真实上市日**算）",
    )
    screen.add_argument("--limit", type=int, default=None, help="只取前 N 个标的（试跑用）")
    screen.add_argument("--symbols-file", default=None, help="只跑文件里列出的标的（每行一个）")

    update = sub.add_parser("update", help="检查本地数据有没有变化（手动触发，不设定时任务）")
    update.add_argument("--tdx-root", required=True, help="通达信 vipdoc 根目录")
    update.add_argument(
        "--baseline",
        default=None,
        help="上次的边界清单（CSV）。不给则建立基线；给了则与之比较并报出回补/修正",
    )
    update.add_argument("--limit", type=int, default=None, help="只检查前 N 个标的（试跑用）")
    update.add_argument("--symbols-file", default=None, help="只检查文件里列出的标的（每行一个）")
    update.add_argument(
        "--save-baseline",
        default=None,
        help="把本次的边界清单写到这个路径，供下次 --baseline 用",
    )

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

    if args.non_loss and not args.cw_root:
        print("错误：--non-loss 需要 --cw-root 指定财务数据目录（vipdoc/cw）", file=stderr)
        return 1

    from mbt.backtest import run_portfolio_backtest  # 延迟导入：backtrader 只在真跑时才需要
    from mbt.data.fundamental import load_financials, non_loss_mask

    extra_mask = None
    if args.non_loss:
        try:
            financials = load_financials(args.cw_root)
            closes = pd.DataFrame(
                {market.symbol: market.prices["close"] for market in loaded.markets}
            )
            extra_mask = non_loss_mask(financials, closes.index, closes.columns)
            # 汇总要说得有信息量：`any()` 数的是「有过任一天通过的标的」，那会让人以为没筛掉
            # 任何东西。真正要看的是**每个评估日**有多少标的通过。
            per_day = extra_mask.sum(axis=1)
            print(
                f"基本面过滤：每个评估日平均 {per_day.mean():.0f}/{len(closes.columns)} 个标的通过"
                f"（首个评估日 {int(per_day.iloc[0])} 个）",
                file=stdout,
            )
        except Exception as exc:  # noqa: BLE001
            print(f"错误：财务数据读取失败——{type(exc).__name__}: {exc}", file=stderr)
            return 1

    # 主表缺失或某标的上市日未知 → 回退到「本地行情根数」的近似口径。**退化必须可见**，
    # 否则「用的是哪个口径」就成了谜。
    listing_dates = None
    if args.master:
        try:
            every = load_listing_dates(
                args.master, symbols=[market.symbol for market in loaded.markets]
            )
            # **只留本次用到的标的**：`load_listing_dates` 给的是整表（本机 8,088 条），
            # 直接报它会把「本次有几个标的有上市日」说成一个无意义的数字。
            listing_dates = {
                market.symbol: every[market.symbol]
                for market in loaded.markets
                if market.symbol in every
            }
            print(
                f"证券主表：{len(listing_dates)}/{len(loaded.markets)} 个标的有真实上市日"
                f"（其余回退到行情根数）",
                file=stdout,
            )
        except Exception as exc:  # noqa: BLE001
            print(f"错误：主表读取失败——{type(exc).__name__}: {exc}", file=stderr)
            return 1
    else:
        print("未提供 --master，次新股门槛用「本地行情根数」的近似口径", file=stdout)

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
            universe_rules=UniverseRules(extra_mask=extra_mask),
            listing_dates=listing_dates,
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
        listing_dates = None
        if args.master:
            every = load_listing_dates(args.master, symbols=[m.symbol for m in loaded.markets])
            # 只留本次用到的标的（理由同 backtest 那条注释）。
            listing_dates = {m.symbol: every[m.symbol] for m in loaded.markets if m.symbol in every}
            print(
                f"证券主表：{len(listing_dates)}/{len(loaded.markets)} 个标的有真实上市日"
                f"（其余回退到行情根数）",
                file=stdout,
            )
        else:
            print("未提供 --master，次新股门槛用「本地行情根数」的近似口径", file=stdout)
        pool = build_universe(
            list(loaded.markets), rules=UniverseRules(), listing_dates=listing_dates
        )
        masks = [pool]
        # AC 要求「非亏损」过滤**同时**接入股票池与选股规则。选股侧走的是 `Screen.apply` 的
        # `universe_mask`——它只收一份掩码，故在这里把股票池与基本面**合成**一份
        # （`combine_masks` 就是为了这件事存在的，不然它会是个没有生产调用者的测试专用品）。
        if args.non_loss:
            if not args.cw_root:
                print("错误：--non-loss 需要 --cw-root 指定财务数据目录", file=stderr)
                return 1
            financials = load_financials(args.cw_root)
            masks.append(
                non_loss_mask(
                    financials, panel.fields["close"].index, panel.fields["close"].columns
                )
            )
            print("基本面过滤：已按公告日叠加「非亏损」条件", file=stdout)

        result = momentum_screen(window=20, top_n=args.top_n).apply(
            panel, as_of=as_of, universe_mask=combine_masks(*masks)
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


def run_update_command(args, *, stdout=sys.stdout, stderr=sys.stderr) -> int:
    """检查数据有没有变化，返回退出码。**纯函数**，理由同另两条命令。

    退出码：**有回补/修正 → 1**（那意味着上次的回测结果作废了，脚本应当知道），否则 0。
    本命令的**唯一价值**就是回答「我的数据变了没有、变了要不要重跑」，而「有修正」正是
    「要重跑」的信号。
    """
    from mbt.data.updates import (
        APPENDED,
        INSIDE_GAP,
        MISSING,
        NEW,
        REVISED,
        TAIL_GAP,
        UNCHANGED,
        check_updates,
        load_boundaries,
        save_boundaries,
    )

    symbols = _select_symbols(args, stderr)
    if symbols is None:
        return 1

    previous = None
    if args.baseline:
        try:
            previous = load_boundaries(args.baseline)
        except MarketDataError as exc:
            print(f"错误：{exc}", file=stderr)
            return 1

    try:
        report = check_updates(symbols, tdx_root=args.tdx_root, previous=previous)
    except MarketDataError as exc:
        print(f"错误：{exc}", file=stderr)
        return 1

    counts = report.counts()
    print(
        f"检查 {sum(counts.values())} 个标的（市场交易日 {report.calendar_days} 天）：", file=stdout
    )
    for kind in (UNCHANGED, APPENDED, REVISED, MISSING, NEW):
        if kind in counts:
            print(f"  {kind}：{counts[kind]}", file=stdout)
    if report.compared_with is None:
        print("  首次运行——已建立基线，本次无可比对象", file=stdout)
    if report.unreadable:
        # 取不到数据 ≠ 没变化。必须说出来，否则它会静默地从基线里消失。
        print(f"  取不到数据（未检查）：{len(report.unreadable)}", file=stdout)
        for symbol in report.unreadable[:10]:
            print(f"    {symbol}", file=stderr)

    if args.limit or args.symbols_file:
        # 市场日历来自**被检查的**那批标的；只查一部分会让两类缺口都失真
        # （区间内缺口看不见、尾部空缺恒为零）。换个说法：这会削弱本命令的判据。
        print(
            "  警告：--limit / --symbols-file 使市场日历不完整——"
            "区间内缺口会看不见、尾部空缺会偏小。要准确判定请检查全市场。",
            file=stderr,
        )

    inside = report.gaps_by_kind(INSIDE_GAP)
    tails = report.gaps_by_kind(TAIL_GAP)
    if inside:
        print(f"\n区间内无成交 {len(inside)} 处（前 5）：", file=stdout)
        for gap in inside[:5]:
            print(
                f"  {gap.symbol}: 自 {gap.start} 至 {gap.end} 无成交，"
                f"共缺 {gap.missing} 个交易日，{gap.resumed} 恢复",
                file=stdout,
            )
    if tails:
        print(f"\n尾部空缺 {len(tails)} 个标的（前 5）：", file=stdout)
        for gap in sorted(tails, key=lambda item: -item.missing)[:5]:
            print(
                f"  {gap.symbol}: 最后一根 {gap.end}，其后缺 {gap.missing} 个交易日",
                file=stdout,
            )
        print(
            "  注：尾部空缺**无法**在本地数据里区分为停牌还是退市——只报事实，不猜。",
            file=stdout,
        )

    if report.revised:
        print(f"\n注：有回补/修正的标的 {len(report.revised)} 个（前 10）：", file=stdout)
        for symbol in report.revised[:10]:
            print(f"  {symbol}", file=stdout)
        print("  这些标的的**旧回测结果不再可信**，请重跑。", file=stdout)

    if args.save_baseline:
        try:
            path = save_boundaries(args.save_baseline, report.boundaries)
        except OSError as exc:
            print(f"错误：写入基线失败——{exc}", file=stderr)
            return 1
        print(f"\n基线已写入：{path}", file=stdout)

    return 1 if report.needs_rerun else 0


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
    if args.command == "update":
        return run_update_command(args)
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
