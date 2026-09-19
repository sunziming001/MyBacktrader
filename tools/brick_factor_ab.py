"""砖型换因子（两项 → 四项）到底改了什么、改得好不好（票据 #98 的收尾测量）。

**为什么不是跑两次回测。** 那条路上有一个已知的机械放大：持仓分配是**每笔固定 20 万**，
资金不够就用剩下的——净值缩水之后每一笔都在赌大半个账户，于是**任何**因子集都会跑出接近
归零的结果（实测一轮：100 万 → 0.14 元）。那个结果分不出「因子好不好」，只说明那条口径
自己会归零。故这里量的是**选出来的名字之后怎么走**——按天配对、不含费用与择时：

.. code-block:: text

    信号在 T 收盘算出 → 成交在 T+1 开盘（引擎的既有口径）→ 看 T+1+k 的收盘

**为什么按天配对。** 两条过滤器没动，故两个因子集在**每一天选出的候选集合完全相同**，
只有次序不同；又因为引擎按「取前 N」截断，差异只出现在那 N 个名额上。于是可比的量是：

.. code-block:: text

    被换进来的那几只（new − old）  比上年      被换出去的那几只（old − new）

同一个交易日、同一份市场环境、同一批候选，差别只在因子给出的次序——这是最干净的归因。

用法::

    $env:MBT_TDX_ROOT = "D:\\Tools\\new_tdx\\vipdoc"
    $env:MBT_TDX_GBBQ = "D:\\Tools\\new_tdx\\T0002\\hq_cache\\gbbq"
    .venv\\Scripts\\python.exe tools\\brick_factor_ab.py --top-n 5
    .venv\\Scripts\\python.exe tools\\brick_factor_ab.py --top-n 5 --compare HEAD

``--compare`` 给出「旧因子」从哪个 git 版本取（默认 ``HEAD`` 的**父提交**，即换因子之前
那个状态）。用 git 里的那份 ``screen.py`` 而不是复制一段旧代码，是为了确保量的是**真的
旧实现**，不是我记得的旧实现。

判读的纪律（与 ``tools/oom_loop.py`` 一样，本脚本只给数，不下结论）：

- 报的是**均值 / 标准误**，不是「涨了多少」。均值相对标准误只有 1 上下的，就是噪声。
- 逐笔的绝对值小，别拿它跟净值涨幅比——中间隔着费用与仓位（见 ``README`` 那条）。
- 多个持有期一起看：只在某一个 k 上成立、换个 k 就反号的，不要当成结论。
"""

from __future__ import annotations

import argparse
import importlib.util
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

#: 要看几个持有期（交易日）。策略实测持有期中位约 5 个自然日，故短的那几个为主。
HORIZONS = (1, 3, 5, 10, 20)


def environment() -> tuple[str, str]:
    """按仓库纪律从环境变量取路径：**没有默认值**，指错就当场说清是哪一个。"""
    root = os.environ.get("MBT_TDX_ROOT")
    gbbq = os.environ.get("MBT_TDX_GBBQ")
    missing = [
        name for name, value in (("MBT_TDX_ROOT", root), ("MBT_TDX_GBBQ", gbbq)) if not value
    ]
    if missing:
        raise SystemExit(f"缺少环境变量：{missing}（本脚本不猜路径，理由见 README）")
    return root, gbbq


def old_screen_module(rev: str):
    """把某个 git 版本上的 ``mbt/screen.py`` 导入成一个独立模块。

    **不是在测试里复制一段旧代码**：复制出来的那份只能证明「我记得的旧实现是这样」，而这里
    要回答的是「换掉的那一版到底什么样」。相对导入不用改——模块内用的是 ``mbt.*`` 绝对导入。
    """
    source = subprocess.run(
        ["git", "-C", str(REPO_ROOT), "show", f"{rev}:src/mbt/screen.py"],
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    if not source.stdout:
        raise SystemExit(f"取不到 {rev}:src/mbt/screen.py —— {source.stderr.strip()}")

    path = Path(tempfile.mkdtemp()) / f"old_screen_{rev.replace('/', '_')}.py"
    path.write_text(source.stdout, encoding="utf-8")
    spec = importlib.util.spec_from_file_location(f"old_screen_{rev}", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def load_panel(*, quality_bars: int, limit: int | None, progress=None):
    """按生产那条路取数并组装面板：与 ``mbt screen`` 逐项同口径。

    只做这一步（不复用 ``run_screen_command`` 的其余部分）：本脚本要的是**面板本身**，
    好在同一张面板上跑两套因子。
    """
    from mbt.data import backward_adjusted_markets
    from mbt.data.loader import load_universe_data, stock_symbols
    from mbt.data.panel import assemble_panel
    from mbt.data.tdx import TdxDataSource
    from mbt.screen import SCREEN_FIELDS

    root, gbbq = environment()
    scanned = TdxDataSource(root).symbols()
    symbols = stock_symbols(scanned)
    if limit:
        symbols = symbols[:limit]

    loaded = load_universe_data(
        symbols,
        tdx_root=root,
        gbbq_path=gbbq,
        rules=None,
        quality_bars=quality_bars or None,
        progress=progress,
    )
    adjusted = backward_adjusted_markets(loaded.markets)
    return loaded, assemble_panel(adjusted, SCREEN_FIELDS)


def universe_mask(panel, markets, *, listing_dates=None) -> pd.DataFrame:
    """砖型那个池子（它自己声明的板块范围 + 出厂门槛）。"""
    from mbt.screen import brick_screen
    from mbt.universe import UniverseRules, build_universe

    rules = UniverseRules(boards=brick_screen().boards)
    pool = build_universe(markets, rules=rules, listing_dates=listing_dates)
    reference = next(iter(panel.fields.values()))
    return pool.reindex(index=reference.index, columns=reference.columns, fill_value=False)


def picks_per_day(result, *, top_n: int) -> dict[pd.Timestamp, tuple[str, ...]]:
    """逐日的「取前 N」，**按分数从优到劣**。

    刻意不复用 ``ScreenResult.candidates``：那一个是给单日用的，而这里要扫过整段日历。
    排序口径与它一致（分数降序、同分按代码升序），故两者不会给出不同的清单。
    """
    selected = result.selected
    scores = result.scores
    out: dict[pd.Timestamp, tuple[str, ...]] = {}
    for day in selected.index:
        picked = [name for name in selected.columns if bool(selected.at[day, name])]
        if not picked:
            out[day] = ()
            continue
        row = scores.loc[day] if not scores.empty else None
        if row is None:
            picked.sort()
        else:
            picked.sort(key=lambda name: (-row[name] if row[name] == row[name] else np.inf, name))
        out[day] = tuple(picked[:top_n])
    return out


def forward_returns(panel, *, horizons=HORIZONS) -> dict[int, pd.DataFrame]:
    """``{k: 未来 k 个交易日的收益}``：进场取**次根开盘**，出场取第 k 根**收盘**。

    这与引擎的既有口径一致（信号在 T 收盘算出、成交在 T+1 开盘），而收益按**同一张复权
    面板**算——故送股除权不会假装成亏损（ADR-0003）。
    """
    open_ = panel["open"]
    close = panel["close"]
    entry = open_.shift(-1)  # T+1 开盘
    horizon = {}
    for k in horizons:
        exit_ = close.shift(-(1 + k))
        horizon[k] = exit_ / entry - 1.0
    return horizon


def paired_report(*, old: dict, new: dict, returns: dict[int, pd.DataFrame]) -> pd.DataFrame:
    """按天配对：被换进来的那批 vs 被换出去的那批，各自的未来收益。

    每天一行：那天换出的几只在未来 k 天的**均收益**，以及换进的那几只的均收益。天数才是
    样本量——同一天里那几只是同涨同跌的，逐笔当独立样本会把标准误算小（这是本项目反复
    警惕的那类「虚假信心」）。
    """
    rows = []
    for day in sorted(set(old) & set(new)):
        dropped = [name for name in old[day] if name not in new[day]]
        added = [name for name in new[day] if name not in old[day]]
        if not dropped and not added:
            continue
        row: dict[str, object] = {"date": day, "dropped": len(dropped), "added": len(added)}
        for k, frame in returns.items():
            if day not in frame.index:
                continue
            values = frame.loc[day]
            old_values = [values[name] for name in dropped if name in values.index]
            new_values = [values[name] for name in added if name in values.index]
            old_values = [v for v in old_values if v == v]
            new_values = [v for v in new_values if v == v]
            row[f"dropped_{k}"] = float(np.mean(old_values)) if old_values else np.nan
            row[f"added_{k}"] = float(np.mean(new_values)) if new_values else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def summarise(table: pd.DataFrame) -> None:
    """报每一天的配对差：``added − dropped``，以及它的均值 / 标准误 / t。"""
    print()
    print("按天配对的差（新换进来的 减去 被换出去的），单位：百分点")
    print("=" * 74)
    print(f"{'k':>3}  {'天数':>6}  {'均值':>9}  {'标准误':>8}  {'均值/标准误':>11}  {'胜率':>7}")
    for k in HORIZONS:
        column = table.get(f"added_{k}")
        if column is None:
            continue
        difference = (table[f"added_{k}"] - table[f"dropped_{k}"]).dropna()
        if difference.empty:
            print(f"{k:>3}  {'0':>6}  （没有可配对的格子）")
            continue
        mean = float(difference.mean())
        error = (
            float(difference.std(ddof=1) / np.sqrt(len(difference)))
            if len(difference) > 1
            else float("nan")
        )
        ratio = mean / error if error and error == error and error > 0 else float("nan")
        win = float((difference > 0).mean())
        print(
            f"{k:>3}  {len(difference):>6}  {mean * 100:>8.4f}%  {error * 100:>7.4f}%"
            f"  {ratio:>11.2f}  {win * 100:>6.1f}%"
        )
    print("=" * 74)
    print("判读：均值/标准误只有 1 上下的是噪声。多个 k 同号且量级稳定，才值得再看。")


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--top-n", type=int, default=5, help="每天取前几名（差异只出现在这里）")
    parser.add_argument(
        "--compare",
        default=None,
        help="从哪个 git 版本取「旧因子」（默认本提交的父提交，即换因子之前）",
    )
    parser.add_argument(
        "--quality-bars", type=int, default=1300, help="质检窗口（同 mbt screen 的默认）"
    )
    parser.add_argument("--limit", type=int, default=None, help="只取前 N 个标的（试跑用）")
    parser.add_argument("--out", default=None, help="把按天配对的表写成 CSV")
    args = parser.parse_args()

    from mbt.screen import brick_screen

    rev = args.compare or "HEAD^"
    module = old_screen_module(rev)
    print(f"新因子：{len(brick_screen().given_factors)} 项 {brick_screen().weights}")
    old_rule = module.brick_screen(top_n=args.top_n)
    print(f"旧因子（{rev}）：{len(old_rule.given_factors)} 项 {old_rule.weights}")

    loaded, panel = load_panel(quality_bars=args.quality_bars, limit=args.limit)
    print(f"取数：成功 {len(loaded.markets)}，跳过 {len(loaded.skipped)}")
    pool = universe_mask(panel, loaded.markets)
    print(f"池子：{int(pool.to_numpy().sum())} 个「标的 × 日」格")

    new_rule = brick_screen(top_n=args.top_n)
    new_result = new_rule.apply(panel, universe_mask=pool)
    old_result = old_rule.apply(panel, universe_mask=pool)

    old = picks_per_day(old_result, top_n=args.top_n)
    new = picks_per_day(new_result, top_n=args.top_n)

    same = sum(1 for day in old if old[day] == new.get(day))
    days = len(old)
    print(f"逐日名单完全相同的天数：{same}/{days}（{same / days * 100:.1f}%）")

    returns = forward_returns(panel)
    table = paired_report(old=old, new=new, returns=returns)
    print(f"有换手的交易日：{len(table)}")
    if table.empty:
        print("两套因子给出的名单逐日相同——没有可比的差。")
        return 0

    summarise(table)
    if args.out:
        table.to_csv(args.out, index=False)
        print(f"\n按天配对的表：{args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
