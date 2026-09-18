"""砖型规则的全市场计时（票据 #85）。

**为什么不放进测试套件**：它要读本机通达信数据（环境变量指路），且一次跑几分钟——与
``tools/oom_loop.py`` 同一类：依赖本机数据、手动跑、把一句「够快吗」变成两个数。

判据取自票据 #85：**总用时不超过 4 分钟**，其中**去掉取数**之后的净开销不超过 60 秒。
两个数分开记是因为它们回答不同的问题：总用时说的是「能不能每天跑」，净开销说的是「这条规则
自己贵不贵」——而取数那一百多秒是 I/O，与规则的深度无关，算进规则头上会掩盖真正要证的东西。

用法::

    $env:MBT_TDX_ROOT = "D:\\Tools\\new_tdx\\vipdoc"
    $env:MBT_TDX_GBBQ = "D:\\Tools\\new_tdx\\T0002\\hq_cache\\gbbq"
    .venv\\Scripts\\python.exe tools\\brick_screen_bench.py
    .venv\\Scripts\\python.exe tools\\brick_screen_bench.py --panel-bars 200   # 带上窗口

**同机交替 A/B**：要比较两个版本，就在同一台机器上交替跑两次（一次改前、一次改后），别拿
两个时间点上的绝对值直接比——同一份代码在同一台机器上的阶段用时能差三倍（机器状态所致）。
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

#: 判据（秒）。总用时含取数，净开销不含。
TOTAL_BUDGET_SECONDS = 240.0
NET_BUDGET_SECONDS = 60.0


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


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--panel-bars",
        type=int,
        default=0,
        help="面板日历截到评估日往前多少根（0 = 不截断）。砖型的递推记忆约一百根，"
        "而现在的暖机闸门是全局的那一个，故给小值会被拦下。",
    )
    parser.add_argument("--as-of", default=None, help="评估日，不给则取最近一个齐全的交易日")
    parser.add_argument("--top-n", default="all", help="取前 N 名；all 表示不截断")
    parser.add_argument("--progress", type=float, default=None, help="每 N 秒报一次进度")
    args = parser.parse_args()

    from mbt.cli import build_parser, run_screen_command

    root, gbbq = environment()
    argv = [
        "screen",
        "--screen",
        "brick",
        "--tdx-root",
        root,
        "--gbbq",
        gbbq,
        "--top-n",
        args.top_n,
        "--panel-bars",
        str(args.panel_bars),
        "--output-dir",
        str(REPO_ROOT / "tmp" / "brick-bench"),
    ]
    if args.as_of:
        argv += ["--as-of", args.as_of]
    if args.progress:
        argv += ["--progress", str(args.progress)]
    parsed = build_parser().parse_args(argv)

    print(f"跑：{' '.join(argv)}", flush=True)
    started = time.perf_counter()
    code = run_screen_command(parsed, stdout=sys.stdout, stderr=sys.stderr)
    total = time.perf_counter() - started

    print()
    print(f"退出码 {code}")
    print(
        f"总用时 {total:.1f}s（判据 ≤ {TOTAL_BUDGET_SECONDS:.0f}s）"
        f" → {'通过' if total <= TOTAL_BUDGET_SECONDS else '超出'}"
    )
    print(
        "净开销请从上面那份**耗时汇总**里读「排除取数」的那几项之和，"
        f"判据 ≤ {NET_BUDGET_SECONDS:.0f}s——本脚本不替它猜阈值在哪一行。"
    )
    return 0 if code == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
