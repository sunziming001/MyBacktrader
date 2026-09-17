"""回测产物的落盘：元数据、指标、净值数据与两张图（票据 #10）。

**只做 I/O**：算指标在 :mod:`mbt.metrics`（纯函数），本模块负责把它变成可归档、可批量生成
的一堆文件。

## 一个 run 的产物是一个整体

落成 ``<output_dir>/<run_id>/`` 而不是散在一层目录里，因为「哪份指标是那次回测的」这种事
靠文件名猜迟早出错。目录已存在即**报错**：回测产物是证据，覆盖它等于毁灭证据。

## 为什么不依赖引擎自带绘图

规格故事 54 要的是「可归档、可批量生成、不依赖引擎自带绘图的不可控输出」。这里自己写
**SVG**：输出是文本，可比对、可归档、批量生成无副作用，且不引入约 200 MB 的绘图依赖。
代价是只有线性坐标、无交互——要做多因子叠加或 K 线图时再换。
"""

from __future__ import annotations

import datetime as dt
import hashlib
import importlib
import json
import math
import platform
import subprocess
from collections.abc import Iterable, Sequence
from pathlib import Path

import pandas as pd

from mbt.metrics import Metrics, compute_metrics

#: 基准的默认标的：沪深300。`CONTEXT.md` 的**基准**条已定此默认。
DEFAULT_BENCHMARK_SYMBOL = "sh000300"

#: 折线配色。三条够用（策略、基准，留一条给将来）。
_COLOURS = ("#1f77b4", "#d62728", "#2ca02c")

#: 图表的画布与边距（像素）。刻意写成常数而非可配参数：要改就改代码，免得堆积没人调的旋钮。
_CHART = {
    "width": 800,
    "height": 300,
    "left": 64,
    "right": 16,
    "top": 16,
    "bottom": 32,
}


def data_snapshot(paths: Iterable[Path | str]) -> dict:
    """本次回测实际读取的数据文件的**清单摘要**。

    对每个文件取（路径、字节数、``mtime_ns``），排序后哈希成一个十六进制摘要。

    为什么不是内容哈希：那要读几百 MB。而这份摘要足以侦测规格故事 5 关心的情形——
    通达信**回补或修正**历史会造成文件长度与修改时间变化。

    局限要写明：``mtime`` 不是内容。内容改回原样会误报（无害）；刻意保留 ``mtime`` 而改
    内容会漏报（不太可能）。需要内容级保证时另行全量哈希。
    """
    entries = []
    for raw in sorted({str(path) for path in paths}):
        path = Path(raw)
        stat = path.stat()
        entries.append({"path": raw, "size": stat.st_size, "mtime_ns": stat.st_mtime_ns})

    payload = json.dumps(entries, sort_keys=True, separators=(",", ":"))
    return {
        "digest": hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16],
        "files": len(entries),
        "manifest": entries,
    }


def git_version() -> dict:
    """当前代码版本：``{"commit": ..., "dirty": ...}``。

    取不到就记 ``None``，**不报错**——元数据不该因为「这台机器没装 git」而让一次回测失败。
    """

    def run(*args):
        return subprocess.run(
            args, capture_output=True, text=True, check=False, timeout=10
        ).stdout.strip()

    try:
        commit = run("git", "rev-parse", "--short", "HEAD")
        if not commit:
            return {"commit": None, "dirty": None}
        return {"commit": commit, "dirty": bool(run("git", "status", "--porcelain"))}
    except (OSError, subprocess.SubprocessError):
        return {"commit": None, "dirty": None}


def describe_strategy(strategy, params: dict | None = None) -> dict:
    """策略的**可记**信息：导入路径 + 参数。

    参数必须 JSON 可序列化，否则**报错**——「三个月后能复现」的依据就是它，静默丢掉一个
    参数会让元数据看着完整而实则缺条件（那比不记更糟）。
    """
    cls = strategy if isinstance(strategy, type) else type(strategy)
    module = getattr(cls, "__module__", "")
    payload = dict(params or {})
    try:
        json.dumps(payload)
    except TypeError as exc:
        raise ValueError(
            f"策略参数必须可序列化为 JSON 才谈得上复现，而 {payload!r} 不行：{exc}"
        ) from exc
    return {"path": f"{module}.{cls.__qualname__}", "params": payload}


def describe_screen(screen, label: str | None = None) -> dict | None:
    """选股规则的**可读描述**，并显式标注它**未必能逐字复现**。

    过滤器与排序因子是 Python 可调用对象（常是 lambda），无法序列化。与其做一个看着能
    复现、实则悄悄丢条件的机制，不如把缺口写在元数据里。

    **排序因子记在 ``factors``（列表），不是单数的 ``factor``**（票据 #72）。多因子屏幕
    （ADR-0012 的 ``factors`` + ``weights`` + ``normalize``）的 ``Screen.factor`` 是
    ``None``，原先只读它会让这条规则的排序因子在留痕里消失——「换了排序因子」于是成了
    产物答不出的问题。列表没有「有因子却记成 null」这个歧义；单数的旧写法也归一到这里
    （:attr:`mbt.screen.Screen.given_factors`），只是列表长度为 1。``weights`` /
    ``normalize`` 原样记下当时的配置——``None`` 表示等权 / 不归一。
    """
    if screen is None:
        return None

    def names(parts) -> list[str]:
        """部件的名字：优先它自己的 ``__name__``，只有可调用对象才退回类型名。

        与 ``mbt.screen._part_name``（进度输出用的那处）同一口径——这里不 import 它是因为
        那是选股库的内部实现，而本模块只消费 ``Screen`` 这个公开类型。
        """
        return [f"{getattr(one, '__name__', type(one).__name__)}" for one in parts]

    return {
        "label": label,
        "filters": names(screen.filters),
        # **排序因子记成列表**，不是单数的 `factor`（票据 #72）：多因子屏幕（ADR-0012 的
        # `factors` + `weights` + `normalize`）的 `Screen.factor` 是 None，原先只读它会让
        # B1 的形态分数在留痕里消失。列表没有「有因子却记成 null」这个歧义；单数那条旧路
        # 也归一到这里（`Screen.given_factors`），只是列表长度为 1。
        "factors": names(screen.given_factors),
        # `weights` / `normalize` 原样记下当时的配置——None 表示等权 / 不归一。
        "weights": list(screen.weights) if screen.weights is not None else None,
        "normalize": screen.normalize,
        "top_n": screen.top_n,
        "reproducible": False,
        "note": "过滤器与排序因子是 Python 可调用对象，无法序列化；"
        "此处只记其名称，故本规则**不能**据此逐字复现。",
    }


#: 每日选股产出的自选股文件名。**固定**——通达信按文件名认自选股，名字一变，每天导入的
#: 就成了另一个板块。它是产物契约的一部分（仓库根目录的 `b1_daily.bat` 与测试都指向它），
#: 故写在这里当唯一出处，而不是散在命令行字符串里。
WATCHLIST_NAME = "每日选股.EBK"


def write_watchlist(candidates, path):
    """把候选写成**通达信自选股文件**（``.EBK``）：每行一个 6 位裸代码，**顺序即优劣**。

    参数:
        candidates: 一串标的符号（``sz300888`` 这种带市场前缀的），顺序按**从优到劣**。
        path: 落盘路径。父目录不在会被建出来——定时任务跑的那一刻没人能在旁边先 mkdir。

    返回落盘后的 :class:`~pathlib.Path`，便于调用方把它印进日志。

    固定的文件名见 :data:`WATCHLIST_NAME`；本函数收**路径**而不是目录，为的是让「固定名」
    这件事只住在 :data:`WATCHLIST_NAME` 一处，由调用方拼装，而不是在这里再写一遍。

    **与 run 目录那套约定刻意相反。** ``write_run_artifacts`` 用时间戳目录且**拒绝覆盖**
    （一次运行是一份不可变的证据）；而这一份是**固定名、每天覆盖**的——通达信按**文件名**
    认自选股，名字一变，每天导入的就成了另一个板块。

    **空清单照样落一份空文件**：文件必须**总是**代表「今日候选」，留一份昨天的清单在原地
    是更坏的谎——它看起来像今天的答案。代价是空文件会把通达信那边的自选股清空，故调用方
    **必须**把「今天没有候选」嚷出来（``mbt screen`` 那条日志的责任，不是这里的）。

    **行尾用 CRLF**：消费它的是一个 Windows 程序，而通达信自己的板块文件就是这个行尾。
    编码不影响内容（代码全是 ASCII 数字），故按平台无关的方式显式写死行尾，不靠 os.linesep。
    """
    from pathlib import Path

    from mbt.data.instrument import bare_code

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    lines = [bare_code(symbol) for symbol in candidates]
    target.write_bytes(("".join(f"{code}\r\n" for code in lines)).encode("ascii"))
    return target


def write_run_artifacts(
    result,
    *,
    output_dir,
    strategy=None,
    strategy_params: dict | None = None,
    screen=None,
    screen_label: str | None = None,
    cash: float | None = None,
    max_positions: int | None = None,
    costs: dict | None = None,
    universe=None,
    skipped=None,
    benchmark_prices: pd.Series | None = None,
    benchmark_symbol: str = DEFAULT_BENCHMARK_SYMBOL,
    snapshot_paths: Sequence[Path | str] = (),
    metrics: Metrics | None = None,
    created_at: dt.datetime | None = None,
) -> Path:
    """把一个 run 的产物落盘，返回该 run 的目录。

    参数:
        result: :class:`~mbt.backtest.BacktestResult`。
        output_dir: 产物的**父**目录。不存在则创建——显式传了路径就是要它存在。
        strategy / strategy_params: 记进元数据，供日后对照。
        screen / screen_label: 选股规则及其可读名（见 :func:`describe_screen`）。
        cash / max_positions / costs / universe: 回测的输入条件，原样记进元数据。
        benchmark_prices: 基准的价格序列。**本函数不读磁盘**——I/O 由调用方掌控。
            只有覆盖到回测区间的那一段会被用到，实际用到的区间记进元数据
            （不静默假装它覆盖了全程）。与回测区间毫无交集时**报错**。
        snapshot_paths: 本次实际读取的数据文件路径，用于生成清单摘要。
        metrics: 已有指标可传入以免重算；``None`` 时按 ``result`` 与基准现算。
        created_at: 生成时刻，默认取当前时间（可传入以便测试可复现）。

    返回:
        ``<output_dir>/<run_id>`` 路径。

    抛:
        ValueError: ``output_dir`` 下同名 run 目录已存在（**不静默覆盖**——产物是证据）；
            基准与回测区间无交集；策略参数不可序列化。
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    equity = result.equity_curve
    aligned_benchmark, benchmark_range = _align_benchmark(benchmark_prices, equity.index)

    if metrics is None:
        metrics = compute_metrics(
            equity,
            result.trades,
            benchmark_equity=aligned_benchmark,
            rejected=result.rejected,
        )

    created_at = created_at or dt.datetime.now()
    snapshot = data_snapshot(snapshot_paths)
    metadata = {
        "created_at": created_at.isoformat(timespec="seconds"),
        "strategy": describe_strategy(strategy, strategy_params) if strategy is not None else None,
        "screen": describe_screen(screen, screen_label),
        "period": {
            "start": str(equity.index[0].date()),
            "end": str(equity.index[-1].date()),
            "trading_days": int(len(equity)),
        },
        "cash": cash,
        "max_positions": max_positions,
        "costs": costs,
        "universe": universe,
        "benchmark": None
        if aligned_benchmark is None
        else {
            "symbol": benchmark_symbol,
            "used_start": str(benchmark_range[0].date()),
            "used_end": str(benchmark_range[1].date()),
            "bars": int(len(aligned_benchmark)),
        },
        "data_snapshot": snapshot,
        "git": git_version(),
        "software": {
            "mbt": _mbt_version(),
            "python": platform.python_version(),
            "pandas": pd.__version__,
            "numpy": _numpy_version(),
        },
    }
    metadata["run_id"] = _run_id(created_at, snapshot["digest"], strategy, strategy_params)
    run_dir = output_dir / metadata["run_id"]
    if run_dir.exists():
        raise ValueError(
            f"产物目录已存在：{run_dir}。回测产物是证据，不静默覆盖——请换 output_dir 或 run_id。"
        )
    run_dir.mkdir(parents=True)

    (run_dir / "run.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False, sort_keys=True), encoding="utf-8"
    )
    (run_dir / "metrics.json").write_text(
        json.dumps(_json_safe(metrics.as_dict()), indent=2, ensure_ascii=False, sort_keys=True),
        encoding="utf-8",
    )

    frame = pd.DataFrame({"equity": equity})
    if aligned_benchmark is not None:
        frame["benchmark"] = aligned_benchmark
    frame.index.name = "date"
    frame.to_csv(run_dir / "equity.csv")

    (run_dir / "equity.svg").write_text(
        render_equity_svg(equity, aligned_benchmark), encoding="utf-8"
    )
    (run_dir / "drawdown.svg").write_text(render_drawdown_svg(equity), encoding="utf-8")

    # 成交明细与**拒单明细**都落盘。拒单必须留下逐笔证据，否则「买单全被拒」这类事实在产物
    # 里完全看不见——实测排查时只能临时加打印，而那种排查不会留下任何可供事后核对的东西。
    result.trades.to_csv(run_dir / "trades.csv", index=False)
    result.rejected.to_csv(run_dir / "rejected.csv", index=False)

    # 被跳过的标的也落盘。控制台只列前几个（几十上百条会把输出冲垮），但**完整名单**必须
    # 能事后查到——实测一次 300 只的抽样里就有 5.7% 因「数据不可信」被拒收，而其中可能
    # 包括最要紧的那几只（票据 #45 的 sh600519 就是其一）。
    pd.DataFrame(
        [
            {"symbol": item.symbol, "kind": item.kind, "detail": item.detail}
            for item in (skipped or ())
        ],
        columns=["symbol", "kind", "detail"],
    ).to_csv(run_dir / "skipped.csv", index=False)

    return run_dir


def render_equity_svg(equity: pd.Series, benchmark: pd.Series | None = None) -> str:
    """净值曲线图。策略与基准**各自归一化到 1.0 起点**，故可直接比高低。

    基准只覆盖部分区间时，它的线**只画在覆盖到的那一段**——横轴按日期对齐，缺口留空。
    否则一条 2 个点的基准会被拉伸到整幅宽度，看上去像「全程都有基准」，而那正是本模块
    在元数据里刻意不假装的事。
    """
    pairs = [(_normalise(equity), "策略")]
    if benchmark is not None and benchmark.notna().any():
        pairs.append((_normalise(benchmark.dropna()), "基准"))
    return _line_chart(
        pairs,
        x_index=equity.index,
        zero_based=False,
        title="净值曲线（起点归一为 1.0）",
    )


def render_drawdown_svg(equity: pd.Series) -> str:
    """回撤图：画**负值**（``−回撤``），纵轴从 0 往下——读起来就是「回撤了多深」。

    注意 :attr:`~mbt.metrics.Metrics.max_drawdown` 报的是**正值**，只有图上取负。
    """
    peak = equity.cummax()
    drawdown = (equity - peak) / peak  # 0 或负
    return _line_chart(
        [(drawdown, "回撤")], x_index=equity.index, zero_based=True, title="回撤（自运行最高点）"
    )


def _json_safe(payload: dict) -> dict:
    """把 ``NaN`` / ``Infinity`` 换成 ``null``。

    Python 的 ``json.dumps`` 默认**会**输出字面量 ``NaN``，而那不是合法 JSON——`jq`、
    JavaScript、Rust 的解析器都会直接拒收整个文件。指标的缺失是正常状态（无波动、无平仓
    交易），故必须落成 ``null`` 而不是让文件变得读不了。
    """
    safe = {}
    for key, value in payload.items():
        if isinstance(value, float) and not math.isfinite(value):
            safe[key] = None
        else:
            safe[key] = value
    return safe


def import_strategy(metadata: dict):
    """按元数据里的导入路径取回策略类。

    这是「依据 run 元数据复现」的第一步：策略类本身是可导入对象，故只需记它的路径。
    取不回来时报错并说明是哪个路径——**不返回 ``None``**，否则调用方会拿着空值继续跑。
    """
    path = (metadata.get("strategy") or {}).get("path")
    if not path:
        raise ValueError("元数据里没有策略导入路径，无法复现")

    module_name, _, class_name = path.rpartition(".")
    try:
        module = importlib.import_module(module_name)
    except ImportError as exc:
        raise ValueError(
            f"复现失败：导入 {module_name!r} 出错（{exc}）。代码可能已改名或删除。"
        ) from exc
    try:
        return getattr(module, class_name)
    except AttributeError as exc:
        raise ValueError(f"复现失败：{module_name!r} 里没有 {class_name!r}") from exc


def replay_arguments(metadata: dict) -> dict:
    """把元数据翻回 ``run_portfolio_backtest`` 的关键字参数。

    **不含** ``markets`` 与 ``screen``：前者要重新取数（由调用方掌控 I/O），后者无法从元数据
    逐字恢复（见 :func:`describe_screen`）。故返回的是一个「除这两样之外都齐了」的参数字典，
    调用方补上那两样即可重跑。
    """
    costs = metadata.get("costs") or {}
    arguments = {
        "cash": metadata.get("cash"),
        "max_positions": metadata.get("max_positions"),
    }
    arguments.update({k: v for k, v in costs.items() if v is not None})
    for key in ("cash", "max_positions"):
        if arguments[key] is None:
            del arguments[key]
    return arguments


def verify_snapshot(metadata: dict, paths: Iterable[Path | str]) -> list[str]:
    """核对当前数据文件是否与元数据记录的一致，返回**问题清单**（空表示一致）。

    这是「复现」的另一半：参数一样但数据变了，重跑出来的就不是当初那次回测。故摘要对不上
    必须能被发现，而不是让人以为复现成功。
    """
    recorded = (metadata.get("data_snapshot") or {}).get("digest")
    if recorded is None:
        return ["元数据里没有数据快照摘要，无从核对数据是否变了"]

    current = data_snapshot(paths)["digest"]
    if current == recorded:
        return []

    return [
        f"数据快照已变化：元数据记的是 {recorded}，当前是 {current}。"
        f"数据被回补或修正过，重跑的结果不会与当初那次相同。"
    ]


def _run_id(created_at: dt.datetime, digest: str, strategy, params) -> str:
    """``YYYYMMDD-HHMMSS-<摘要前 6 位>``。

    摘要里带上「策略 + 参数 + 数据快照」的短码，故同一份数据与同一组参数在不同时刻跑出的
    结果，靠 run_id 就能看出它们是否同源。
    """
    identity = json.dumps(
        {
            "strategy": describe_strategy(strategy, params) if strategy is not None else None,
            "data": digest,
        },
        sort_keys=True,
    )
    short = hashlib.sha256(identity.encode("utf-8")).hexdigest()[:6]
    return f"{created_at:%Y%m%d-%H%M%S}-{short}"


def _align_benchmark(benchmark: pd.Series | None, index: pd.Index):
    """把基准对齐到回测区间：**取交集**，返回（对齐后的序列, 实际用到的区间）。

    与回测区间毫无交集时**报错**——那不叫「基准缺了一点」，那说明基准跟这次回测根本不是
    一段时期，拿它对比没有意义。
    """
    if benchmark is None or benchmark.empty:
        return None, None

    if not isinstance(benchmark, pd.Series):
        # 传进来的多半是 daily() 的**字段宽表**（六列）。不拦的话，它会在算年化时以一句
        # 难懂的 pandas 报错收场（"cannot convert the series to float"）。
        raise ValueError(
            f"基准须是**单列价格序列**，收到的却是 {type(benchmark).__name__}"
            f"（列：{list(benchmark.columns)}）。若来自 TdxDataSource.daily()，请取 ['close']。"
        )

    common = index.intersection(benchmark.index)
    if len(common) < 2:
        raise ValueError(
            f"基准与回测区间没有足够的交集（共 {len(common)} 个交易日）："
            f"回测 {index[0].date()}–{index[-1].date()}，"
            f"基准 {benchmark.index[0].date()}–{benchmark.index[-1].date()}。"
            f"两段时期对不上，对比没有意义。"
        )
    aligned = benchmark.reindex(common)
    return aligned, (common[0], common[-1])


def _normalise(series: pd.Series) -> pd.Series:
    first = float(series.iloc[0])
    return series / first if first else series * math.nan


def _line_chart(pairs, *, x_index: pd.Index, zero_based: bool, title: str) -> str:
    """把若干条 ``(序列, 图例)`` 画成一张 SVG 折线图。

    横轴**按日期对齐**：``x_index`` 给出整幅图的日期刻度（即策略净值曲线的索引），每条线
    按其自身的日期落到对应位置。故只覆盖部分区间的序列只占住那一段，而不会被拉伸到整幅
    宽度——后者会让「基准只覆盖了两天」看上去像「全程都有基准」。

    纯文本生成，无依赖。坐标**线性**：对数轴对「亏了多少」这类判断反而更难读，而本票的
    核心用途是判断策略行不行。
    """
    width, height = _CHART["width"], _CHART["height"]
    left, right, top, bottom = (
        _CHART["left"],
        _CHART["right"],
        _CHART["top"],
        _CHART["bottom"],
    )
    plot_w, plot_h = width - left - right, height - top - bottom

    low, high = _bounds([series for series, _ in pairs], zero_based)
    span = high - low or 1.0
    positions = {stamp: i for i, stamp in enumerate(x_index)}

    def x_at(i):
        return left + (plot_w * i / (len(x_index) - 1) if len(x_index) > 1 else plot_w / 2)

    def y_at(value):
        return top + plot_h * (1.0 - (value - low) / span)

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}" font-family="sans-serif" font-size="11">',
        f'<rect width="{width}" height="{height}" fill="white"/>',
        f'<text x="{left}" y="{top - 4}" font-size="12">{_escape(title)}</text>',
        f'<rect x="{left}" y="{top}" width="{plot_w}" height="{plot_h}" '
        f'fill="none" stroke="#ccc"/>',
    ]

    for frac in (0.0, 0.25, 0.5, 0.75, 1.0):
        value = low + span * frac
        y = y_at(value)
        parts.append(
            f'<line x1="{left}" y1="{y:.2f}" x2="{left + plot_w}" y2="{y:.2f}" stroke="#eee"/>'
        )
        parts.append(
            f'<text x="{left - 6}" y="{y + 3:.2f}" text-anchor="end" fill="#666">'
            f"{value:.3f}</text>"
        )

    for position, (series, _) in enumerate(pairs):
        colour = _COLOURS[position % len(_COLOURS)]
        points = " ".join(
            f"{x_at(positions[stamp]):.2f},{y_at(float(value)):.2f}"
            for stamp, value in series.items()
            if stamp in positions and not math.isnan(float(value))
        )
        parts.append(
            f'<polyline fill="none" stroke="{colour}" stroke-width="1.5" points="{points}"/>'
        )

    for position, (_, label) in enumerate(pairs):
        x = left + 8 + position * 60
        parts.append(
            f'<rect x="{x}" y="{height - bottom + 14}" width="10" height="10" '
            f'fill="{_COLOURS[position % len(_COLOURS)]}"/>'
        )
        parts.append(
            f'<text x="{x + 14}" y="{height - bottom + 23}" fill="#333">' f"{_escape(label)}</text>"
        )

    parts.append("</svg>")
    return "\n".join(parts)


def _bounds(series_list, zero_based: bool) -> tuple[float, float]:
    values = [float(v) for series in series_list for v in series.dropna()]
    low, high = min(values), max(values)
    if zero_based:
        low = min(low, 0.0)
        high = max(high, 0.0)
    if low == high:
        low, high = low - 0.5, high + 0.5
    return low, high


def _escape(text: str) -> str:
    """SVG 是 XML：标题里的 ``&`` / ``<`` 不转义就不是合法文档。"""
    return (
        text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace('"', "&quot;")
    )


def _mbt_version() -> str:
    from mbt import __version__

    return __version__


def _numpy_version() -> str:
    import numpy

    return numpy.__version__
