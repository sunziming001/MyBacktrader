"""OOM 判据夹具（票据 #78）：把提交上限钉死，让偶发失败变成每次都在同一个地方死。

偶发的根因不是代码，是机器当时的余量。故判据不靠等它复现，而是**自己把上限钉死**——
用 Windows Job Object 给本进程设一个提交（commit）上限，再跑与 ``b1_daily.bat`` **同一条**
命令。真实失败本来就是「提交量撞到上限」，钉住它，失败就落在一个固定的位置上。

用法::

    .venv\\Scripts\\python.exe tools\\oom_loop.py --cap-gb 12 --sampled
    .venv\\Scripts\\python.exe tools\\oom_loop.py --cap-gb 12 --panel-bars 0   # 切之前的对照
    .venv\\Scripts\\python.exe tools\\oom_loop.py --selftest --cap-gb 4

路径只从环境变量取，与 ``b1_daily.bat`` 同一套（``MBT_TDX_ROOT`` / ``MBT_TDX_GBBQ`` /
``MBT_TDX_GPONE``，最后一个没设就按历史 PE 跑，行为与那个 bat 一致）。

两个副作用是刻意加的：

- **失败时打完整栈**（``[TRACEBACK]``）。``mbt screen`` 自己只打一行「错误：选股失败——…」，
  栈在它的 ``except`` 里丢了；而要回答「这个满面板数组是谁分配的」，缺的正是那行调用栈。
- **``--sampled``** 每 2 秒往 stdout 打一行 ``[MEM]``。``progress.py`` 的阶段计时在 ``finally``
  里上报，异常退出时照打，故「计时行」不能说明那一段跑完了；提交**峰值**又是单调高水位、归因
  不到阶段。只有按阶段读**当前**提交量才能看出是谁在涨。

这是一次性的诊断工具，不是库的一部分：**它不进测试套件**——依赖 Windows 的 Job Object 与
``GetProcessMemoryInfo``，与 ``realmdata`` 那类同性质，换机器就未必跑得动，手动跑。

**它为什么在仓库里、而不在 ``tmp/``。** 这套夹具丢过一次，判据也就一起没了——本票的判据是
重建出来的。同一个坑仓库里踩过两回：ADR-0015 引作复现入口的 ``smooth_dump_or_compare.py``
住在 gitignore 的 ``.scratch/`` 下，后来全盘搜过，已经不在了。证据脚本要和结论活得一样久。
"""

from __future__ import annotations

import argparse
import ctypes
import datetime as dt
import os
import sys
import threading
import time
import traceback
from ctypes import wintypes
from pathlib import Path

if sys.platform != "win32":
    raise SystemExit(
        "本夹具依赖 Windows 的 Job Object 与 GetProcessMemoryInfo，只在 Windows 上跑。"
    )

_kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
_psapi = ctypes.WinDLL("psapi", use_last_error=True)

#: Job Object 的提交上限标志（``JOB_OBJECT_LIMIT_JOB_MEMORY``，winnt.h）。
#: 别写成 0x2000——那是 ``JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE``。写错这个数的表现是：
#: 上限**看着设上了**（``LimitFlags`` 回读非零）、``JobMemoryLimit`` 却被清零，于是它
#: 一路吃满内存而不报错——「没红」会被误读成「修好了」。故 ``--selftest`` 是必需的。
_LIMIT_JOB_MEMORY = 0x200
_EXTENDED_LIMIT_INFORMATION = 9


class _BasicLimitInformation(ctypes.Structure):
    _fields_ = [
        ("PerProcessUserTimeLimit", ctypes.c_int64),
        ("PerJobUserTimeLimit", ctypes.c_int64),
        ("LimitFlags", wintypes.DWORD),
        ("MinimumWorkingSetSize", ctypes.c_size_t),
        ("MaximumWorkingSetSize", ctypes.c_size_t),
        ("ActiveProcessLimit", wintypes.DWORD),
        ("Affinity", ctypes.c_size_t),
        ("PriorityClass", wintypes.DWORD),
        ("SchedulingClass", wintypes.DWORD),
    ]


class _IoCounters(ctypes.Structure):
    _fields_ = [
        ("ReadOperationCount", ctypes.c_uint64),
        ("WriteOperationCount", ctypes.c_uint64),
        ("OtherOperationCount", ctypes.c_uint64),
        ("ReadTransferCount", ctypes.c_uint64),
        ("WriteTransferCount", ctypes.c_uint64),
        ("OtherTransferCount", ctypes.c_uint64),
    ]


class _ExtendedLimitInformation(ctypes.Structure):
    _fields_ = [
        ("BasicLimitInformation", _BasicLimitInformation),
        ("IoInfo", _IoCounters),
        ("ProcessMemoryLimit", ctypes.c_size_t),
        ("JobMemoryLimit", ctypes.c_size_t),
        ("PeakProcessMemoryUsed", ctypes.c_size_t),
        ("PeakJobMemoryUsed", ctypes.c_size_t),
    ]


class _ProcessMemoryCounters(ctypes.Structure):
    _fields_ = [
        ("cb", wintypes.DWORD),
        ("PageFaultCount", wintypes.DWORD),
        ("PeakWorkingSetSize", ctypes.c_size_t),
        ("WorkingSetSize", ctypes.c_size_t),
        ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
        ("QuotaPagedPoolUsage", ctypes.c_size_t),
        ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
        ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
        ("PagefileUsage", ctypes.c_size_t),
        ("PeakPagefileUsage", ctypes.c_size_t),
    ]


class _MemoryStatusEx(ctypes.Structure):
    _fields_ = [
        ("dwLength", wintypes.DWORD),
        ("dwMemoryLoad", wintypes.DWORD),
        ("ullTotalPhys", ctypes.c_uint64),
        ("ullAvailPhys", ctypes.c_uint64),
        ("ullTotalPageFile", ctypes.c_uint64),
        ("ullAvailPageFile", ctypes.c_uint64),
        ("ullTotalVirtual", ctypes.c_uint64),
        ("ullAvailVirtual", ctypes.c_uint64),
        ("ullAvailExtendedVirtual", ctypes.c_uint64),
    ]


_kernel32.CreateJobObjectW.restype = wintypes.HANDLE
_kernel32.CreateJobObjectW.argtypes = [ctypes.c_void_p, wintypes.LPCWSTR]
_kernel32.GetCurrentProcess.restype = wintypes.HANDLE
_kernel32.SetInformationJobObject.argtypes = [
    wintypes.HANDLE,
    ctypes.c_int,
    ctypes.c_void_p,
    wintypes.DWORD,
]
_kernel32.AssignProcessToJobObject.argtypes = [wintypes.HANDLE, wintypes.HANDLE]
_kernel32.QueryInformationJobObject.argtypes = [
    wintypes.HANDLE,
    ctypes.c_int,
    ctypes.c_void_p,
    wintypes.DWORD,
    ctypes.c_void_p,
]
_psapi.GetProcessMemoryInfo.argtypes = [wintypes.HANDLE, ctypes.c_void_p, wintypes.DWORD]


def commit_limit_of_machine() -> tuple[int, int]:
    """本机的（物理内存, 提交上限）——两个数分开报，因为撞的是后者。"""
    status = _MemoryStatusEx()
    status.dwLength = ctypes.sizeof(status)
    if not _kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
        raise OSError(f"GlobalMemoryStatusEx 失败：{ctypes.get_last_error()}")
    return status.ullTotalPhys, status.ullTotalPageFile


def current_commit() -> int:
    """本进程**当前**提交量（``PagefileUsage``）——按阶段读它才能归因。"""
    counters = _ProcessMemoryCounters()
    counters.cb = ctypes.sizeof(counters)
    if not _psapi.GetProcessMemoryInfo(
        _kernel32.GetCurrentProcess(), ctypes.byref(counters), counters.cb
    ):
        raise OSError(f"GetProcessMemoryInfo 失败：{ctypes.get_last_error()}")
    return counters.PagefileUsage


def cap_commit(limit_bytes: int) -> int:
    """把本进程放进一个设了提交上限的 Job Object，返回它的句柄。

    为什么要 Job Object 而不是 ``setrlimit``：Windows 没有那个。为什么给自己设而不是给子进程：
    Job 的提交上限按**作业**计，本进程独自成组即可；而进程一旦被放进作业，后续
    ``VirtualAlloc`` 撞上限就报 ``MemoryError``——与线上那次逐字同型。
    """
    job = _kernel32.CreateJobObjectW(None, None)
    if not job:
        raise OSError(f"CreateJobObjectW 失败：{ctypes.get_last_error()}")

    info = _ExtendedLimitInformation()
    info.BasicLimitInformation.LimitFlags = _LIMIT_JOB_MEMORY
    info.JobMemoryLimit = limit_bytes
    if not _kernel32.SetInformationJobObject(
        job, _EXTENDED_LIMIT_INFORMATION, ctypes.byref(info), ctypes.sizeof(info)
    ):
        raise OSError(f"SetInformationJobObject 失败：{ctypes.get_last_error()}")

    if not _kernel32.AssignProcessToJobObject(job, _kernel32.GetCurrentProcess()):
        raise OSError(
            f"AssignProcessToJobObject 失败：{ctypes.get_last_error()}"
            "（本进程可能已在一个不允许嵌套的作业里）"
        )
    return job


def peak_commit(job: int) -> int:
    info = _ExtendedLimitInformation()
    if not _kernel32.QueryInformationJobObject(
        job, _EXTENDED_LIMIT_INFORMATION, ctypes.byref(info), ctypes.sizeof(info), None
    ):
        raise OSError(f"QueryInformationJobObject 失败：{ctypes.get_last_error()}")
    return info.PeakJobMemoryUsed


class Tee:
    """把一份输出同时送到控制台与日志文件。"""

    def __init__(self, *streams):
        self._streams = streams

    def write(self, text):
        for stream in self._streams:
            stream.write(text)

    def flush(self):
        for stream in self._streams:
            stream.flush()


class _Failure:
    """记下那次失败的类型与消息，供最后的判据行引用。"""

    def __init__(self):
        self.kind = None
        self.message = None


def install_traceback_print(failure: _Failure):
    """让 ``mbt.cli`` 打那行「错误：…失败——…」时**顺手打完整栈**。

    ``run_screen_command`` 把 ``screen.apply`` 包在 ``except Exception`` 里，只印一行、栈就丢了
    （MemoryError 也是 Exception，故一并被吃掉）。它印的那一行**正在 except 块内**，所以
    ``sys.exc_info()`` 此刻仍然是活的——在打印的那一瞬间把栈取出来即可，不必动库里的代码。

    做法是往 ``mbt.cli`` 的模块全局里塞一个 ``print``，它会遮蔽内建的那个。
    """
    import mbt.cli

    original = print

    def hook(*parts, **kwargs):
        text = " ".join(str(part) for part in parts)
        if text.startswith("错误："):
            active = sys.exc_info()[0] is not None
            if active:
                failure.kind = sys.exc_info()[0].__name__
                failure.message = text
                original(
                    "[TRACEBACK] mbt 只打一行、栈是丢的，故在这里补上完完整整的那一份：",
                    **kwargs,
                )
                original(traceback.format_exc(), **kwargs)
        return original(*parts, **kwargs)

    mbt.cli.print = hook


def sample_loop(interval: float, started: float, stop: threading.Event, stream):
    """每 ``interval`` 秒打一行 ``[MEM]``——与阶段行天然对齐（都在 stdout）。"""
    while not stop.wait(interval):
        try:
            print(
                f"  [MEM] 当前提交 {current_commit() / 1024**3:.2f} GB"
                f" | 已用 {time.monotonic() - started:.0f}s",
                file=stream,
                flush=True,
            )
        except Exception:  # noqa: BLE001 — 采样失败不该弄死被测的那次运行
            pass


def screen_argv(top_n: str, out_dir: Path, panel_bars: int, stream) -> list[str]:
    """与 ``b1_daily.bat`` 同一条命令——路径同样**只从环境变量取、指错就报错退出**。

    ``--panel-bars`` 是**真开关**（ADR-0014），不是猴子补丁：夹具要走的就是生产那条路，
    否则「夹具绿了」证明不了「生产绿了」。

    ``stream`` 是那份 ``Tee``：提示必须**同时进控制台与日志**——日志才是留下判据现场的那份，
    一次跑的日志里若看不出「这次到底用没用前瞻」，那个数字就没法解释。
    """
    root = os.environ.get("MBT_TDX_ROOT")
    gbbq = os.environ.get("MBT_TDX_GBBQ")
    if not root:
        raise SystemExit("MBT_TDX_ROOT 未设置（vipdoc 根）。见 README 那一节。")
    if not gbbq:
        raise SystemExit("MBT_TDX_GBBQ 未设置（gbbq 文件）。见 README 那一节。")

    argv = [
        "screen",
        "--screen",
        "b1",
        "--top-n",
        top_n,
        "--tdx-root",
        root,
        "--gbbq",
        gbbq,
        "--cw-root",
        str(Path(root) / "cw"),
        "--watchlist-dir",
        str(out_dir),
        "--output-dir",
        str(out_dir / "screens"),
        "--panel-bars",
        str(panel_bars),
        "--progress",
    ]

    gpone = os.environ.get("MBT_TDX_GPONE")
    if gpone and (Path(gpone) / "gpshone.dat").is_file():
        argv += ["--forward-root", gpone]
        print(f"NOTE: 前瞻口径已接（--forward-root {gpone}）", file=stream, flush=True)
    else:
        print(
            "NOTE: 没有 MBT_TDX_GPONE（或缺 gpshone.dat）——按历史 PE 跑，与 b1_daily.bat 一致",
            file=stream,
            flush=True,
        )
    return argv


def selftest(limit: int) -> int:
    """确认上限**真的生效**、报的**是** ``MemoryError``。

    不验这一步，夹具就是自欺：上限没设上时它会一路吃满内存，而「没红」看起来像「修好了」。
    分配用 ``np.ones`` 而不是 ``np.empty``：要的是**提交**（写过的页），不是保留。
    """
    import numpy as np

    print(f"自检：提交上限 = {limit / 1024**3:.2f} GB")
    chunk = 64 * 1024 * 1024  # 64 MB 一份
    held = []
    total = 0
    try:
        while total < limit + 512 * 1024 * 1024:
            held.append(np.ones(chunk // 8, dtype=np.float64))
            total += chunk
    except MemoryError as exc:
        print(f"自检：第 {len(held) + 1} 份 64 MB 撞上限，累计 {total / 1024**3:.2f} GB")
        print(f"自检：异常类型 = {type(exc).__name__}（线上那次同型）")
        print(f"自检：当前提交 = {current_commit() / 1024**3:.2f} GB")
        print("自检：通过（上限生效，且报的是 MemoryError）")
        return 0
    print(f"自检：未撞上限却已分配 {total / 1024**3:.2f} GB —— 上限没生效")
    return 1


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="票据 #78 的 OOM 判据夹具")
    parser.add_argument("--cap-gb", type=float, default=12.0, help="提交上限（GB），默认 12")
    parser.add_argument("--top-n", default="all", help="透给 mbt screen 的 --top-n")
    parser.add_argument("--sampled", action="store_true", help="每 2 秒打一行 [MEM] 提交量")
    parser.add_argument("--selftest", action="store_true", help="只验上限真的生效就退出")
    parser.add_argument("--log", default="tmp/oom_loop.log", help="日志路径（同时打控制台）")
    parser.add_argument("--out-dir", default="tmp/oom", help="选股产物目录")
    parser.add_argument(
        "--panel-bars",
        type=int,
        default=1301,
        help="面板日历截到评估日往前 N 根（ADR-0014）；默认 1301 与 b1_daily.bat 一致，"
        "给 0 就是切之前的对照",
    )
    args = parser.parse_args(argv)

    limit = int(args.cap_gb * 1024**3)
    if args.selftest:
        cap_commit(limit)
        return selftest(limit)

    from mbt.cli import build_parser, run_screen_command

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = Path(args.log)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    physical, commit_ceiling = commit_limit_of_machine()
    failure = _Failure()
    code = 0

    with open(log_path, "w", encoding="utf-8") as handle:
        tee = Tee(sys.stdout, handle)
        started = time.monotonic()
        print("=" * 72, file=tee)
        print(f"===== {dt.datetime.now():%Y-%m-%d %H:%M:%S} 票据 #78 的 OOM 判据 =====", file=tee)
        print(
            f"本机：物理 {physical / 1024**3:.2f} GB，提交上限 {commit_ceiling / 1024**3:.2f} GB",
            file=tee,
        )
        print(f"本次钉住的提交上限 = {limit / 1024**3:.2f} GB", file=tee)
        by_window = "不切（对照）" if args.panel_bars <= 0 else f"评估日往前 {args.panel_bars} 根"
        print(f"面板日历截断     = {by_window}", file=tee)
        print(f"启动时提交 = {current_commit() / 1024**3:.2f} GB", file=tee)
        print("=" * 72, file=tee)
        tee.flush()

        job = cap_commit(limit)
        install_traceback_print(failure)

        stop = threading.Event()
        if args.sampled:
            threading.Thread(
                target=sample_loop, args=(2.0, started, stop, tee), daemon=True
            ).start()

        try:
            parsed = build_parser().parse_args(
                screen_argv(args.top_n, out_dir, args.panel_bars, tee)
            )
            code = run_screen_command(parsed, stdout=tee, stderr=tee)
        except BaseException as exc:  # noqa: BLE001 — 夹具的职责是把现场留下来
            failure.kind = type(exc).__name__
            failure.message = str(exc)
            print("[TRACEBACK] 夹具自己被异常带出（不是 mbt 自己接住的那种）：", file=tee)
            print(traceback.format_exc(), file=tee)
            code = 1
        finally:
            stop.set()

        peak = peak_commit(job)
        print("=" * 72, file=tee)
        print(f"退出码     = {code}", file=tee)
        print(f"提交峰值   = {peak / 1024**3:.2f} GB（上限 {limit / 1024**3:.2f} GB）", file=tee)
        print(f"墙钟       = {time.monotonic() - started:.0f}s", file=tee)
        if failure.kind:
            print(f"{failure.kind} = {failure.message}", file=tee)
        print(f"判据：{'红（撞上限了）' if code else '绿（没撞上限）'}", file=tee)
        print(f"日志：{log_path}", file=tee)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
