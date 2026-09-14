"""进度上报：让「跑到哪了」可被**观察**，而不是被**估算**。

## 为什么需要它

全市场回测的耗时是**分钟到小时**量级，而它在跑的时候**一行输出都没有**：取数 5 分钟、算信号
4 分钟、算选股掩码 10 分钟、引擎一两个小时，全是静默的。于是「它在正常地慢」与「它卡死了」
从外部看**完全一样**——实测只能靠 ``Get-Process`` 看进程的 CPU 占用率去猜，而那是进程外的
手段，回答不了「卡在哪一步」。

本模块把这件事变成可观察的：**阶段边界**（现在在哪一步）与**阶段内进度**（这一步推进到哪、
速率多少）。两者的稀疏程度差得很远——阶段是几十个，进度是上万根——故接口分成两个方法，
上报频率各自独立。

## 三条纪律

**一、库默认不打印。** ``progress=None`` 是默认值，此时没有任何输出、也没有额外开销。要输出
就显式传一个 :class:`ConsoleProgress`——库不替调用方决定往哪打、打不打（与 CLI「产物路径必须
显式给出」同一立场）。

**二、进度按**墙上时间**节流，不按根数。** 引擎阶段有上万根，逐根输出会让「打印」本身成为回测
耗时的一部分（stdout 重定向到文件时尤其明显）；而按根数节流在不同规模下粒度又完全不同
（小回测一行不报、大回测刷屏）。故节流间隔由 ``every`` 以**秒**给定。

**三、取时点的函数只在上报时调用。** 取「当前跑到哪一天」要遍历全部标的（见
:class:`~mbt.backtest.engine._EngineClock`），故它被包成一个**回调**传进来，由上报方在决定
要输出**之后**才调用。若每根都调，进度上报本身就会给引擎凭空添一遍 O(标的数) 的遍历。

## 「是不是卡住了」怎么读

引擎是**单线程**的：卡住时它卡在**某一根 K 线内部**，此刻不可能再有任何输出。所以「没有新行」
本身**不能**直接等同于「卡住」——可能只是这一根特别慢。可判读的依据是**节流间隔**：阶段内
第一次上报时会写明「每 N 秒报一次」，于是「明显超过 N 仍无新行」才是信号。要知道具体卡在
哪一行，用标准库的 :mod:`faulthandler`
（``faulthandler.dump_traceback_later(...)``）打印各线程的栈——那是进程外的观察手段，与本
模块互补，见 ``.scratch/run_b1_full_market.py`` 的 ``--watchdog``。

### 已知的盲窗：引擎的装载段

引擎阶段有一处**结构性**的静默，说出来免得被误读成卡死：backtrader 在第一次 tick 之前要把
**全部标的的全部 K 线**读进它自己那套 line buffer，而这一段里策略根本没被调用，**没有任何
逐根进度可言**。实测它占引擎阶段的一大半（45 个标的时 5s／8.7s），到全市场就是**分钟量级**。

它无法从外部插桩：承载它的是 cerebro 的 ``preload``，而 ``preload`` 与 ``runonce`` 是同一个
开关的一体两面（``_dopreload and _dorunonce`` 决定走 ``_runonce`` 还是 ``_runnext``），关掉
它会顺带把引擎换到**另一条执行路径**上——那是「日志」不该动的东西。故处理方式是把它写进阶段
说明，并靠在首行进度里如实报出 ``已用 Ns``：那一个数就是装载段到底多长。
"""

from __future__ import annotations

import sys
import time
from collections.abc import Callable
from typing import Protocol, TextIO

#: 默认的上报间隔（秒）。5 秒够密（能看出推进）又够疏（不打乱输出节奏）。
DEFAULT_INTERVAL = 5.0


class ProgressReporter(Protocol):
    """进度上报的**接收端**。三个方法都由被观察的循环在固定时点调用。

    刻意只有三个方法：多加一个「阶段嵌套」或「百分比回调」都会让每个被观察的循环都要处理
    自己不属于它的复杂度，而实际需要的信息只有三样——**现在在哪一步**、**这一步推进到哪**、
    **每一步花了多久**。

    .. note::

        它是 :class:`typing.Protocol`，故**不要求实现方继承任何东西**：库只按这三个方法名调用，
        测试里给一个记事件的假对象即可，不必 mock 一个具体类。
    """

    def stage(self, name: str, *, note: str = "", unit: str = "根") -> None:
        """进入一个新阶段。``note`` 是补一句说明（规模、口径、为什么可能慢）。

        ``unit`` 是这一阶段 ``tick`` 里 ``done`` 的量词。默认「根」是因为最长的那个阶段
        （引擎）数的就是 K 线；取数阶段数的是标的，那时要传「只」。写成参数而不是让上报方
        自己拼字符串：量词错了不会报错，只会让人读到一个不对的量（「12,242 根股票」）。
        """

    def tick(
        self,
        done: int,
        total: int | None,
        clock: Callable[[], object] | None = None,
    ) -> None:
        """阶段内的进度：已完成 ``done``（总量 ``total``，未知时给 ``None``）。

        ``clock`` 是**取当前时点的函数**（如当前交易日、当前标的），只在真正要输出时才调用，
        理由见模块说明的第三条纪律。上报方自己决定要不要真的输出。
        """

    def finish(self) -> None:
        """本次运行结束——把最后一个阶段的用时结算出来。"""


class ConsoleProgress:
    """把进度打到文本流上，**按墙上时间节流**。

    输出的两类行：

    ======================  ====================================================
    阶段行                  ``阶段：<名字>（<说明>）``，进入该阶段时输出一次
    进度行                  ``  [<阶段>] <done>/<total> <量词>（<pct>）| <时点> | ...``
    ======================  ====================================================

    进度行里带**速率**与**线性外推的剩余时间**。后者刻意写明「线性外推」：它是拿已观察到的
    平均速率直接乘剩余量，而每根的代价并非恒定（越到后面上市标的越多、持仓越多），故它只是
    量级参考，不是承诺。**不要**把外推值当作 ETA 去排计划——本项目已经因为「从不具代表性的
    样本外推」给出过一个差 6 倍的估计。

    .. warning::

        **输出只含 GBK 能编码的字符**（无 emoji）。CLI 的三条硬规矩之一就是这个：控制台是 GBK，
        一个 emoji 会让整条命令崩在 ``UnicodeEncodeError`` 上，而那与数据无关。

    参数:
        stream: 写往哪个流。默认 ``sys.stdout``。
        every: 两次进度行之间至少间隔多少秒。**必须为正**——0 会让每根都输出，把打印变成耗时
            的主项。
        now: 取当前时间的函数（默认 ``time.monotonic``）。**可注入是为了可测**：测试不必真的
            睡 5 秒，把时间做成可控的即可。
    """

    def __init__(
        self,
        stream: TextIO | None = None,
        *,
        every: float = DEFAULT_INTERVAL,
        now: Callable[[], float] = time.monotonic,
    ) -> None:
        if every <= 0:
            raise ValueError(
                f"every 必须为正（秒），收到 {every!r}——按根数节流会让打印本身成为耗时的主项"
            )
        self.stream = sys.stdout if stream is None else stream
        self.every = every
        self._now = now
        self._name: str | None = None
        self._unit = "根"
        self._started = 0.0
        self._done = 0
        self._last: float | None = None

    # --- 接口 ---------------------------------------------------------------

    def stage(self, name: str, *, note: str = "", unit: str = "根") -> None:
        self._close()
        self._name = name
        self._unit = unit
        self._started = self._now()
        self._done = 0
        self._last = None
        suffix = f"（{note}）" if note else ""
        self._write(f"阶段：{name}{suffix}")

    def tick(self, done: int, total: int | None, clock=None) -> None:
        self._done = done
        now = self._now()
        if self._last is not None and now - self._last < self.every:
            return
        first = self._last is None
        self._last = now

        self._write(self._line(done, total, clock))
        if first:
            # 只在阶段内第一次上报时说一次节流间隔——「没有新行」是否算异常，全靠它判读。
            self._write(
                f"  （本阶段每 {self.every:g} 秒报一次；明显超过该间隔仍无新行，"
                f"说明卡在某一根内部）"
            )

    def finish(self) -> None:
        self._close()

    # --- 内部 ---------------------------------------------------------------

    def _line(self, done: int, total: int | None, clock) -> str:
        elapsed = self._now() - self._started
        head = f"  [{self._name or '进度'}] {done:,}"
        head += f"/{total:,} {self._unit}（{done / total:.1%}）" if total else f" {self._unit}"

        parts = [head]
        at = _safe_clock(clock)
        if at is not None:
            parts.append(str(at))
        parts.append(f"已用 {elapsed:.0f}s")
        if elapsed > 0 and done > 0:
            rate = done / elapsed
            parts.append(f"{rate:.1f} {self._unit}/秒")
            if total and total > done:
                parts.append(f"线性外推还需约 {(total - done) / rate:.0f}s")
        return " | ".join(parts)

    def _close(self) -> None:
        if self._name is None:
            return
        elapsed = self._now() - self._started
        counted = f"，{self._done:,} {self._unit}" if self._done else ""
        self._write(f"阶段：{self._name} 结束，用时 {elapsed:.1f}s{counted}")
        self._name = None

    def _write(self, line: str) -> None:
        # **必须 flush**：stdout 重定向到文件时是块缓冲，不 flush 就只能在结束时一次看到全部
        # ——那正是这个模块要消除的「静默」。
        print(line, file=self.stream, flush=True)


def _safe_clock(clock: Callable[[], object] | None) -> object | None:
    """取时点；取不到就返回 ``None``，**不让它把回测弄挂**。

    取时点要遍历全部标的，而它只是为了一行显示。为显示失败而中断一个两小时的运行，是把
    诊断工具的代价加到了被诊断的对象上。但也不静默吞掉：调用方会把「取不到」写进输出。
    """
    if clock is None:
        return None
    try:
        return clock()
    except Exception:  # noqa: BLE001 —— 见 docstring：显示失败不得中断回测
        return "(时点取不到)"
