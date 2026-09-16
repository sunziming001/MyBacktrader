"""进度上报：让「跑到哪了」可被观察（``mbt.progress``）。

本文件盯四件事，每一件都对应 ``mbt.progress`` 里写下的一条纪律：

1. **节流是按时间的，且取时点的函数只在真要输出时才被调用**——那条纪律省下的正是每次
   遍历全部标的的开销，故它必须被测住，而不是只写在注释里；
2. **阶段与进度都要说得清**：阶段名、量词、分母、节流间隔；
3. **上报不改变回测结果**——只读的东西不得动一个数；
4. **输出全部落在 GBK 内**——控制台是 GBK，一个 emoji 就让整条命令崩在
   ``UnicodeEncodeError`` 上，而那与数据无关。

进度上报本身是「给人看」的，故这里的断言都盯**可判读性**（阶段名对不对、分母是不是真有
那么多、多久报一次说没说），而不是盯某一行的字面格式——格式会变，可判读性不能。
"""

from __future__ import annotations

import io

import backtrader as bt
import pandas as pd
import pytest

from mbt.backtest import run_portfolio_backtest
from mbt.data import load_universe_data
from mbt.progress import ConsoleProgress, timed
from mbt.screen import Screen
from mbt.universe import UniverseRules

# --- 工具 -------------------------------------------------------------------


class FakeClock:
    """可控的单调时钟：测试不必真的睡 5 秒，把时间推过去就行。"""

    def __init__(self, start=1000.0):
        self.now = start

    def __call__(self):
        return self.now

    def advance(self, seconds):
        self.now += seconds


class Recording:
    """记事件的假上报端。

    它是**鸭子类型**的——刻意不继承 :class:`~mbt.progress.ProgressReporter`：那个 Protocol 的
    存在意义就是「库只按方法名调用，不要求实现方继承任何东西」，测试用假对象正是这条的验证。
    """

    def __init__(self):
        self.stages: list[tuple[str, str, str]] = []
        self.ticks: list[tuple[int, int | None]] = []
        self.finished = 0

    def stage(self, name, *, note="", unit="根"):
        self.stages.append((name, note, unit))

    def tick(self, done, total, clock=None):
        self.ticks.append((done, total))

    def finish(self):
        self.finished += 1

    @property
    def names(self) -> list[str]:
        return [name for name, _, _ in self.stages]


def flat_bars(n, value, start="2024-01-02", volume=1000):
    return pd.DataFrame(
        {
            "open": [value] * n,
            "high": [value] * n,
            "low": [value] * n,
            "close": [value] * n,
            "volume": [volume] * n,
        },
        index=pd.bdate_range(start, periods=n),
    )


def lines(progress: ConsoleProgress) -> list[str]:
    return progress.stream.getvalue().splitlines()


def make_progress(clock=None, every=5.0):
    return ConsoleProgress(io.StringIO(), every=every, now=clock or FakeClock())


# --- ConsoleProgress：节流、取时点、格式 --------------------------------------


def test_a_tick_inside_the_interval_is_silent_and_the_next_stage_closes_the_last():
    """进度行按墙上时间节流；进入新阶段时把上一阶段的用时结算出来。

    「阶段用时」是排查「哪一步慢」的第一手材料，故它必须出现——否则只看得到进度行，
    看不出每个阶段各自花了多久。
    """
    clock = FakeClock()
    progress = make_progress(clock)

    progress.stage("回测引擎", note="4752 个标的")
    progress.tick(1, 100)
    progress.tick(50, 100)  # 同一时刻，不到间隔 → 应当无输出
    clock.advance(1.0)
    progress.tick(60, 100)  # 才过 1 秒 → 仍然无输出

    assert len(lines(progress)) == 1 + 2, "1 行阶段 + 首次进度的两行，中间的 tick 不该出声"

    clock.advance(10.0)
    progress.tick(70, 100)  # 过了一个间隔 → 出声
    progress.stage("落盘")

    printed = lines(progress)
    assert any("回测引擎 结束，用时 11.0s" in line for line in printed), printed
    assert printed[-1] == "阶段：落盘"


def test_the_clock_is_asked_for_exactly_once_per_printed_progress_line():
    """**取时点只在真要输出时调用**——这条纪律是进度上报不能拖慢回测的全部理由。

    取一次「今天跑到哪一天」要遍历全部标的（全市场 4,752 个）。若每根都取，进度上报本身
    就给引擎凭空添一遍 O(标的数) 的遍历；被节流掉的那些 tick 一次都不该取。
    """
    clock = FakeClock()
    asked = []

    def counting_clock():
        asked.append(1)
        return "2024-01-02"

    progress = make_progress(clock)
    progress.stage("回测引擎")
    for i in range(1, 11):
        progress.tick(i, 1000, counting_clock)
        clock.advance(0.1)  # 10 次调用全落在同一个 5 秒间隔内

    printed = [line for line in lines(progress) if line.startswith("  [")]
    assert (
        len(asked) == len(printed) == 1
    ), f"取时点 {len(asked)} 次、进度行 {len(printed)} 行；两者必须一一对应"


def test_the_first_progress_line_states_how_often_progress_is_reported():
    """首次进度行要写明**每几秒报一次**。

    「是不是卡住了」唯一的判读依据就是「更新间隔」：进程卡在某一根里时不可能再有输出，
    故「没有新行」只有配上「本该多久有一行」才算证据。不写这个间隔，进度条就只能证明
    「曾经在跑」，证明不了「现在还活着」。
    """
    progress = make_progress(every=7.0)
    progress.stage("回测引擎")
    progress.tick(1, 100)

    printed = lines(progress)
    assert any("每 7 秒报一次" in line for line in printed), printed
    # 只说一次，不是每行都念一遍——否则进度输出会淹没在提示里
    assert sum("每 7 秒报一次" in line for line in printed) == 1


def test_a_progress_line_carries_position_total_and_rate():
    """进度行的可判读内容：第几根、总共几根、跑了多久、快不快。"""
    clock = FakeClock()
    progress = make_progress(clock)
    progress.stage("回测引擎", unit="根")
    clock.advance(100.0)
    progress.tick(250, 1000, lambda: "2024-01-02")

    (line,) = [line for line in lines(progress) if line.startswith("  [")]
    assert "250/1,000 根（25.0%）" in line
    assert "2024-01-02" in line
    assert "100s" in line
    assert "2.5 根/秒" in line
    # 外推值必须**标明是外推**：每根的实际代价并非恒定，把它当 ETA 会给出错误承诺。
    assert "线性外推" in line


def test_the_unit_of_the_stage_is_used_in_the_progress_line():
    """量词按阶段给：取数数的是标的（只），引擎数的是 K 线（根）。

    量词错了不会报错，只会让人读到一个不对的量（「12,242 根股票」）。
    """
    progress = make_progress()
    progress.stage("取数", note="12243 个候选标的", unit="只")
    progress.tick(500, 12243, lambda: "sh600000")

    (line,) = [line for line in lines(progress) if line.startswith("  [")]
    assert "500/12,243 只" in line, line


def test_an_unknown_total_still_reports_progress_without_a_fraction():
    """总量未知时照报，只是不给比例与外推——**不猜总量**。"""
    progress = make_progress()
    progress.stage("回测引擎")
    progress.tick(250, None, lambda: "2024-01-02")

    (line,) = [line for line in lines(progress) if line.startswith("  [")]
    assert "250 根" in line
    assert "%" not in line
    assert "线性外推" not in line


def test_a_broken_clock_does_not_take_the_backtest_down_with_it():
    """取时点失败只影响这一行的显示，**不得中断一个几小时的运行**。

    但也不静默：输出里要写得出「取不到」，否则一个恒久显示同一日期的进度条会让人以为
    引擎停在那儿了。
    """

    def broken():
        raise RuntimeError("标的还没就绪")

    progress = make_progress()
    progress.stage("回测引擎")
    progress.tick(1, 100, broken)  # 不抛

    (line,) = [line for line in lines(progress) if line.startswith("  [")]
    assert "时点取不到" in line


def test_a_nonpositive_interval_is_rejected_rather_than_silently_outputting_per_bar():
    """``every <= 0`` 必须当场报错。

    放它过去就等于按根数输出：上万根 K 线逐根打印，打印本身会成为回测耗时的主项——是
    一种**看起来在工作、实际在拖慢**的失败，比报错难查得多。
    """
    for bad in (0, -1.0):
        with pytest.raises(ValueError, match="every 必须为正"):
            ConsoleProgress(io.StringIO(), every=bad)


def test_every_printed_line_stays_within_gbk():
    """输出必须全部落在 GBK 内——控制台是 GBK，一个 emoji 就让整条命令崩掉。"""
    clock = FakeClock()
    progress = make_progress(clock)
    progress.stage("回测引擎", note="4752 个标的 × 2,700 根 K 线")
    clock.advance(30.0)
    progress.tick(1234, 2700, lambda: "2021-03-05")
    progress.stage("落盘", unit="只")
    progress.finish()

    for line in lines(progress):
        line.encode("gbk")  # 编不出来就抛，那正是要拦的


# --- 选股阶段：逐个过滤器上报 ------------------------------------------------


def test_the_screen_reports_each_filter_by_its_own_name(panel):
    """``Screen.apply`` 逐个过滤器上报，且用部件**自己的名字**。

    全市场下一个过滤器就是分钟量级，而此前这一段没有任何输出；名字取自 ``__name__``
    是因为规则在本项目里总写成有名字的局部函数——用序号就分不出是哪一个在慢。
    """

    def tall_enough(frame):
        return frame.fields["close"] > 10.5

    def volume_ok(frame):
        return frame.fields["volume"] > 0

    def score(frame):
        return frame.fields["close"]

    screen = Screen(filters=(tall_enough, volume_ok), factor=score, top_n=1)
    recorder = Recording()
    screen.apply(two_symbol_panel(panel), progress=recorder)

    assert recorder.names == ["过滤器 1/2：tall_enough", "过滤器 2/2：volume_ok", "排序因子：score"]


def two_symbol_panel(panel):
    """两个标的、6 根 K 线的最小面板。"""
    values = {symbol: [10.0 + i for i in range(6)] for symbol in ("sh600000", "sz000001")}
    return panel({"close": values, "volume": {s: [1000] * 6 for s in values}})


def test_a_lambda_filter_still_gets_a_readable_name(panel):
    """匿名函数的名字是 ``<lambda>``——比序号强，认得出「这里是个匿名件」。"""
    screen = Screen(filters=(lambda frame: frame.fields["close"] > 0,))
    recorder = Recording()
    screen.apply(two_symbol_panel(panel), progress=recorder)

    assert recorder.names == ["过滤器 1/1：<lambda>"]


def test_a_nameless_callable_falls_back_to_its_type_name(panel):
    """可调用对象没有 ``__name__``，退回类型名——宁可含糊，也不去猜。"""

    class AlwaysKeep:
        def __call__(self, frame):
            return frame.fields["close"] > 0

    screen = Screen(filters=(AlwaysKeep(),))
    recorder = Recording()
    screen.apply(two_symbol_panel(panel), progress=recorder)

    assert recorder.names == ["过滤器 1/1：AlwaysKeep"]


# --- 引擎阶段：分母、不改变结果 ----------------------------------------------


class BuyEverything(bt.Strategy):
    def next(self):
        if len(self) == 1:
            for data in self.datas:
                self.buy(data=data)


def test_the_engine_reports_a_denominator_that_really_is_the_number_of_ticks(
    make_market, zero_cost_rules
):
    """分母必须是**引擎的 tick 数**（各标的交易日的并集），不是任何单只标的的根数。

    用错分母不会报错，只会让进度条永远到不了 100%——而「到不了 100%」恰恰会被读成
    「卡住了」。这里 A 有 10 根、B 短且晚开始、C 中间缺一天，三者并集是 10 根。
    """
    a = make_market("sh600000", flat_bars(10, 10.0))
    b = make_market("sz000001", flat_bars(6, 20.0, start="2024-01-04"))
    c_full = flat_bars(10, 30.0)
    c = make_market("sz000002", c_full.drop([c_full.index[4]]))

    recorder = Recording()
    result = run_portfolio_backtest(
        [a, b, c],
        BuyEverything,
        cash=100_000.0,
        max_positions=3,
        rules=zero_cost_rules,
        universe_rules=UniverseRules(min_bars=0),
        progress=recorder,
    )

    done, total = recorder.ticks[-1]
    assert total == len(result.equity_curve) == 10
    assert done == total, "进度条到不了 100%——分母与 tick 数不是一回事"
    assert recorder.names == ["复权", "建股票池", "回测引擎"]
    assert recorder.ticks == [(i, 10) for i in range(1, 11)], "每一根都要计数，一根不落"

    # 引擎阶段在第一次 tick 之前有一段**没有逐根进度可言**的装载（backtrader 把全部 K 线
    # 读进 line buffer，此时策略还没被调用）。这一段不能说没就没：它占引擎阶段的一大半，
    # 全市场就是分钟量级。故阶段说明里必须写明「首行进度会晚到」，否则那一段静默会被读成卡死。
    engine_note = dict((name, note) for name, note, _ in recorder.stages)["回测引擎"]
    assert "装载" in engine_note, engine_note


def test_progress_does_not_change_the_backtest_by_a_single_number(make_market, zero_cost_rules):
    """带进度与不带进度，结果必须**逐位相同**。

    上报是只读的；它若动了结果（或反过来把结果动了），那它就不是观察工具而是参与者了
    ——而那会让「有进度时跑出来的数」失去意义。
    """
    a = make_market("sh600000", flat_bars(8, 10.0))
    b = make_market("sz000001", flat_bars(8, 20.0))

    def run(progress):
        return run_portfolio_backtest(
            [a, b],
            BuyEverything,
            cash=100_000.0,
            max_positions=2,
            rules=zero_cost_rules,
            universe_rules=UniverseRules(min_bars=0),
            progress=progress,
        )

    quiet = run(None)
    loud = run(Recording())

    pd.testing.assert_series_equal(quiet.equity_curve, loud.equity_curve)
    pd.testing.assert_frame_equal(quiet.trades, loud.trades)
    pd.testing.assert_frame_equal(quiet.rejected, loud.rejected)
    assert quiet.final_value == loud.final_value


def test_the_engine_closes_its_stage_when_it_returns(make_market, zero_cost_rules):
    """跑完要把最后一个阶段的用时结算出来，否则「引擎花了多久」只能靠外部计时。"""
    recorder = Recording()
    run_portfolio_backtest(
        [make_market("sh600000", flat_bars(4, 10.0))],
        BuyEverything,
        cash=100_000.0,
        rules=zero_cost_rules,
        universe_rules=UniverseRules(min_bars=0),
        progress=recorder,
    )

    assert recorder.finished == 1


# --- 取数阶段：逐标的 --------------------------------------------------------


def test_the_loader_counts_every_candidate_including_the_skipped_ones(fixture_root, gbbq_file):
    """进度数的是**候选**，含被跳过的那只。

    若只数成功的，进度就永远到不了 100%（实测跳过率 21%），而「到不了 100%」会被读成
    「卡住了」——一个因为量错了对象而凭空造出来的假警报。
    """
    symbols = ["sh600000", "sh000001", "sz000001"]  # 中间那只是指数，必定被跳过
    recorder = Recording()

    loaded = load_universe_data(
        symbols, tdx_root=fixture_root, gbbq_path=gbbq_file, progress=recorder
    )

    done, total = recorder.ticks[-1]
    assert total == 3
    assert done == len(loaded.markets) + len(loaded.skipped) == 3
    assert recorder.names == ["取数"]
    assert recorder.stages[0][2] == "只", "取数量的是标的，量词不该是「根」"


# --- 一次性计时：阶段内部的单独一笔，不与阶段相加 ----------------------------


def test_a_timing_line_is_printed_and_lands_in_the_summary():
    """``timing`` 立即打一行，并在汇总里单列。

    为什么要这一类：同一个中间量（如 ``swings``）只算一次、按面板缓存，代价会被记进**第一个
    碰到它的那个阶段**。实测 ``position`` 那个过滤器报 121s，其中 64.8s 是 ``swings`` 的首算；
    而 ``top_calm`` 报的 88~101s 就是整次 ``volume_pattern``，它后面两个因子因为「同一次调用」
    而几乎免费。只读阶段账会得出「position 很贵、pullback_shrink 不要钱」这类错误结论。
    """
    progress = make_progress()
    progress.stage("过滤器 3/8：position")
    progress.timing("swings（按面板缓存）", 64.8)
    progress.finish()

    printed = lines(progress)
    assert any(line == "计时：swings（按面板缓存） 用时 64.8s" for line in printed), printed
    summary = printed[printed.index("===== 耗时汇总 =====") :]
    assert any("swings（按面板缓存）" in line for line in summary), summary


def test_the_summary_keeps_stages_and_timings_in_separate_blocks():
    """汇总**分两块**列，且明说一次性计时不与阶段相加。

    它们是**重叠**的两笔账（计时落在某个阶段内部），故不能合成一块。这一条盯住的就是那个
    「不能相加」在输出里说清楚了——不写清楚，读的人会把两块的数字加起来，于是把同一段时间
    算两遍。
    """
    progress = make_progress()
    progress.stage("过滤器 1/8：trend")
    progress.timing("swings（首算）", 3.0)
    progress.stage("过滤器 2/8：volume")
    progress.finish()

    printed = lines(progress)
    summary = printed[printed.index("===== 耗时汇总 =====") :]
    joined = "\n".join(summary)

    assert "阶段（互不重叠" in joined, summary
    assert "一次性计时（落在上面某个阶段**内部**，故不与阶段相加）" in joined, summary

    # 两块各自排自己的序：阶段那块里不该出现计时的名字，反之亦然。
    stage_block, timing_block = joined.split("一次性计时")
    assert "swings" not in stage_block, stage_block
    assert "过滤器" not in timing_block, timing_block


def test_the_timed_context_manager_reports_through_any_reporter_that_has_timing():
    """``timed`` 走**鸭子类型**：只要上报端有 ``timing`` 就调它，不要求继承什么。"""

    class OnlyTiming:
        def __init__(self):
            self.entries = []

        def timing(self, name, seconds, *, note=""):
            self.entries.append((name, seconds, note))

    reporter = OnlyTiming()
    with timed(reporter, "某中间量的首算", note="按面板缓存"):
        pass

    ((name, seconds, note),) = reporter.entries
    assert name == "某中间量的首算"
    assert note == "按面板缓存"
    assert seconds >= 0.0


def test_the_timed_context_manager_is_a_noop_without_a_timing_hook():
    """``progress`` 为 ``None``、或它没有 ``timing``（测试里那些老假上报端）时**退化为止空**。

    理由与「取时点失败不弄挂回测」同一条：观察手段不得给被观察的对象添麻烦。另外被计时的那段
    代码自己抛错时，异常必须**照常抛出**——计时只是 `finally` 里的记账，不该吞掉它。
    """
    with timed(None, "没有上报端"):  # 不抛
        pass

    class OnlyStage:
        """只有 stage/tick/finish 的旧式假上报端。"""

    with timed(OnlyStage(), "没有 timing 方法"):  # 不抛
        pass

    with pytest.raises(ValueError, match="被计时块里的错"):
        with timed(None, "块里抛错"):
            raise ValueError("被计时块里的错")
