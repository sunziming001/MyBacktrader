"""信号层契约：纯函数、标的宽表进出、**只用评估日及之前**（票据 #5）。

前三个测试文件逐个锁数值；这里锁的是**三类信号共同的行为**，与具体算法无关：

- **时点正确性**用「截断重算不变性」表达——把输入截到第 k 行重算，前 k 行必须与全长
  结果逐值相同。这是个独立于实现的判据：任何「用了未来」的写法都会在这里露馅，
  而不需要作者先想清楚哪里可能漏。
- **纯函数**用「同样输入两次结果相同」与「输入未被就地修改」表达。

信号分两组，因为它们的输入形状不同（ADR-0009）：取**标的宽表**的与取**行情面板**的。
两组的因果性判据相同，只是截断的对象不同。
"""

from __future__ import annotations

import pandas as pd
import pytest

import mbt.signals as signals

#: 取**标的宽表**的信号（``DataFrame → DataFrame``）。
SYMBOL_FRAME_SIGNALS = [
    ("sma", lambda frame: signals.sma(frame, n=3)),
    ("rolling_max", lambda frame: signals.rolling_max(frame, n=3)),
    ("volume_ratio", lambda frame: signals.volume_ratio(frame, n=3)),
    ("new_high", lambda frame: signals.new_high(frame, n=3)),
    ("above_ma", lambda frame: signals.above_ma(frame, n=3)),
    ("ma_cross_up", lambda frame: signals.ma_cross_up(frame, n=3)),
    ("rising_streak", lambda frame: signals.rising_streak(frame, n=3)),
    ("volume_surge", lambda frame: signals.volume_surge(frame, k=2.0, n=3)),
    ("momentum", lambda frame: signals.momentum(frame, n=3)),
    ("distance_to_high", lambda frame: signals.distance_to_high(frame, n=3)),
]

#: 取**行情面板**的信号（跨字段，仍返回标的宽表）。
PANEL_SIGNALS = [
    ("atr", lambda panel: signals.atr(panel, n=3)),
]

#: 八根 K 线、两个标的，含一处停牌造成的缺失——缺失正是因果性最容易出错的地方。
SYMBOL_FRAME_VALUES = {
    "sh600000": [10.0, 11.0, float("nan"), 13.0, 12.0, 14.0, 15.0, 14.0],
    "sz000001": [20.0, 19.0, 18.0, 19.0, float("nan"), 21.0, 22.0, 23.0],
}

#: 面板输入：同一组价量，拆成三个字段。缺失同样保留。
PANEL_VALUES = {
    "high": {
        "sh600000": [10.5, 11.5, float("nan"), 13.5, 12.5, 14.5, 15.5, 14.5],
        "sz000001": [20.5, 19.5, 18.5, 19.5, float("nan"), 21.5, 22.5, 23.5],
    },
    "low": {
        "sh600000": [9.5, 10.5, float("nan"), 12.5, 11.5, 13.5, 14.5, 13.5],
        "sz000001": [19.5, 18.5, 17.5, 18.5, float("nan"), 20.5, 21.5, 22.5],
    },
    "close": SYMBOL_FRAME_VALUES,
}


def test_every_public_signal_is_covered_here():
    """公开了什么就要检查什么——否则新增信号会悄悄绕过契约测试。"""
    covered = {name for name, _ in SYMBOL_FRAME_SIGNALS + PANEL_SIGNALS}
    assert covered == set(signals.__all__)


@pytest.mark.parametrize("name,run", SYMBOL_FRAME_SIGNALS, ids=[n for n, _ in SYMBOL_FRAME_SIGNALS])
def test_a_symbol_frame_signal_never_depends_on_bars_after_the_evaluation_day(
    name, run, symbol_frame
):
    """截断重算不变性：评估日之后发生了什么，都不该改变评估日的值。"""
    frame = symbol_frame(SYMBOL_FRAME_VALUES)
    full = run(frame)

    for k in range(1, len(frame) + 1):
        truncated = run(frame.iloc[:k])

        pd.testing.assert_frame_equal(truncated, full.iloc[:k], check_exact=True)


@pytest.mark.parametrize("name,run", PANEL_SIGNALS, ids=[n for n, _ in PANEL_SIGNALS])
def test_a_panel_signal_never_depends_on_bars_after_the_evaluation_day(name, run, panel):
    """同上，但输入是面板：逐字段一起截断到第 k 行再重算。"""
    bars = len(next(iter(PANEL_VALUES["close"].values())))
    full = run(panel(PANEL_VALUES))

    for k in range(1, bars + 1):
        truncated_values = {
            field: {symbol: values[:k] for symbol, values in per_symbol.items()}
            for field, per_symbol in PANEL_VALUES.items()
        }
        truncated = run(panel(truncated_values))

        pd.testing.assert_frame_equal(truncated, full.iloc[:k], check_exact=True)


def test_the_causality_judgement_itself_catches_a_look_ahead_implementation(symbol_frame):
    """自检：把「用了未来」的实现喂给同一条判据，判据必须变红。

    没有这一条，上面那个参数化测试可能是**空转**的——一条永远不会失败的断言比没有断言
    更糟，因为它会让人以为已经锁住了。
    """

    def peeking_signal(frame: pd.DataFrame) -> pd.DataFrame:
        return frame / frame.mean()  # 故意用了整列（含未来）的均值

    frame = symbol_frame(SYMBOL_FRAME_VALUES)
    full = peeking_signal(frame)

    with pytest.raises(AssertionError):
        pd.testing.assert_frame_equal(
            peeking_signal(frame.iloc[:3]), full.iloc[:3], check_exact=True
        )


def test_signals_are_stateless_so_the_same_input_gives_the_same_output(symbol_frame, panel):
    """同样的输入两次必须给出同样的输出——没有隐藏状态、不读全局配置。"""
    frame = symbol_frame(SYMBOL_FRAME_VALUES)
    for name, run in SYMBOL_FRAME_SIGNALS:
        pd.testing.assert_frame_equal(run(frame), run(frame), check_exact=True, obj=name)

    for name, run in PANEL_SIGNALS:
        # 面板每次都由同一组数值重建，故两次调用之间没有共享的可变状态。
        first = run(panel(PANEL_VALUES))
        second = run(panel(PANEL_VALUES))
        pd.testing.assert_frame_equal(first, second, check_exact=True, obj=name)


def test_signals_do_not_mutate_their_input(symbol_frame, panel):
    """信号层只读输入。就地修改会让调用方的价格表在背后改变，是最难查的一类缺陷。"""
    frame = symbol_frame(SYMBOL_FRAME_VALUES)
    before = frame.copy(deep=True)
    for _, run in SYMBOL_FRAME_SIGNALS:
        run(frame)
    pd.testing.assert_frame_equal(frame, before, check_exact=True)

    built = panel(PANEL_VALUES)
    fields_before = {name: field.copy(deep=True) for name, field in built.fields.items()}
    for _, run in PANEL_SIGNALS:
        run(built)
    for name, field in built.fields.items():
        pd.testing.assert_frame_equal(field, fields_before[name], check_exact=True)
