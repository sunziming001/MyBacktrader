"""``b1_screen`` 的两处 B1 专属改动：**盈亏比过滤器**与**盈亏比排序因子**。

**为什么要另造一段合成行情。** 其余五条过滤器（趋势、价格在慢线上、位置、J 值、量能）全
读同一段价格，故要证明「盈亏比这一条真的在筛人」，必须先有一段**五条都过**的行情，再看它被
盈亏比砍掉。真实行情当不了这个底板——套件里没有行情（``realmdata`` 那一组按环境变量跳过，
见 :mod:`tests.test_real_data_smoke`），而合成行情能把落点**钉到某一根**。故这里用网格搜出来
的那一段（搜索脚本 ``.scratch/search_b1_filter_bite.py``）::

    缓跌 40 根（11.20 → 10.00）→ 上涨 90 根（+250%，峰 35.00）
    → 下跌 8 根（−18%）→ 反弹 3 根（+20%）      共 141 根

它有意思的地方是**深跌之后接一段反弹**：反弹把「自峰值以来的最低价」留在下面，于是分母
（到止损还有多远）变大、盈亏比掉下来。逐根的落点是：

=================  ==========  ==========  ==========================================
第几根（0 起）     收盘        盈亏比      它说明了什么
=================  ==========  ==========  ==========================================
134 ~ 136          31.85 →     6.62 ~      六条过滤器**全过**（含盈亏比 > 4）
                   30.27       10.44
137                28.70       —（缺失）    收在黄线下方（28.70 < 28.77），
                                          亏头为负 → 比值无含义
138                30.61       2.43        **其余五条全过，唯独盈亏比不达标**
139 ~ 140          32.53 →     0.68 →      自峰值只回落 7.1% / 1.6%，
                   34.44       0.10        「位置」那条要求 ≥ 8%
=================  ==========  ==========  ==========================================

于是「第 138 根」就是那条断言要用的格子：**把门槛放开到负数它就会被选上，按默认的 4.0 就
不会**。这两件事同时成立，才说明过滤器在筛人而不是恰好没有候选。
"""

from __future__ import annotations

import inspect

import numpy as np
import pytest

from examples.strategies import B1, b1_screen
from mbt.signals import reward_risk_ratio, swings, yellow_proximity

WINDOWS = (14, 28, 57, 114)

#: 六条过滤器全过的根（含盈亏比）。
PASSING_BARS = (134, 135, 136)

#: 其余五条全过、但盈亏比只有 2.43（< 4.0）的那一根——过滤器的落点。
REJECTED_BAR = 138


def climb_fall_bounce():
    """造出上面说明表里的那段行情：缓跌 → 大涨 → 深跌 → 反弹。返回 ``(close, volume)``。"""
    pre = np.linspace(11.20, 10.00, 40)  # 先跌一段，好让摆动点先确认一个谷
    up = np.linspace(10.00, 35.00, 91)[1:]  # 峰 = 35.00（+250%）
    down = np.linspace(35.00, 28.70, 9)[1:]  # 回撤 18%
    bounce = np.linspace(28.70, 34.44, 4)[1:]  # 再反弹 20%——前低就留在下面了
    close = np.concatenate([pre, up, down, bounce])

    warm = len(up) // 2
    volume = np.concatenate(
        [
            np.full(40, 1000.0),
            np.linspace(1000.0, 6000.0, warm),  # 上涨放量
            np.full(len(up) - warm, 6000.0),
            np.full(len(down) + len(bounce), 300.0),  # 回调缩量
        ]
    )
    return close, volume


@pytest.fixture
def b1_setup(panel):
    """上面那段行情的行情面板：``high = close × 1.005``、``low = close × 0.995``。"""
    close, volume = climb_fall_bounce()
    return panel(
        {
            "open": {"sh600000": close},
            "high": {"sh600000": close * 1.005},
            "low": {"sh600000": close * 0.995},
            "close": {"sh600000": close},
            "volume": {"sh600000": volume},
        }
    )


def selected_bars(screen, b1_setup):
    """``screen`` 在这段行情上选中的根号（0 起）列表。"""
    result = screen.apply(b1_setup)
    return [position for position, picked in enumerate(result.selected.iloc[:, 0]) if bool(picked)]


def test_the_five_other_filters_alone_do_keep_the_rejected_bar(b1_setup):
    """自检前提：把盈亏比门槛放开到负数，第 138 根**会被选上**。

    没有这条，下面那条「默认不选它」可能只是因为它在别处就已经出局，而用例看不出来。
    ``min_reward_risk`` 取负数等于只保留「比值非缺失」，故它恰好把盈亏比这一条**摘掉**。
    """
    kept = selected_bars(b1_screen(min_reward_risk=-1.0), b1_setup)

    assert REJECTED_BAR in kept, "其余五条在这里本就放行——不然这条过滤器无从测起"
    assert all(bar in kept for bar in PASSING_BARS), "盈亏比 > 4 的那几根不该被负数门槛改变"


def test_the_default_threshold_drops_the_bar_the_other_filters_keep(b1_setup):
    """默认门槛（4.0）下，第 138 根被砍掉，而 134~136 留下——**砍的不是一批，是一格**。

    这一条是「过滤器已接上」的直接证据：若实现里漏掉 ``worth_the_risk``，第 138 根会重新
    出现；若实现把门槛写死成别的值，落点会移位。
    """
    kept = selected_bars(b1_screen(), b1_setup)

    assert REJECTED_BAR not in kept, "盈亏比只有 2.43，默认门槛 4.0 应当把它砍掉"
    assert list(kept) == list(PASSING_BARS), f"存活的不该变，实际 {kept}"


def test_a_threshold_above_every_ratio_selects_nothing(b1_setup):
    """门槛高到没有一格够得着时选出**空集**——过滤器不会被当成「没给」而放过。

    这一条与上一条测的是两个不同的失效方式：上一条管「该砍的那一格砍没砍」，这一条管
    「一条什么都不放行的过滤器会不会被静默忽略」。
    """
    assert selected_bars(b1_screen(min_reward_risk=1e9), b1_setup) == []


def test_the_ranking_factor_is_the_reward_risk_ratio_not_yellow_proximity(b1_setup):
    """排序因子的**逐格取值**就是 :func:`reward_risk_ratio`，且不再是黄线贴近度。

    这里比的是数值而不是名次：名次要靠两只标的才谈得上，而因子的横向可比性已经由
    ``tests/test_signal_factors.py`` 单独钉过。取值相同即「同一个量」，比名次更严。
    """
    anchors = swings(b1_setup["close"], retracement=0.08)
    # 0.01 就是 b1_screen 的默认 stop_buffer（它与策略一致这件事由最后一条用例钉住）。
    expected = reward_risk_ratio(b1_setup, anchors, windows=WINDOWS, stop_buffer=0.01).to_numpy()

    got = b1_screen().factor(b1_setup).to_numpy()
    assert np.allclose(got, expected, equal_nan=True), "排序因子不是按公式算的那个比值"

    proximity = yellow_proximity(b1_setup["close"], WINDOWS).to_numpy()
    assert not np.allclose(got, proximity, equal_nan=True), "排序因子还是旧的黄线贴近度"


def test_the_screen_and_the_strategy_share_the_stop_buffer():
    """``b1_screen`` 与策略 ``B1`` 的 ``stop_buffer`` 默认值必须一致——两边不一致就是静默失真。

    比的是**各自的默认值**而不是算出来的数：比值的分母是「前低 × (1 − stop_buffer)」，
    两处取不同的值会让选股时度量的是一条策略**不会执行**的止损。
    """
    screen_default = inspect.signature(b1_screen).parameters["stop_buffer"].default
    strategy_default = B1.params.stop_buffer

    assert screen_default == strategy_default
