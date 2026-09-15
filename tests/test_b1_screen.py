"""``b1_screen`` 的 B1 专属改动：过滤器集合与**形态分数排序因子**（ADR-0012）。

**为什么要另造一段合成行情。** 要证明「某一条过滤器真的在筛人」，必须先有一段**其余条都过**
的行情，再看落点被哪一条砍掉。真实行情当不了这个底板——套件里没有行情（``realmdata`` 那一组
按环境变量跳过，见 :mod:`tests.test_real_data_smoke`），而合成行情能把落点**钉到某一根**。
故这里用网格搜出来的那一段（搜索脚本 ``.scratch/search_b1_filter_bite.py``）::

    缓跌 40 根（11.20 → 10.00）→ 上涨 90 根（+250%，峰 35.00）
    → 下跌 8 根（−18%）→ 反弹 3 根（+20%）      共 141 根

它有意思的地方是**深跌之后接一段反弹**：反弹把收盘重新推离黄线，于是「位置」那条（要求自峰值
回落 ≥ 8%）逐根失效。逐根的落点是：

=================  ==========  ==================================================
第几根（0 起）     收盘        它说明了什么
=================  ==========  ==================================================
134 ~ 136          31.06 →     **七条过滤器全过**
                   29.49
137                28.70       收在黄线下方（28.70 < 28.77）→ 被「收盘 > 黄线」挡掉
138 ~ 140          30.61 →     自峰值只回落 7.1% / 1.6% → 被「位置」挡掉（要求 ≥ 8%）
                   34.44
=================  ==========  ==================================================

**134/135 是本文件的关键落点**：它们的交易盈亏比只有 0.84 / 1.75，**低于**当年那条过滤器
的门槛 4.0——也就是说，在「盈亏比 > 4」还在过滤器里时它们买不进来。现在它们通过了，
故这一对格子就是「那条过滤器已被移除」的直接证据（见最后三条用例）。
"""

from __future__ import annotations

import inspect

import numpy as np
import pandas as pd
import pytest

from examples.strategies import B1, b1_screen
from mbt.data import Panel
from mbt.signals import j_oversold, reward_risk_ratio, swings, volume_pattern, yellow_line

WINDOWS = (14, 28, 57, 114)

#: 七条过滤器全过的三根。
PASSING_BARS = (134, 135, 136)

#: 收在黄线下方的那一根——被「收盘 > 黄线」挡掉。
BELOW_LINE_BAR = 137

#: 自峰值回落不足 8% 的那几根——被「位置」挡掉。
SHALLOW_FALL_BARS = (138, 139, 140)

#: 当年那条「交易盈亏比 > 4.0」的门槛。134/135 的比值低于它，故它们现在通过 = 过滤已移除。
REMOVED_REWARD_RISK_THRESHOLD = 4.0


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
    """上面那段行情的面板：``high = close × 1.005``、``low = close × 0.995``，
    外加两个**估值字段**（B1 的 PE 两条过滤器要读，它们来自财务数据而非行情）。

    估值给成**恒值**：全部格 ``pe = 15.0``（正）、``pe_percentile = 0.05``（低于 0.20 门槛），
    故这两条在默认参数下**恒放行**——本文件测的是其余六条的落点，不该被估值搅进来。
    估值那两条自己的用例在文件末尾，各自改这两个值。
    """
    close, volume = climb_fall_bounce()
    bars = len(close)
    return panel(
        {
            "open": {"sh600000": close},
            "high": {"sh600000": close * 1.005},
            "low": {"sh600000": close * 0.995},
            "close": {"sh600000": close},
            "volume": {"sh600000": volume},
            "pe": {"sh600000": [15.0] * bars},
            "pe_percentile": {"sh600000": [0.05] * bars},
        }
    )


def with_valuation(b1_setup, *, pe=None, percentile=None):
    """把面板里的估值字段换掉，其余不动——供估值那两条过滤器自己的用例使用。"""
    fields = dict(b1_setup.fields)
    if pe is not None:
        fields["pe"] = pe
    if percentile is not None:
        fields["pe_percentile"] = percentile

    return Panel(fields)


def selected_bars(screen, b1_setup):
    """``screen`` 在这段行情上选中的根号（0 起）列表。"""
    result = screen.apply(b1_setup)
    return [position for position, picked in enumerate(result.selected.iloc[:, 0]) if bool(picked)]


def test_the_selected_bars_are_the_three_the_remaining_filters_allow(b1_setup):
    """完整规则的落点 = 134/135/136——**不多不少**。

    若哪天有人悄悄加回一条过滤（或把某条的门槛改紧），落点会少；改松会多。故这一条同时
    管两侧，而不只是「至少有东西被选中」。
    """
    kept = selected_bars(b1_screen(), b1_setup)

    assert list(kept) == list(PASSING_BARS), f"落点应当恰好是 {PASSING_BARS}，实际 {kept}"


def test_a_bar_below_the_yellow_line_is_excluded_by_the_price_filter(b1_setup):
    """收盘落在黄线下方的第 137 根被「价格在慢线上」挡掉——那一条仍在，且仍严格。

    它此前是被「盈亏比缺失」挡下的（收盘 ≤ 黄线 ⇒ 亏头为负 ⇒ 比值无含义）。那条过滤器移除
    之后，同一格必须**仍然**被挡住，只是换了一条拦它——否则移除一条会顺带放过这一类格子，
    而那是「移除 A 却动了 B」的静默失真。
    """
    close = b1_setup["close"]["sh600000"]
    yellow = yellow_line(b1_setup["close"], windows=WINDOWS)["sh600000"]

    assert float(close.iloc[BELOW_LINE_BAR]) < float(
        yellow.iloc[BELOW_LINE_BAR]
    ), "构造前提：该根应收在黄线下方"
    assert BELOW_LINE_BAR not in selected_bars(b1_screen(), b1_setup)


def test_the_shallow_fall_bars_are_excluded_by_the_position_filter(b1_setup):
    """自峰值回落不足 8% 的那几根被「位置」挡掉——同样是「移除 A 不该动 B」的对照。

    这几根此前**已经**被位置那条挡住（它们的盈亏比也不达标），故它们在本用例里起的是
    「位置」那条仍未失效的证据。
    """
    kept = selected_bars(b1_screen(), b1_setup)

    assert not set(SHALLOW_FALL_BARS) & set(
        kept
    ), f"回落不足 8% 的根不该入选，实际入选了 {set(SHALLOW_FALL_BARS) & set(kept)}"


def test_the_reward_risk_filter_is_gone_so_low_ratio_bars_now_pass(b1_setup):
    """**本文件的核心断言**：134/135 的交易盈亏比低于当年那条门槛，而它们现在入选。

    这是「盈亏比过滤已移除」的直接证据，而不是间接推断：先算出这两根的比值、断言它确实
    小于当年的门槛 ``REMOVED_REWARD_RISK_THRESHOLD``（4.0），再断言它们**在**落点里。
    两件事同时成立才说明移除生效了——若哪天门槛被加回来（或 ``min_reward_risk`` 被重新接上），
    后者会变红。
    """
    close = b1_setup["close"]
    ratio = reward_risk_ratio(close, white_n=10, windows=WINDOWS)["sh600000"]

    for bar in (134, 135):
        value = float(ratio.iloc[bar])
        assert np.isfinite(value), f"第 {bar} 根应当算得出比值"
        assert (
            value < REMOVED_REWARD_RISK_THRESHOLD
        ), f"第 {bar} 根的比值 {value:.2f} 本应低于门槛 {REMOVED_REWARD_RISK_THRESHOLD}"

    kept = selected_bars(b1_screen(), b1_setup)
    assert set(PASSING_BARS) <= set(
        kept
    ), f"比值低于旧门槛的 {PASSING_BARS} 现在应当入选，实际 {kept}"


def test_the_screen_no_longer_accepts_a_reward_risk_threshold():
    """``min_reward_risk`` 这个旋钮已从签名里去掉——**不留没人读的参数**。

    留着它比去掉更糟：调用方传了它会以为自己在调一条过滤，而那条已经不存在。这与
    ``b1_signals`` 去掉 ``above_white`` / ``trim`` 两个死字段同一处置。
    """
    assert "min_reward_risk" not in inspect.signature(b1_screen).parameters


def test_the_ranking_factor_is_the_three_part_pattern_score(b1_setup):
    """排序接的是 ADR-0012 那个**三项等权形态分数**，而不是 :func:`j_oversold`。

    比的是**逐格原始读数**而不是名次：三项各自的输出必须与独立算出的
    :func:`~mbt.signals.volume_pattern` 读数（取负，上影再截门槛）逐值相同。名次怎么算、
    权重怎么加由 ``tests/test_screen.py`` 单独钉过，这里只管**接线**。
    """
    screen = b1_screen()

    assert screen.factor is None, "单因子那条路不该再被用"
    assert len(screen.factors) == 3, "形态分数是三项"
    assert screen.weights == (1.0, 1.0, 1.0), "三项等权"
    assert screen.normalize == "rank", "合成前要逐日秩归一"

    readings = volume_pattern(
        b1_setup, swings(b1_setup["close"], retracement=0.08), base_bars=10, atr_n=14
    )
    top_calm, pullback_shrink, top_shadow = screen.factors

    assert np.allclose(
        top_calm(b1_setup).to_numpy(), -readings.top_calm.to_numpy(), equal_nan=True
    ), "第一项不是「−顶部无量」"
    assert np.allclose(
        pullback_shrink(b1_setup).to_numpy(),
        -readings.pullback_vs_advance.to_numpy(),
        equal_nan=True,
    ), "第二项不是「−调整缩量」"
    assert np.allclose(
        top_shadow(b1_setup).to_numpy(),
        -np.maximum(0.0, readings.top_shadow_atr.to_numpy() - 1.0),
        equal_nan=True,
    ), "第三项不是「−max(0, 影÷ATR − 1)」"

    assert not np.allclose(
        top_calm(b1_setup).to_numpy(), j_oversold(b1_setup, 9, 3, 3).to_numpy(), equal_nan=True
    ), "排序因子不该还是 −J"

    ratio = reward_risk_ratio(b1_setup["close"], white_n=10, windows=WINDOWS).to_numpy()
    assert not np.allclose(
        top_calm(b1_setup).to_numpy(), ratio, equal_nan=True
    ), "排序因子不该是盈亏比"


def test_the_shadow_component_is_a_penalty_so_short_shadows_score_the_same(b1_setup):
    """上影那一项是**门槛式扣分**：不超过门槛的候选一律记 0，故它们在成分上并列。

    这条盯的是 ADR-0012 写的 ``max(0, 影÷ATR − 1)``——不是未截的 ``影÷ATR``。两者在本
    夹具上都会出现（夹具的 ``high = close × 1.005``，故影很小、绝大多数低于门槛）。截过之后
    「影 0.1 倍」与「影 0.9 倍」得到同一个值，而未截形态下它们不同——这正是这条能判别实现
    的地方。
    """
    screen = b1_screen()
    top_shadow = screen.factors[2]

    got = top_shadow(b1_setup)
    readings = volume_pattern(
        b1_setup, swings(b1_setup["close"], retracement=0.08), base_bars=10, atr_n=14
    )

    # 掩码要**同时**要求「有值」：缺失处 `影 < 1.0` 也是 False。而且必须**挑出**这些格子
    # 来比，不能用 `got.where(below)`——那会把掩码之外的格子留成 NaN，而 `NaN == 0` 为假，
    # 于是断言会因为缺失（而不是因为没截断）变红。
    shadow = readings.top_shadow_atr
    below = (shadow.notna() & (shadow < 1.0)).to_numpy()
    assert below.any(), "夹具里应当有低于门槛的上影——否则这条测不到截断"
    assert (got.to_numpy()[below] == 0.0).all(), "门槛之下必须记 0，而不是 −影÷ATR"


def test_the_screen_and_the_strategy_share_the_two_lines():
    """``b1_screen`` 与策略 ``B1`` 的白线窗口、黄线窗口默认值必须一致。

    比的是**各自的默认值**：盈亏比的分子是白线、分母是黄线，而两条卖出规则读的也是这两条线。
    两处取不同的窗口，选股时度量的就是策略**不会看**的那两条线——不报错，只是结论错。

    上一版这里比的是 ``stop_buffer``；本期把盈亏比的参照换成两条均线之后，那份耦合一并消失，
    新的耦合正是这一条。
    """
    screen_params = inspect.signature(b1_screen).parameters
    assert screen_params["white_n"].default == B1.params.white_n
    assert tuple(screen_params["yellow_windows"].default) == tuple(B1.params.yellow_windows)


# --- 估值两条：PE > 0 且 PE 百分位 < 20% ---------------------------------------


def three_symbols(b1_setup, *, pe, percentile, names=("a", "b", "c")):
    """把单标的夹具复制成三个**列名各异**的标的，各自换掉估值。

    价格三列**完全相同**，故其余六条过滤器的落点对三者一致；差异只可能来自估值。
    直接构造 :class:`~mbt.data.Panel`（而不是走 ``panel`` 夹具）——它收的是
    ``{标的: 序列}``，而这里手里已经是造好的数据帧，只需要换列名。
    """

    from mbt.data import Panel

    fields = {}
    for field, frame in b1_setup.fields.items():
        if field.startswith("pe"):
            continue
        fields[field] = pd.concat(
            [frame.iloc[:, [0]].set_axis([name], axis=1) for name in names], axis=1
        )
    index = b1_setup["close"].index
    fields["pe"] = pd.DataFrame(dict(zip(names, pe, strict=False)), index=index)
    fields["pe_percentile"] = pd.DataFrame(dict(zip(names, percentile, strict=False)), index=index)
    return Panel(fields)


def test_a_non_positive_pe_is_excluded(panel, b1_setup):
    """``PE > 0`` 是**严格**的：负 PE 与 PE = 0 都被挡掉，只有正的放行。

    百分位这三列都设为全放行的 0.05，故落点只可能来自 PE 这一条。这也正是「PE 为正」
    不可省的原因：**非正 PE 照常参与**百分位的 min-max 归一化（见
    :func:`~mbt.data.valuation.pe_percentile`），故亏损股完全可能落在低位、只能被这一条拦住。
    """
    bars = len(b1_setup["close"])
    setup = three_symbols(
        b1_setup,
        pe=([-8.0] * bars, [0.0] * bars, [12.0] * bars),
        percentile=([0.05] * bars, [0.05] * bars, [0.05] * bars),
        names=("loss", "zero", "profit"),
    )

    picked = b1_screen().apply(setup).selected.any(axis=0)

    assert bool(picked["loss"]) is False, "负 PE 必须被挡掉"
    assert bool(picked["zero"]) is False, "PE = 0 不算「为正」（严格比较）"
    assert bool(picked["profit"]) is True, "正 PE 应当放行"


def test_a_pe_percentile_at_or_above_the_threshold_is_excluded(panel, b1_setup):
    """``PE 百分位 < 0.20`` 是**严格**的：恰好 0.20 不算，0.19 才算。

    百分位是 ``[0, 1]`` 上的连续量，写成 ``<=`` 会静默多收一档，而那一档正好压在门槛线上。
    PE 三列都设为放行的 15.0，故落点只可能来自百分位这一条。
    """
    bars = len(b1_setup["close"])
    setup = three_symbols(
        b1_setup,
        pe=([15.0] * bars,) * 3,
        percentile=([0.20] * bars, [0.19] * bars, [0.21] * bars),
        names=("edge", "below", "above"),
    )

    picked = b1_screen().apply(setup).selected.any(axis=0)

    assert bool(picked["edge"]) is False, "恰好 0.20 不算「低于 20%」（严格比较）"
    assert bool(picked["below"]) is True, "0.19 应当放行"
    assert bool(picked["above"]) is False, "0.21 应当被挡掉"


def test_the_two_valuation_thresholds_are_configurable(panel, b1_setup):
    """两条门槛可由调用方改——它们是要按样本标定的量，不该写死在函数体里。"""
    bars = len(b1_setup["close"])
    setup = three_symbols(
        b1_setup,
        pe=([5.0] * bars, [15.0] * bars, [40.0] * bars),
        percentile=([0.10] * bars, [0.10] * bars, [0.10] * bars),
        names=("cheap", "mid", "rich"),
    )

    picked = b1_screen(pe_above=10.0).apply(setup).selected.any(axis=0)

    assert bool(picked["cheap"]) is False, "PE 5 低于新门槛 10，应当被挡掉"
    assert bool(picked["mid"]) is True
    assert bool(picked["rich"]) is True


def test_the_screen_raises_a_clear_error_when_the_valuation_fields_are_missing(b1_setup):
    """面板缺估值字段时报错，且**说明该怎么办**——而不是少一条过滤静默放行。

    这是本规则最容易踩的一脚：行情全都对得上，唯独估值没接上，于是「PE 百分位 < 20%」
    悄悄消失、候选集凭空变大。故报错文本必须指向 `valuation_for` 与 `signals=`，
    而不是只抛一句「没有这个字段」。
    """

    without = Panel({k: v for k, v in b1_setup.fields.items() if not k.startswith("pe")})
    assert "pe" not in without and "pe_percentile" not in without

    with pytest.raises(ValueError, match="valuation_for"):
        b1_screen().apply(without)


def test_the_contained_run_filter_is_wired_and_can_bite(b1_setup):
    """证明「调整期不含横盘串」那条**已接上**且真的会咬人。

    原始夹具证明不了这件事：它的回调段每天收盘都跌破前一根的低点，**一根内含都没有**，
    故门槛收到最严也咬不到。所以这里把回调段改成**横盘**（130 起收到 30.0 附近，相对峰值
    35.00 仍是 14% 的回调，其余条件照旧成立），横盘才会真的生出内含串。

    断言的是**单调性**而不是具体落点：门槛越小，落点只能越少，且收到某一档时全空。这比写死
    几根更稳——它同时证明「接上了」与「方向对」，而写死落点只证明前者。
    """
    import numpy as np

    from mbt.data import Panel

    close, volume = climb_fall_bounce()
    sideways = np.array(close, dtype=float)
    sideways[130:] = 30.0  # 回调段横盘，不再逐日下破
    bars = len(sideways)
    index = b1_setup["close"].index
    fields = {
        "open": pd.DataFrame({"sh600000": sideways}, index=index),
        "high": pd.DataFrame({"sh600000": sideways * 1.005}, index=index),
        "low": pd.DataFrame({"sh600000": sideways * 0.995}, index=index),
        "close": pd.DataFrame({"sh600000": sideways}, index=index),
        "volume": pd.DataFrame({"sh600000": volume}, index=index),
        "pe": pd.DataFrame({"sh600000": [15.0] * bars}, index=index),
        "pe_percentile": pd.DataFrame({"sh600000": [0.05] * bars}, index=index),
    }
    flat = Panel(fields)

    loose = set(selected_bars(b1_screen(contained_days=10), flat))
    tight = set(selected_bars(b1_screen(contained_days=6), flat))
    tighter = set(selected_bars(b1_screen(contained_days=4), flat))

    assert loose, "门槛 10 时应当还有入选——否则这条测的不是「咬人」而是「夹具本身没候选」"
    assert tighter == set(), "门槛收到 4 时应当一个都不放行"
    assert tight <= loose, "门槛变小不该让落点变大"
    assert tighter <= tight

    # 原始夹具上反过来：回调段没有内含，故门槛收到 1 也咬不到——这正是「只看调整期」的
    # 直接体现（不是「历史上出现过就排除」）。
    assert list(selected_bars(b1_screen(contained_days=1), b1_setup)) == list(PASSING_BARS)


def test_the_volume_filter_judges_the_pullback_against_the_spike_not_the_top(b1_setup):
    """门是**旧口径的两半**：``surge_ratio >= 2`` **且** ``pullback_ratio <= max_pullback``，
    后者比的是「回调段 ÷ 上涨段**单日最大量**」，不是「顶部段均量」。

    **这条在 ③ 接线时被删过、由 ④ 的对照退回来**（ADR-0012）：③ 把这道门整条换成了新口径的
    ``surge_vs_base``、并去掉缩量那半条，于是「回调再热闹也能入选」。④ 的全市场对照把那处
    连带改动量出来是**单独 −7.53 pt**——量级是分数那一步收益（+3.01 pt）的三倍、方向相反，
    且它本来就没有独立证据。故退回旧口径，本条随之恢复。

    为什么需要它：夹具原来的上涨段末尾是**平的 6000**，故「顶部段均量」与「上涨段最大量」
    相等，两个口径给出同一个数——换参照物时全套测试没红。这里把量改成**爆量在中段、顶部反而
    安静**，两口径才分家：

    - 上涨段 90 根：前 45 根 1000 → 8000，后 45 根 8000 → 1800（故顶部 3 根系均约 1941）
    - 回调段 11 根：1200

    ====================  ==================  ==========  ==============
    口径                  算式                结果        门槛 0.5
    ====================  ==================  ==========  ==============
    相对上涨段**最大**量   1200 / 8000         0.150       **放行**
    相对**顶部段均量**     1200 / 1940.9       0.618       不放行
    ====================  ==================  ==========  ==============

    故默认参数下**仍应入选**（判据看的是前者）。两半各配一条**双向**断言——收紧哪一半都必须
    变空，否则这一条就退化成「门没在判」而不是「门在放行」。

    **顶部口径为什么不在用**：它实现过、也跑过全市场对照，结果为**更差**——逐笔 8,601 → 3,145、
    收益率均值 +0.164% → +0.116%、均值/标准误 3.90 → 1.68，而被它排掉的 5,969 笔反而更好
    （均值 +0.190%、均值/标准误 +3.74）。见 ADR-0011。
    """
    import numpy as np

    from examples.strategies import b1_screen
    from mbt.data import Panel

    close, _ = climb_fall_bounce()
    bars = len(close)
    volume = np.full(bars, 1000.0)
    # 上涨段 = [40, 129]（夹具里 up 的 90 根），爆量在中段、到顶部已回落
    volume[40:130] = np.concatenate(
        [np.linspace(1000.0, 8000.0, 45), np.linspace(8000.0, 1800.0, 45)]
    )
    volume[130:] = 1200.0  # 回调与反弹

    fields = dict(b1_setup.fields)
    fields["volume"] = pd.DataFrame({"sh600000": volume}, index=b1_setup["close"].index)
    flat = Panel(fields)

    kept = selected_bars(b1_screen(), flat)
    assert list(kept) == list(PASSING_BARS), (
        f"相对上涨最大量是 0.15 <= 0.5、放量 8.0 >= 2.0，两半都过，落点应为 {PASSING_BARS}，"
        f"实际 {kept}——若为空说明判据改看顶部段了"
    )
    assert (
        selected_bars(b1_screen(max_pullback=0.10), flat) == []
    ), "把缩量那半收到 0.10 应当挡住 0.15——否则缩量那半不在门里（③ 删过它）"
    assert (
        selected_bars(b1_screen(min_surge=9.0), flat) == []
    ), "把放量那半收到 9.0 应当挡住 8.0——否则门根本没在判"


def test_the_volume_gate_still_accepts_both_knobs():
    """门的两个旋钮都还在签名里：``volume_edge_bars`` 与 ``max_pullback``。

    ③ 接线时这两个参数被删掉过（门换成只看放量的新口径），④ 的对照把那处连带改动量出来是
    单独 −7.53 pt，故退回旧口径、两个旋钮一并回来。这条盯着它们别再被删掉——**留着没人读的
    参数**固然不好（``min_reward_risk`` 的处置），但**门自己读**它们，删掉就等于把门换了。
    """
    params = inspect.signature(b1_screen).parameters
    assert "volume_edge_bars" in params, "门的「上涨段取几根」旋钮被删了——门被换过"
    assert "max_pullback" in params, "缩量那半的旋钮被删了——门被换成只看放量的新口径了"
