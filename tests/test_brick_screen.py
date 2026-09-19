"""砖型规则的过滤器与排序因子（票据 #85；后两项因子 2026-09-19 加）。

**这条规则就是那三条判据加四项读数**，故这里测的是「它算的是不是那件事」，不是「它算得准不
准」——砖型图本身的数值在 ``tests/test_signal_brick.py`` 里另有手算常数与对拍钉住。

各用例的证据各有来路，刻意不重复：

- **判据的同一性**用一份长行情（走出一段段涨跌，故红绿砖真的交替），把规则的 ``selected``
  与**照需求直接写的三条布尔表达式**逐格对齐。三条判据各自还配一条「它真的在筛人」的守卫
  （那一类格子必须存在且非空），否则一次全假的比对也能过。
- **0 底那一段**用一段定制行情：先阴跌到 0 底、再抬头、再小幅回落、最后猛抬头。它把
  ADR-0016 那个后果钉到规则这一层——底部连续为 0 意味着**没有绿砖**，故从 0 底抬起来的
  第一根红砖**永远选不中**，而那不是瑕疵，是截断照字面沿用的直接结果。
- **配权**分两层验：一条用**独立重算的算式**（``mean_percentile``）比全表，一条用**桩读数**
  把其余三项钉死、只让一项在两只标的之间不同——后者才能把「某一项的方向与份额」分开验干净
  （真实行情里四项同时变，涨得多的那天砖也更大、量也更高）。
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from mbt.data import Panel
from mbt.screen import BRICK_BOARDS, Screen, brick_screen
from mbt.signals import above_yellow, brick_line, momentum, upper_shadow_atr, volume_ratio
from mbt.universe import CHINEXT, MAIN_BOARD, STAR_MARKET

#: 砖型那条上影读数用的 ATR 窗口——与 B1 那个 `top_shadow_atr` 同口径，故两条规则可比。
BRICK_ATR_N = 14

#: 「收在黄线之上」那道门用的窗口。与 B1 **同一组**——`CONTEXT.md` 把黄线定义成一个概念，
#: 两处取不同窗口就等于造了第二条黄线。
YELLOW_WINDOWS = (14, 28, 57, 114)

#: 一份长行情里落子的那一根（见 :func:`floor_panel`）——三条判据在此**同时**成立。
FLOOR_SERIES_PICK = pd.Timestamp("2024-02-05")

#: 三个板块的名字必须齐全（需求写的是「主板、创业板、科创板」）。
EXPECTED_BOARDS = frozenset({MAIN_BOARD, CHINEXT, STAR_MARKET})


def symbol_frame(values, start="2024-01-02"):
    """``{标的: 序列}`` → 标的宽表；日期用交易日。"""
    index = pd.bdate_range(start, periods=len(next(iter(values.values()))))
    return pd.DataFrame(values, index=index)


def wave_panel(*, symbols=3, bars=600, seed=20260919) -> Panel:
    """一段段涨跌交替的合成行情——砖型图因此红绿相间，三条判据都有落点。

    种子写死：这一份行情是**夹具**，不是随机抽样。换种子会让下面那些「守卫」数字变，而那些
    数字正是「这条测试没空转」的证据。
    """
    rng = np.random.default_rng(seed)
    fields: dict[str, dict[str, np.ndarray]] = {
        "open": {},
        "high": {},
        "low": {},
        "close": {},
        "volume": {},
    }
    for i in range(symbols):
        name = f"sh60{i:04d}"
        legs = []
        while sum(len(leg) for leg in legs) < bars:
            span = int(rng.integers(8, 26))
            drift = rng.normal(0.0, 0.012)
            legs.append(np.exp(np.cumsum(np.full(span, drift) + rng.normal(0, 0.006, span))))
        close = np.concatenate(legs)[:bars] * 10.0
        fields["close"][name] = close
        fields["open"][name] = close * (1.0 + rng.uniform(-0.01, 0.01, bars))
        fields["high"][name] = close * (1.0 + rng.uniform(0.0, 0.02, bars))
        fields["low"][name] = close * (1.0 - rng.uniform(0.0, 0.02, bars))
        fields["volume"][name] = 1000.0 * np.exp(rng.normal(0.0, 0.5, bars))
    return Panel({field: symbol_frame(values) for field, values in fields.items()})


def floor_panel(*, symbol="sh600000") -> Panel:
    """跌到 0 底 → 抬头 → 小幅回落（出现绿砖）→ 更猛地抬头（红砖更大）。

    它同时给出两种落点：抬头那几根**在 0 底上**（前一根是平的，不是绿砖），而
    :data:`FLOOR_SERIES_PICK` 那根是「绿砖之后一根更大的红砖」。
    """
    closes = np.concatenate(
        [
            np.linspace(20.0, 10.0, 16),  # 一路阴跌，砖型图落到 0 底
            np.linspace(10.2, 11.4, 6),  # 抬头
            np.array([11.35, 11.30]),  # 只回落两根：绿砖很小
            np.linspace(11.9, 15.5, 6),  # 更猛地抬头：红砖比那根绿砖大
        ]
    )
    fields = {
        "open": {symbol: closes * 0.995},
        "high": {symbol: closes + 0.2},
        "low": {symbol: closes},
        "close": {symbol: closes},
        "volume": {symbol: np.full(closes.size, 1000.0)},
    }
    return Panel({field: symbol_frame(values) for field, values in fields.items()})


def every_symbol(field: pd.DataFrame) -> pd.DataFrame:
    """全池掩码——这些用例要量的是判据本身，不是股票池那一步。"""
    return pd.DataFrame(True, index=field.index, columns=field.columns)


def mean_percentile(panel: Panel, pool: pd.DataFrame, *, volume_window: int = 1) -> pd.DataFrame:
    """照 ``Screen`` 的口径手算一遍砖型的分数。

    这一版是**四项、权重 1/3 / 1/3 / 1/6 / 1/6**：砖的大小之比与量比各占三分之一，而「当根
    涨幅」与「当根上影」两条**合成**那第三个三分之一，各占六分之一。等价于权重写
    ``(1, 1, 0.5, 0.5)``（``Screen`` 会归一到和为 1）。

    它是**独立的第二份算式**（这里逐行写开，``Screen`` 那边另有实现），用来回答「喂进去的是
    这四个数、各占多少」。四项都缺的格子留缺失——不是记 0 分。
    """
    size_ratio = brick_line(panel).size_ratio
    volume = volume_ratio(panel["volume"], n=volume_window)
    gain = momentum(panel["close"], 1)
    shadow = upper_shadow_atr(panel, BRICK_ATR_N)

    def percentile_of(frame):
        pooled = frame.where(pool)
        ranks = pooled.rank(axis=1, ascending=True)
        counts = pooled.notna().sum(axis=1)
        return ranks.div(counts.replace(0, np.nan), axis=0).fillna(0.0)

    # 全部项都缺 → 留缺失；只缺一项的记 0 分外位（与 `Screen._combine` 同一处置）。
    every = [size_ratio, volume, gain, shadow]
    missing_all = every[0].isna()
    for frame in every[1:]:
        missing_all &= frame.isna()

    # 四条读数都翻成「越大越靠前」：砖的大小之比与量比**本来就是这个方向**，而涨幅与上影
    # 是「越小越好」，故那两条要取负。权重按上面的份额。
    weighted = (
        percentile_of(size_ratio) / 3.0
        + percentile_of(volume) / 3.0
        + percentile_of(-gain) / 6.0
        + percentile_of(-shadow) / 6.0
    )
    return weighted.where(~missing_all)


def conditions(panel: Panel):
    """**照需求直接写**的四条判据（不经过 ``Screen`` 的机制）。

    前三条刻意用**下标错位**来写「前一根是绿砖」，而不是复用过滤器里的 ``shift``——两条路各写
    一遍，对上了才说明规则的组合没写错。第三条同样照需求写「大小之比 > 1」，而不是复用 ``size``
    的相邻比较。第四条照需求写「收盘 > 黄线」。
    """
    readings = brick_line(panel)
    line, size_ratio = readings.line, readings.size_ratio
    previous_is_green = (line.shift(1) < line.shift(2)).fillna(False)
    today_is_red = (line > line.shift(1)).fillna(False)
    this_red_is_bigger = (size_ratio > 1.0).fillna(False)
    above_the_yellow = above_yellow(panel["close"], YELLOW_WINDOWS).fillna(False)
    return previous_is_green, today_is_red, this_red_is_bigger, above_the_yellow


# --- 四条判据：规则的选定集就是它们的交 ---------------------------------------


def test_the_selection_is_exactly_the_four_conditions():
    """``selected`` 与四条判据的交**逐格相同**——规则没有多一条、也没有少一条。"""
    price_panel = wave_panel()
    result = brick_screen().apply(price_panel, universe_mask=every_symbol(price_panel["close"]))

    previous_is_green, today_is_red, bigger, above_the_yellow = conditions(price_panel)
    expected = previous_is_green & today_is_red & bigger & above_the_yellow

    # 守卫：四类落点都真的存在，否则「逐格相同」可能只是在比对两张全假的表。
    assert int(previous_is_green.to_numpy().sum()) > 100, "样本里几乎没有绿砖"
    assert int((previous_is_green & today_is_red).to_numpy().sum()) > 100, "样本里几乎没有绿转红"
    assert int(expected.to_numpy().sum()) > 3, "样本里几乎没有选中格"
    assert (
        int((previous_is_green & today_is_red & ~bigger).to_numpy().sum()) > 50
    ), "样本里没有「绿转红但红砖更小」的格子——第三条判据就没被验到"
    assert int(above_the_yellow.to_numpy().sum()) > 100, "样本里几乎没有「收在黄线之上」的格子"

    np.testing.assert_array_equal(result.selected.to_numpy(), expected.to_numpy())


def test_a_bar_that_passes_the_brick_conditions_but_not_the_yellow_gate_is_rejected():
    """前三条都成立，只差「收在黄线之上」——被第四条挡掉。

    这一条单独在筛人，而它恰恰是最容易「顺手去掉」的那一条（它看着像多余的确认）。没有这条
    测试，改坏了不会有人发现——因为前三条的测试全都还在绿。
    """
    price_panel = wave_panel()
    pool = every_symbol(price_panel["close"])
    result = brick_screen().apply(price_panel, universe_mask=pool)

    previous_is_green, today_is_red, bigger, above_the_yellow = conditions(price_panel)
    brick_only = previous_is_green & today_is_red & bigger
    bite = brick_only & ~above_the_yellow

    assert int(bite.to_numpy().sum()) > 3, "样本里没有这一类格子，这条测试将空转"
    assert not (result.selected.to_numpy() & bite.to_numpy()).any(), "没站上黄线却被选中了"

    # 而放宽这道门之后，那些格子**应当**被选中——否则上面那句可能是别的原因造成的。
    relaxed = brick_screen(yellow_windows=(2,)).apply(price_panel, universe_mask=pool)
    assert (
        relaxed.selected.to_numpy() & bite.to_numpy()
    ).any(), "放宽这道门之后照样选不中——那说明挡掉它的不是这道门"


def test_a_green_to_red_whose_red_is_not_bigger_is_rejected():
    """绿砖之后确实是红砖，而红砖**更小**——不合格。第三条判据单独在筛人。"""
    price_panel = wave_panel()
    result = brick_screen().apply(price_panel, universe_mask=every_symbol(price_panel["close"]))

    previous_is_green, today_is_red, bigger, _ = conditions(price_panel)
    bite = previous_is_green & today_is_red & ~bigger

    assert int(bite.to_numpy().sum()) > 50, "样本里没有这一类格子，这条测试将空转"
    assert not (result.selected.to_numpy() & bite.to_numpy()).any(), "红砖更小却被选中了"


def test_a_green_brick_not_followed_by_a_red_one_is_rejected():
    """前一根是绿砖、当根**不是红砖**（继续绿、或两根持平）——不合格。"""
    price_panel = wave_panel()
    result = brick_screen().apply(price_panel, universe_mask=every_symbol(price_panel["close"]))

    previous_is_green, today_is_red, _, _ = conditions(price_panel)
    bite = previous_is_green & ~today_is_red

    assert int(bite.to_numpy().sum()) > 50, "样本里没有这一类格子，这条测试将空转"
    assert not (result.selected.to_numpy() & bite.to_numpy()).any()


def test_a_red_brick_whose_previous_bar_was_not_green_is_rejected():
    """当根是红砖、比值也大于 1，而**前一根不是绿砖**（它也在涨，或持平）——不合格。

    这一条挡的是「连续两根红砖」那类形态：那种上涨与需求里那句「前一天绿砖」无关。
    """
    price_panel = wave_panel()
    result = brick_screen().apply(price_panel, universe_mask=every_symbol(price_panel["close"]))

    previous_is_green, today_is_red, bigger, _ = conditions(price_panel)
    bite = ~previous_is_green & today_is_red & bigger

    assert int(bite.to_numpy().sum()) > 50, "样本里没有这一类格子，这条测试将空转"
    assert not (result.selected.to_numpy() & bite.to_numpy()).any()


# --- 「恰好相等」与「两根持平」这两处边界 --------------------------------------
#
# 这两格靠**真实行情构造不出来**：砖型图的每个值都由公式递推算出，让两根砖的大小恰好相等
# 需要一串巧合（而「恰好相等」在浮点下还有一层不可控）。故这里换掉**砖型图本身**，拿一条
# 手写的线喂进规则——测的仍是规则（它怎么判那三件事），而砖型图算得对不对另有测试钉住
# （`tests/test_signal_brick.py`）。


def stubbed_line(monkeypatch, values: dict[str, list[float]], **rule_options) -> Screen:
    """把 ``brick_line`` 换成一条**手写的线**，再照常造出砖型规则。

    必须在 ``brick_screen()`` **之前**换：规则把 ``brick_line`` 收进闭包，造好之后换就晚了。

    ``rule_options`` 透给 ``brick_screen``（例如把「收在黄线之上」那道门的窗口收到 ``(2,)``），好让
    这些用例把要测的那一处**单独**暴露出来。
    """
    from mbt.data.panel import Panel as PanelType
    from mbt.signals import BrickLine

    def fake(panel: PanelType) -> BrickLine:
        index = next(iter(panel.fields.values())).index
        line = pd.DataFrame(values, index=index)
        return BrickLine(
            line=line,
            size=line.diff().abs(),
            size_ratio=line.diff().abs() / line.diff().abs().shift(1),
        )

    import mbt.signals as signals_module

    monkeypatch.setattr(signals_module, "brick_line", fake)
    return brick_screen(**rule_options)


def apply_stub(
    monkeypatch,
    values,
    symbol="sh600000",
    *,
    yellow_windows=(2,),
    gate_passes=None,
):
    """造一条手写的线、跑一遍规则，返回逐日的 ``selected``（一个 ``Series``）。

    ``yellow_windows`` 默认收到 **(2,)**：这些用例测的是砖型那三条判据，而它们手写的线都很短，
    默认窗口（40 根）根本不满、会把每一格都筛掉——于是「红砖更小」那类断言会因为**门**而
    通过，而不是因为判据。窗口 1 根时门只看「当根收盘 > 前一根最高价」，用一根价格单调
    上升的线就能确定地放行。

    ``gate_passes`` 给出**应当**通过这道门的行号；给了就当场断言，免得上面那件事悄悄反过来
    （门把整条线都筛掉时，测试看着仍是绿的）。
    """
    index = pd.bdate_range("2024-01-02", periods=len(next(iter(values.values()))))
    # 价格面板与手写的砖型线**互相独立**：线是桩给的，故价格只负责让那道门按预期放行。
    # 取单调上升的收盘价（`close = 20 + i`），最高价与之相同——于是「当根收盘 > 前一根最高价」
    # 在每一格都成立。
    closes = [20.0 + i for i in range(len(index))]
    frame = pd.DataFrame({symbol: closes}, index=index)
    price_panel = Panel({field: frame.copy() for field in ("open", "high", "low", "close")})
    price_panel = Panel({**price_panel.fields, "volume": frame.copy()})

    rule = stubbed_line(monkeypatch, values, yellow_windows=yellow_windows)
    universe = every_symbol(price_panel["close"])
    selected = rule.apply(price_panel, universe_mask=universe).selected[symbol]

    if gate_passes is not None:
        gate = above_yellow(price_panel["close"], yellow_windows)[symbol]
        for row in gate_passes:
            assert bool(
                gate.iloc[row]
            ), f"第 {row} 行没通过「收在黄线之上」那道门——这条测试会变成在测门，不是在测砖型"
    return selected


def test_a_red_brick_exactly_as_big_as_the_green_one_is_rejected(monkeypatch):
    """**恰好相等即不合格**——第三条判据是严格的「大于 1」。

    手写的线：``10 → 6 → 10``。第二根是绿砖（大小 4），第三根是红砖（大小 4），比值**恰好**
    是 1，故第三根必须被筛掉；把实现改成 ``>= 1`` 这条就会红。
    """
    selected = apply_stub(monkeypatch, {"sh600000": [np.nan, np.nan, 10.0, 6.0, 10.0]})

    assert not bool(selected.iloc[4]), "比值恰好等于 1 的红砖被选中了——判据不是严格的"


def test_a_red_brick_one_cent_bigger_than_the_green_one_is_taken(monkeypatch):
    """对照组：比值只比 1 大一丁点就算数——上一条不是「整条规则不选人」造成的。

    线是 ``10 → 6 → 10.01``：绿砖大小 4、红砖大小 4.01。

    这条与它的上一条都**故意把「收在黄线之上」那道门的窗口收到 ``(2,)``**——它们要测的是比值那
    一处的严格性，故其余三道门必须**确定地放行**（窗口 1 根时「当根收盘 > 前一根最高价」，
    而这里前一根恰好更低）。下面那句 ``gate_passes`` 断言把这件事钉住：门若不放行，这条测试
    就变成在测门，而不是在测比值。
    """
    values = {"sh600000": [np.nan, np.nan, 10.0, 6.0, 10.01]}
    selected = apply_stub(monkeypatch, values, yellow_windows=(2,), gate_passes=[4])

    assert bool(selected.iloc[4]), "只大一点点就该算「更大」"


def test_a_flat_pair_is_neither_red_nor_green_so_it_is_rejected(monkeypatch):
    """两根持平那一格**既非红也非绿**，故「前一根是绿砖」与「当根是红砖」同时不成立。

    手写的线：``10 → 6 → 6``。第三根与前一根持平，而图上那一格不画砖——它不该被选中。
    """
    selected = apply_stub(monkeypatch, {"sh600000": [np.nan, np.nan, 10.0, 6.0, 6.0]})

    assert not bool(selected.iloc[4]), "持平那根被当成红砖了"


# --- 0 底那一段（ADR-0016 记下的后果） ----------------------------------------


def test_a_brick_rising_off_the_zero_floor_is_never_selected():
    """**从 0 底抬起来的第一根红砖永远选不中**——因为前一根是平的，不是绿砖。

    这不是瑕疵，是 ADR-0016 那个截断照字面沿用的直接结果：底部连续多根为 0 ⇒ 那一段
    **既无红砖也无绿砖** ⇒ 「前一天绿砖」在那里结构上不可能成立，与标的当天的强弱无关。
    """
    price_panel = floor_panel()
    result = brick_screen().apply(price_panel, universe_mask=every_symbol(price_panel["close"]))

    line = brick_line(price_panel).line["sh600000"]
    lift_off = line.notna() & line.shift(1).eq(0) & (line > 0)

    assert lift_off.sum() > 0, "样本里没有「从 0 底抬起」的那一根，这条测试将空转"
    assert not result.selected["sh600000"][lift_off].any(), "0 底抬起的那一根不该被选中"


def test_the_pattern_is_selectable_once_the_series_is_off_the_floor():
    """离开 0 底之后，同一条规则照常选出「绿砖之后一根更大的红砖」——规则不是全废的。

    它与上一条成对：一条证明底部选不中，一条证明底部之外选得中。少一条都会让另一条读歪
    （只有前者，看着像规则坏了；只有后者，看不到截断的代价）。

    **这道门的窗口在这里收到 ``(2,)``**，而这一点是刻意的：这段行情只有二十来根，而出厂默认的
    黄线最长那条均线要 **114 根**——窗口填不满时黄线整段缺失、这道门会把每一格都筛掉。这里要
    验的是砖型那三条判据在离开 0 底之后能成立，故把门收到 ``(2,)``、并断言它确实放行，而不是
    让它替这段行情做判断。
    """
    price_panel = floor_panel()
    rule = brick_screen(yellow_windows=(2,))
    result = rule.apply(price_panel, universe_mask=every_symbol(price_panel["close"]))

    gate = above_yellow(price_panel["close"], (2,))["sh600000"]
    assert bool(gate.at[FLOOR_SERIES_PICK]), "门没有放行——那这条测试会变成在测门"

    assert bool(result.selected.at[FLOOR_SERIES_PICK, "sh600000"]), "这一根本该被选中"


# --- 配权：四项，前两项各 1/3、后两项各 1/6 ------------------------------------


def test_the_score_is_the_weighted_average_of_the_four_readings():
    """分数 = **四项**的池内百分位加权平均，权重 **1/3、1/3、1/6、1/6**。

    这一条同时钉住三件事：用的是哪四项、方向都翻正了、以及各占多少。百分位与「全都缺才留
    缺失」这两条口径本身由 ``tests/test_screen.py`` 钉住（``normalize="rank"``），这里在它
    的基础上验「喂进去的是这四个数、按这个份额相加」。

    它与下面那条**桩读数**的用例分工不同：这条用真实读数跑全流程（证明四条真的接上了），
    那条把其余三项钉死（证明**某一项**的方向与份额）。
    """
    price_panel = wave_panel()
    pool = every_symbol(price_panel["close"])
    result = brick_screen().apply(price_panel, universe_mask=pool)

    assert not result.scores.empty, "规则没有产出分数"
    pd.testing.assert_frame_equal(
        result.scores, mean_percentile(price_panel, pool), check_dtype=False
    )


# --- 用**桩读数**把四项分开验 --------------------------------------------------
#
# 真实行情里四项同时变，故「某一项的方向」验不干净：涨得多的那天，砖也更大、量也更高。
# 这里把四条读数换成手写的表，只让**一项**在两只标的之间不同，其余三项完全相同——于是分数
# 的差只可能来自那一项。

#: 桩读数用的两个标的。
STUB_SYMBOLS = ("sh600000", "sh600001")

#: 桩面板的日期数（够长即可，读数全是桩给的，值不重要）。
STUB_BARS = 6


def stub_panel() -> Panel:
    """桩用例用的面板：形状对就行，四条读数全由 :func:`stubbed_screen` 顶掉。"""
    index = pd.bdate_range("2024-01-02", periods=STUB_BARS)
    values = {symbol: [10.0] * STUB_BARS for symbol in STUB_SYMBOLS}
    return Panel(
        {
            field: pd.DataFrame(values, index=index)
            for field in ("open", "high", "low", "close", "volume")
        }
    )


def stubbed_screen(monkeypatch, **readings):
    """把四条读数换成手写的表，造出砖型规则。

    必须在 ``brick_screen()`` **之前**换：规则把这几个函数收进闭包，造好之后再换就晚了。
    每个关键字给一张「日期 × 标的」的表（或一对逐标的的序列）。
    """
    import mbt.signals as signals_module
    from mbt.signals import BrickLine

    index = pd.bdate_range("2024-01-02", periods=STUB_BARS)

    def as_frame(value):
        """``{标的: 序列}`` 直接变表；一对 ``(基准值, 被改的值)`` 摊成两只标的的逐日表。

        后者是 :func:`score_for` 给的形状：前 ``STUB_BARS - 1`` 天两只**完全相同**，只有
        最后一天分开——于是分数差只可能来自那一项。
        """
        if isinstance(value, pd.DataFrame):
            return value
        if isinstance(value, dict):
            return pd.DataFrame(value, index=index)
        first, last = value[0], value[-1]
        return pd.DataFrame(
            {
                STUB_SYMBOLS[0]: [first] * STUB_BARS,
                STUB_SYMBOLS[1]: [first] * (STUB_BARS - 1) + [last],
            },
            index=index,
        )

    size_ratio = as_frame(readings["size_ratio"])
    line = as_frame(readings.get("line", {s: [10.0] * STUB_BARS for s in STUB_SYMBOLS}))

    monkeypatch.setattr(
        signals_module,
        "brick_line",
        lambda panel: BrickLine(line=line, size=line.diff().abs(), size_ratio=size_ratio),
    )
    monkeypatch.setattr(
        signals_module, "volume_ratio", lambda volumes, n: as_frame(readings["volume"])
    )
    monkeypatch.setattr(signals_module, "momentum", lambda prices, n: as_frame(readings["gain"]))
    monkeypatch.setattr(
        signals_module, "upper_shadow_atr", lambda panel, n: as_frame(readings["shadow"])
    )
    return brick_screen()


#: 四个读数在「基准」那只标的上都给中间值。
BASELINE = {"size_ratio": 1.0, "volume": 1.0, "gain": 0.0, "shadow": 0.0}


def score_for(monkeypatch, **changed):
    """跑一次桩读数，返回最后一天两只标的的分数 ``(基准那只, 被改的那只)``。

    每一项都是 ``(基准值, 被改的值)``：两只标的在前 ``STUB_BARS - 1`` 天完全相同，只有
    **最后一天**在那一项上分开——于是分数差只可能来自那一项。
    """
    readings = {name: [value] * STUB_BARS for name, value in BASELINE.items()}
    for name, value in changed.items():
        readings[name] = [value[0]] * (STUB_BARS - 1) + [value[1]]

    panel = stub_panel()
    rule = stubbed_screen(monkeypatch, **readings)
    pool = every_symbol(panel["close"])
    scores = rule.apply(panel, universe_mask=pool).scores
    last = scores.index[-1]
    return scores.at[last, STUB_SYMBOLS[0]], scores.at[last, STUB_SYMBOLS[1]]


def test_a_smaller_gain_ranks_higher(monkeypatch):
    """**当根涨幅越小越靠前**——其余三项完全相同，故分数的差只可能来自涨幅。"""
    quiet, hot = score_for(monkeypatch, gain=(0.0, 0.09))

    assert quiet > hot, "涨得多的那只反而排在前面——涨幅那一项的方向写反了"


def test_a_shorter_upper_shadow_ranks_higher(monkeypatch):
    """**当根上影越短越靠前**——上影取正值，故因子那边必须取负。"""
    quiet, spiky = score_for(monkeypatch, shadow=(0.0, 2.0))

    assert quiet > spiky, "上影长的那只反而排在前面——上影那一项的方向写反了"


def test_the_gain_and_the_shadow_each_carry_one_sixth(monkeypatch):
    """两条新读数各占 **1/6**：三项各 1/3，而那第三项由这两条对半构成。

    验法是**同一对标的、只改一项的值**，比较「两只之间的分差」。只有两只标的时，池内百分位
    必是 0.5 与 1.0，故某一项的份额 ``s`` 给出的分差恰是 ``s ÷ 2``：砖那一项（1/3）应当正好
    是新的两条各自（1/6）的**两倍**——份额写错这条立刻红。
    """
    brick_quiet, brick_big = score_for(monkeypatch, size_ratio=(1.0, 2.0))
    gain_quiet, gain_big = score_for(monkeypatch, gain=(0.0, 1.0))
    shadow_quiet, shadow_big = score_for(monkeypatch, shadow=(0.0, 1.0))

    brick_gap = abs(brick_big - brick_quiet)
    gain_gap = abs(gain_big - gain_quiet)
    shadow_gap = abs(shadow_big - shadow_quiet)

    assert gain_gap == pytest.approx(shadow_gap), "两条新读数的份额该相同"
    assert brick_gap == pytest.approx(2 * gain_gap), "砖那一项该是新的两条各自的两倍"


def test_the_volume_factor_divides_by_the_previous_bar_only():
    """量那一项是「当根 ÷ **前一根**」（窗口为 1），不是更长的均量。

    守在这里是因为它最容易悄悄改：换个窗口不会报错，只是名次变了——而「量比」这个词本身
    就容易被读成交易所那个 5 日盘中口径（``CONTEXT.md`` 明令禁用）。
    """
    price_panel = wave_panel()
    pool = every_symbol(price_panel["close"])
    scores = brick_screen().apply(price_panel, universe_mask=pool).scores

    assert not scores.equals(
        mean_percentile(price_panel, pool, volume_window=3)
    ), "把量那一项换成前 3 根，分数却一点没变——它没在算量"


def test_the_atr_window_actually_reaches_the_shadow_reading(monkeypatch):
    """``atr_n`` 是**真的要传下去**的——别让它变成一个没人读的旋钮。

    桩读数认不出窗口，故这条走**真实读数**：同一份行情、两个不同的 ATR 窗口，上影那一项
    必须给出不同的值；若一样，说明窗口根本没传进去。
    """
    price_panel = wave_panel()
    pool = every_symbol(price_panel["close"])

    short_window = upper_shadow_atr(price_panel, 5)
    long_window = upper_shadow_atr(price_panel, 30)
    assert not short_window.equals(long_window), "样品里两个窗口给出同一张表，这条测试将空转"

    # 分数表同样是窗口的函数：换了窗口，那一项变了，分数就该跟着变。
    default_scores = brick_screen().apply(price_panel, universe_mask=pool).scores
    wider_scores = brick_screen(atr_n=5).apply(price_panel, universe_mask=pool).scores
    assert not default_scores.equals(wider_scores), "改了 atr_n 分数却一格没变——窗口没传下去"


def test_top_n_keeps_the_highest_scores():
    """给了 ``top_n`` 就只留分数最高的那几名（逐日）。"""
    price_panel = wave_panel()
    pool = every_symbol(price_panel["close"])
    screen = brick_screen(top_n=2)

    result = screen.apply(price_panel, universe_mask=pool)
    full = brick_screen().apply(price_panel, universe_mask=pool)

    per_day = result.selected.sum(axis=1)
    assert int(per_day.max()) <= 2, "某一天留了不止两名"
    # 留下的必须是全量结果里分数最高的那几个
    kept = result.selected.to_numpy()
    assert int(kept.sum()) > 0, "一个都没选中——这条测试将空转"
    for day in result.selected.index:
        candidates = full.selected.loc[day]
        if not candidates.any():
            continue
        scores = full.scores.loc[day]
        best = scores[candidates].nlargest(min(2, int(candidates.sum()))).index
        assert set(result.selected.loc[day][result.selected.loc[day]].index) == set(best)


# --- 板块范围 -----------------------------------------------------------------


def test_the_rule_declares_main_board_chinext_and_star_market():
    """规则自带板块范围：主板、创业板、科创板，**排除北交所**。

    写在规则上而不是让调用方每次手填——板块范围与「这条规则在哪些市场上成立」是同一件事，
    分开写就会出现「同一条规则在两台机器上跑出不同的池子」，而那不会报错。
    """
    assert brick_screen().boards == EXPECTED_BOARDS
    assert BRICK_BOARDS == EXPECTED_BOARDS
    assert "北交所" not in BRICK_BOARDS
