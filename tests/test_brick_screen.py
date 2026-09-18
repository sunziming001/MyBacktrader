"""砖型规则的过滤器与排序因子（票据 #85）。

**这条规则就是那三条判据加两个因子**，故这里测的是「它算的是不是那件事」，不是「它算得准不
准」——砖型图本身的数值在 ``tests/test_signal_brick.py`` 里另有手算常数与对拍钉住。

各用例的证据各有来路，刻意不重复：

- **判据的同一性**用一份长行情（走出一段段涨跌，故红绿砖真的交替），把规则的 ``selected``
  与**照需求直接写的三条布尔表达式**逐格对齐。三条判据各自还配一条「它真的在筛人」的守卫
  （那一类格子必须存在且非空），否则一次全假的比对也能过。
- **0 底那一段**用一段定制行情：先阴跌到 0 底、再抬头、再小幅回落、最后猛抬头。它把
  ADR-0016 那个后果钉到规则这一层——底部连续为 0 意味着**没有绿砖**，故从 0 底抬起来的
  第一根红砖**永远选不中**，而那不是瑕疵，是截断照字面沿用的直接结果。
- **等权**用「分数 = 两项百分位的平均」验——它同时钉住「用了哪两项」与「各占多少」。
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from mbt.data import Panel
from mbt.screen import BRICK_BOARDS, Screen, brick_screen
from mbt.signals import brick_line, volume_ratio
from mbt.universe import CHINEXT, MAIN_BOARD, STAR_MARKET

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
    fields: dict[str, dict[str, np.ndarray]] = {"high": {}, "low": {}, "close": {}, "volume": {}}
    for i in range(symbols):
        name = f"sh60{i:04d}"
        legs = []
        while sum(len(leg) for leg in legs) < bars:
            span = int(rng.integers(8, 26))
            drift = rng.normal(0.0, 0.012)
            legs.append(np.exp(np.cumsum(np.full(span, drift) + rng.normal(0, 0.006, span))))
        close = np.concatenate(legs)[:bars] * 10.0
        fields["close"][name] = close
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
        "high": {symbol: closes + 0.2},
        "low": {symbol: closes},
        "close": {symbol: closes},
        "volume": {symbol: np.full(closes.size, 1000.0)},
    }
    return Panel({field: symbol_frame(values) for field, values in fields.items()})


def every_symbol(field: pd.DataFrame) -> pd.DataFrame:
    """全池掩码——这些用例要量的是判据本身，不是股票池那一步。"""
    return pd.DataFrame(True, index=field.index, columns=field.columns)


def mean_percentile(panel: Panel, pool: pd.DataFrame, *, volume_window: int) -> pd.DataFrame:
    """照 ``Screen`` 的口径手算一遍分数：两项在池内各转百分位，再等权平均。

    这是**独立的第二份算式**（这里逐行写开，``Screen`` 那边另有实现），用来回答「喂进去的是
    这两个数、各占一半吗」。两项都缺的格子留缺失——不是记 0 分。
    """
    size_ratio = brick_line(panel).size_ratio
    volume = volume_ratio(panel["volume"], n=volume_window)
    missing_both = size_ratio.isna() & volume.isna()

    def percentile_of(frame):
        pooled = frame.where(pool)
        ranks = pooled.rank(axis=1, ascending=True)
        counts = pooled.notna().sum(axis=1)
        return ranks.div(counts.replace(0, np.nan), axis=0).fillna(0.0)

    return ((percentile_of(size_ratio) + percentile_of(volume)) / 2.0).where(~missing_both)


def conditions(panel: Panel):
    """**照需求直接写**的三条判据（不经过 ``Screen`` 的机制）。

    刻意用**下标错位**来写「前一根是绿砖」，而不是复用过滤器里的 ``shift``——两条路各写一遍，
    对上了才说明规则的组合没写错。第三条同样照需求写「大小之比 > 1」，而不是复用 ``size``
    的相邻比较。
    """
    readings = brick_line(panel)
    line, size_ratio = readings.line, readings.size_ratio
    previous_is_green = (line.shift(1) < line.shift(2)).fillna(False)
    today_is_red = (line > line.shift(1)).fillna(False)
    this_red_is_bigger = (size_ratio > 1.0).fillna(False)
    return previous_is_green, today_is_red, this_red_is_bigger


# --- 三条判据：规则的选定集就是它们的交 ---------------------------------------


def test_the_selection_is_exactly_the_three_conditions():
    """``selected`` 与三条判据的交**逐格相同**——规则没有多一条、也没有少一条。"""
    price_panel = wave_panel()
    result = brick_screen().apply(price_panel, universe_mask=every_symbol(price_panel["close"]))

    previous_is_green, today_is_red, bigger = conditions(price_panel)
    expected = previous_is_green & today_is_red & bigger

    # 守卫：三类落点都真的存在，否则「逐格相同」可能只是在比对两张全假的表。
    assert int(previous_is_green.to_numpy().sum()) > 100, "样本里几乎没有绿砖"
    assert int((previous_is_green & today_is_red).to_numpy().sum()) > 100, "样本里几乎没有绿转红"
    assert int(expected.to_numpy().sum()) > 50, "样本里几乎没有选中格"
    assert (
        int((previous_is_green & today_is_red & ~bigger).to_numpy().sum()) > 50
    ), "样本里没有「绿转红但红砖更小」的格子——第三条判据就没被验到"

    np.testing.assert_array_equal(result.selected.to_numpy(), expected.to_numpy())


def test_a_green_to_red_whose_red_is_not_bigger_is_rejected():
    """绿砖之后确实是红砖，而红砖**更小**——不合格。第三条判据单独在筛人。"""
    price_panel = wave_panel()
    result = brick_screen().apply(price_panel, universe_mask=every_symbol(price_panel["close"]))

    previous_is_green, today_is_red, bigger = conditions(price_panel)
    bite = previous_is_green & today_is_red & ~bigger

    assert int(bite.to_numpy().sum()) > 50, "样本里没有这一类格子，这条测试将空转"
    assert not (result.selected.to_numpy() & bite.to_numpy()).any(), "红砖更小却被选中了"


def test_a_green_brick_not_followed_by_a_red_one_is_rejected():
    """前一根是绿砖、当根**不是红砖**（继续绿、或两根持平）——不合格。"""
    price_panel = wave_panel()
    result = brick_screen().apply(price_panel, universe_mask=every_symbol(price_panel["close"]))

    previous_is_green, today_is_red, _ = conditions(price_panel)
    bite = previous_is_green & ~today_is_red

    assert int(bite.to_numpy().sum()) > 50, "样本里没有这一类格子，这条测试将空转"
    assert not (result.selected.to_numpy() & bite.to_numpy()).any()


def test_a_red_brick_whose_previous_bar_was_not_green_is_rejected():
    """当根是红砖、比值也大于 1，而**前一根不是绿砖**（它也在涨，或持平）——不合格。

    这一条挡的是「连续两根红砖」那类形态：那种上涨与需求里那句「前一天绿砖」无关。
    """
    price_panel = wave_panel()
    result = brick_screen().apply(price_panel, universe_mask=every_symbol(price_panel["close"]))

    previous_is_green, today_is_red, bigger = conditions(price_panel)
    bite = ~previous_is_green & today_is_red & bigger

    assert int(bite.to_numpy().sum()) > 50, "样本里没有这一类格子，这条测试将空转"
    assert not (result.selected.to_numpy() & bite.to_numpy()).any()


# --- 「恰好相等」与「两根持平」这两处边界 --------------------------------------
#
# 这两格靠**真实行情构造不出来**：砖型图的每个值都由公式递推算出，让两根砖的大小恰好相等
# 需要一串巧合（而「恰好相等」在浮点下还有一层不可控）。故这里换掉**砖型图本身**，拿一条
# 手写的线喂进规则——测的仍是规则（它怎么判那三件事），而砖型图算得对不对另有测试钉住
# （`tests/test_signal_brick.py`）。


def stubbed_line(monkeypatch, values: dict[str, list[float]]) -> Screen:
    """把 ``brick_line`` 换成一条**手写的线**，再照常造出砖型规则。

    必须在 ``brick_screen()`` **之前**换：规则把 ``brick_line`` 收进闭包，造好之后换就晚了。
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
    return brick_screen()


def apply_stub(monkeypatch, values, symbol="sh600000"):
    """造一条手写的线、跑一遍规则，返回逐日的 ``selected``（一个 ``Series``）。"""
    index = pd.bdate_range("2024-01-02", periods=len(next(iter(values.values()))))
    frame = pd.DataFrame(values, index=index)
    price_panel = Panel({field: frame.copy() for field in ("high", "low", "close", "volume")})
    universe = every_symbol(price_panel["close"])
    return (
        stubbed_line(monkeypatch, values)
        .apply(price_panel, universe_mask=universe)
        .selected[symbol]
    )


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
    """
    selected = apply_stub(monkeypatch, {"sh600000": [np.nan, np.nan, 10.0, 6.0, 10.01]})

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
    """
    price_panel = floor_panel()
    result = brick_screen().apply(price_panel, universe_mask=every_symbol(price_panel["close"]))

    assert bool(result.selected.at[FLOOR_SERIES_PICK, "sh600000"]), "这一根本该被选中"


# --- 两个因子等权 -------------------------------------------------------------


def test_the_score_is_the_average_of_the_two_percentiles():
    """分数 = **砖的大小之比**与**量比**两项在池内百分位的平均。

    这一条同时钉住两件事：用的是哪两项、以及各占一半。百分位与「两项都缺才留缺失」这两条
    口径本身由 ``tests/test_screen.py`` 钉住（``normalize="rank"``），这里在它的基础上验
    「喂进去的是这两个数」。
    """
    price_panel = wave_panel()
    pool = every_symbol(price_panel["close"])
    result = brick_screen().apply(price_panel, universe_mask=pool)

    assert not result.scores.empty, "规则没有产出分数"
    pd.testing.assert_frame_equal(
        result.scores, mean_percentile(price_panel, pool, volume_window=1), check_dtype=False
    )


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
