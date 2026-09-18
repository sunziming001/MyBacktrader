"""砖型图与红绿砖判据（票据 #82）。

两层证据，缺一不可：

- **手算常数**把公式钉住。下面 ``HAND_*`` 那三个值是按公式**手算**出来的，每一步都写在
  注释里，不是照实现抄的（照抄的断言按构造必然通过，量不出任何东西）。
- **独立实现的对拍**只证明「向量化没写错」，**翻转统计**才证明「结论没变」（ADR-0015）。
  故两者分开写：前者要求数值接近，后者要求四处判据**零翻转**。

0 底截断照字面保留（ADR-0016），故它有一组自己的用例——它**会**改变结果，不是等价改写。

手算那一组（六根 K 线，下面各用例共用）::

    bar        0     1     2     3     4     5
    high    10.0  14.0  12.0  16.0  13.0  18.0
    low      8.0   8.0  10.0  10.0  11.0  11.0
    close    9.0   9.0  11.0  11.0  12.0  12.0

    HHV(H,4)   —     —     —   16.0  16.0  18.0      （窗口不满 4 根即缺失）
    LLV(L,4)   —     —     —    8.0   8.0  10.0

    VAR1A = (HHV4 − C) ÷ (HHV4 − LLV4) × 100 − 90： −27.5  −40.0  −15.0
    VAR2A = SMA(VAR1A, 4, 1)：                      −27.5  −30.625  −26.71875
    VAR3A = (C − LLV4) ÷ (HHV4 − LLV4) × 100：       37.5   50.0   25.0
    VAR4A = SMA(VAR3A, 6, 1)：                       37.5   39.58333…  37.15277…
    VAR5A = SMA(VAR4A, 6, 1)：                       37.5   37.84722…  37.73148…
    VAR6A = VAR5A − VAR2A：                          65.0   68.47222…  64.45023…
    砖型图 = VAR6A − 4（VAR6A ≤ 4 时取 0）：         61.0   64.47222…  60.45023…

    分数形式（用例里就按这个写，免得小数位数手工截断）：61、``2321/36``、``52229/864``。
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from mbt.data import TdxDataSource
from mbt.signals import brick_line, green_brick, red_brick

#: 手算那一组 K 线（见模块说明的表）。
HAND_BARS = {
    "high": [10.0, 14.0, 12.0, 16.0, 13.0, 18.0],
    "low": [8.0, 8.0, 10.0, 10.0, 11.0, 11.0],
    "close": [9.0, 9.0, 11.0, 11.0, 12.0, 12.0],
}

#: 手算的砖型图：前 3 根窗口不满 4 根，故无值。
HAND_LINE = [np.nan, np.nan, np.nan, 61.0, 2321 / 36, 52229 / 864]

#: 手算的砖的大小：``2321/36 − 61 = 125/36``；``52229/864 − 2321/36 = 3475/864``。
HAND_SIZE = [np.nan, np.nan, np.nan, np.nan, 125 / 36, 3475 / 864]

#: **实测**的最大相对差（两份实现之间），按样本记。ADR-0015 的规矩是把它**记下来**，因为
#: 「浮点误差可忽略」是一句主张，而它需要一个数。容差取 :data:`TOLERANCE`，比实测宽三个
#: 数量级——它挡的是写错，不是舍入。
MEASURED_MAX_RELATIVE_DIFFERENCE = {"random": 9.6e-16, "fixture": 7.1e-16}

#: 数值对照的容差。**不要求逐位相同**：两条求值路径的差别就是这个量级（ADR-0015）。
TOLERANCE = 1e-12

#: 每份对照样本里至少要有的有效格数——真实切片只有 45 根，显然比随机行情少得多。
MIN_COMPARED = {"random": 900, "fixture": 40}


def hand_panel(panel):
    """由 :data:`HAND_BARS` 造一份单标的面板。"""
    return panel({field: {"sh600000": values} for field, values in HAND_BARS.items()})


def flat_panel(panel):
    """一条**恒定的相对位置**：三根砖型图都等于 161，故相邻之差为 0（既非红也非绿）。

    high / low / close 三者同步平移时，两个位置读数都不变，故递推平滑也不变。
    """
    return panel(
        {
            "high": {"sh600000": [10.0, 11.0, 12.0, 13.0, 14.0, 15.0]},
            "low": {"sh600000": [9.0, 10.0, 11.0, 12.0, 13.0, 14.0]},
            "close": {"sh600000": [9.5, 10.5, 11.5, 12.5, 13.5, 14.5]},
        }
    )


def floor_panel(panel):
    """先跌到 0 底、再抬起来：收盘就是当天最低价，故 ``VAR3A`` 从 0 起步。"""
    closes = np.concatenate([np.linspace(20.0, 10.0, 16), np.linspace(10.5, 14.0, 5)])
    return panel(
        {
            "high": {"sh600000": closes + 0.2},
            "low": {"sh600000": closes},
            "close": {"sh600000": closes},
        }
    )


# --- 手算常数把公式钉住 ---------------------------------------------------------


def test_brick_line_matches_the_hand_computed_values(panel):
    """公式照字面实现：三层递推平滑 + 减 4 + 与 0 取大，逐位对上手算常数。"""
    got = brick_line(hand_panel(panel)).line["sh600000"]

    assert got.tolist() == pytest.approx(HAND_LINE, nan_ok=True)


def test_the_first_three_bars_are_missing_because_the_window_needs_four(panel):
    """``HHV(H,4)`` 窗口不满 4 根即缺失，故砖型图从第 4 根才有值（ADR-0005 的缺口纪律）。"""
    got = brick_line(hand_panel(panel)).line["sh600000"]

    assert got.iloc[:3].isna().all()
    assert got.iloc[3:].notna().all()


def test_brick_size_is_the_absolute_difference_of_adjacent_bars(panel):
    """砖的大小是相邻两根之差的**绝对值**，故绿砖的大小也恒为正。"""
    got = brick_line(hand_panel(panel)).size["sh600000"]

    assert got.tolist() == pytest.approx(HAND_SIZE, nan_ok=True)


def test_size_ratio_divides_today_by_the_previous_bar(panel):
    """砖的大小之比 = 当根 ÷ 前一根：``(3475/864) ÷ (125/36) = 139/120``。"""
    got = brick_line(hand_panel(panel)).size_ratio["sh600000"]

    assert pd.isna(got.iloc[4])  # 前一根没有砖，故比值无定义
    assert got.iloc[5] == pytest.approx(139 / 120)


# --- 红砖 / 绿砖 --------------------------------------------------------------


def test_a_rise_is_a_red_brick_and_a_fall_is_a_green_brick(panel):
    """手算那一组里第 4 根是红砖（61 → 64.47）、第 5 根是绿砖（64.47 → 60.45）。"""
    line = brick_line(hand_panel(panel)).line

    red = red_brick(line)["sh600000"]
    green = green_brick(line)["sh600000"]

    assert red.tolist() == [False, False, False, False, True, False]
    assert green.tolist() == [False, False, False, False, False, True]


def test_a_flat_pair_is_neither_red_nor_green(panel):
    """两根相等即图上不画砖：既非红也非绿，且大小为 0。

    这条同时钉住绿砖**不是**红砖的补集——把 ``green_brick`` 写成 ``~red_brick`` 会在这里红。
    """
    result = brick_line(flat_panel(panel))
    line = result.line["sh600000"]

    assert line.dropna().tolist() == pytest.approx([161.0, 161.0, 161.0])

    red = red_brick(result.line)["sh600000"]
    green = green_brick(result.line)["sh600000"]

    assert not red.any()
    assert not green.any()
    assert result.size["sh600000"].dropna().tolist() == [0.0, 0.0]


def test_a_green_bricks_size_is_always_positive(panel):
    """跨全体的不变量：**前一根是绿砖**时，它的大小必然为正，故「红砖 ÷ 绿砖」不会除以零。

    这条是分母不会为零的**依据**，不是巧合：绿砖是严格下降，其差值的绝对值必大于 0。
    """
    result = brick_line(fixture_panel(panel))
    green = green_brick(result.line)["sh600000"]

    assert green.any(), "样本里没有绿砖——那这条不变量就没被验到"
    assert (result.size["sh600000"][green] > 0).all()


# --- 0 底截断（ADR-0016） -----------------------------------------------------


def test_a_steady_decline_sits_exactly_on_the_zero_floor(panel):
    """持续下跌时砖型图**恰好是 0**，不是负数——末句那个「与 0 取大」照字面保留。"""
    line = brick_line(floor_panel(panel)).line["sh600000"].dropna()

    assert (line >= 0).all()
    assert (line == 0).any()


def test_a_brick_rising_off_the_floor_is_a_red_brick_as_high_as_its_rise(panel):
    """从 0 底抬起来的那一根是红砖，且**大小就等于抬起的幅度**。

    这是 0 底截断的直接后果，也是 ADR-0016 记下的那个代价：一根从 0 抬到 3 的红砖，与一根
    8 → 11 的红砖，被判为**同高**——因为底部那一根量的是「抬起了多少」而不是「有多高」。
    """
    result = brick_line(floor_panel(panel))
    line = result.line["sh600000"]
    size = result.size["sh600000"]
    red = red_brick(result.line)["sh600000"]

    rising = line.notna() & line.shift(1).eq(0) & (line > 0)

    assert rising.any(), "样本里没有「从 0 底抬起」的那一根——那这条就没被验到"
    assert red[rising].all()
    assert size[rising].to_numpy() == pytest.approx(line[rising].to_numpy())


# --- 缺失的处置（ADR-0005） ---------------------------------------------------


def test_a_window_with_no_range_yields_missing_not_zero(panel):
    """窗口内最高价等于最低价（一字板、长期停牌复牌）时分母为 0，此时**缺失**而不是 0。

    取 0 会让「算不出来」与「真的落在 0 底」两件事混成一件，而那不会报错。
    """
    flat = panel(
        {
            "high": {"sh600000": [10.0] * 6},
            "low": {"sh600000": [10.0] * 6},
            "close": {"sh600000": [10.0] * 6},
        }
    )

    assert brick_line(flat).line["sh600000"].isna().all()


def test_a_gap_does_not_become_the_zero_floor(panel):
    """缺口处砖型图保持**缺失**，不因公式里的 ``IF(..., 0)`` 变成 0。

    行情软件上没有缺口行，照字面它的 ``IF`` 会把缺失判成假、给出 0；本项目按缺口纪律取「缺失」
    ——把停牌日伪造成「落在 0 底的一天」会让它凭空变成一根砖（ADR-0005）。

    缺口之后要先**重新攒满 4 根**窗口才有值，故这里取 8 根：第 4 根是缺口，第 5~7 根是重攒，
    第 8 根起恢复。窗口里有缺口与窗口不满 4 根是**同一件事**——这正是独立实现必须照抄的口径
    （见 :func:`_literal_brick`）。
    """
    nan = float("nan")
    gapped = panel(
        {
            "high": {"sh600000": [10.0, 14.0, 12.0, nan, 13.0, 18.0, 17.0, 16.0]},
            "low": {"sh600000": [8.0, 8.0, 10.0, nan, 11.0, 11.0, 11.0, 11.0]},
            "close": {"sh600000": [9.0, 9.0, 11.0, nan, 12.0, 12.0, 12.0, 12.0]},
        }
    )
    line = brick_line(gapped).line["sh600000"]

    assert pd.isna(line.iloc[3]), "缺口当根必须是缺失，而不是 0 底"
    assert line.iloc[4:7].isna().all(), "缺口之后要重新攒满 4 根窗口，那几根算不出来"
    assert line.iloc[7:].notna().all()
    assert (line.dropna() >= 0).all()


def test_a_zero_denominator_has_no_ratio_rather_than_infinity(panel):
    """前一根没有砖（大小为 0）时比值**无定义**：取缺失，不取无穷，也不取 0。"""
    ratio = brick_line(flat_panel(panel)).size_ratio["sh600000"]

    assert ratio.isna().all()


# --- 形状与类型 ---------------------------------------------------------------


def test_brick_line_keeps_the_symbol_frame_shape_and_float_dtype(panel):
    """三张表都是标的宽表，形状与面板一致（ADR-0009）。"""
    result = brick_line(hand_panel(panel))

    for frame in (result.line, result.size, result.size_ratio):
        assert list(frame.columns) == ["sh600000"]
        assert frame.index.equals(pd.bdate_range("2024-01-02", periods=6))
        assert all(dtype.kind == "f" for dtype in frame.dtypes)


# --- 与独立实现的对照（ADR-0015） ---------------------------------------------


def _literal_sma(values: np.ndarray, n: int, m: int) -> np.ndarray:
    """通达信 ``SMA(X, N, M)`` 的**字面**实现：``(M·X + (N−M)·Y_prev) ÷ N``，逐根推进。"""
    out = np.full(values.size, np.nan)
    previous = np.nan
    for i, value in enumerate(values):
        if np.isnan(value):
            previous = np.nan  # 缺口断开递推，段首以第一个可用值播种
            continue
        previous = value if np.isnan(previous) else (m * value + (n - m) * previous) / n
        out[i] = previous
    return out


def _literal_brick(high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
    """照公式**字面**逐列写的独立实现——只作对照，不作实现。

    写法刻意与库里的实现不同：这里是纯 Python 循环、按列推进，那边是按行向量化。两者算的是
    同一个数学序列，故差异只应来自浮点求值路径（ADR-0015）。

    **窗口的缺口口径必须照抄**：库里的 ``rolling(4, min_periods=4)`` 要求窗口里 4 根**全有值**，
    故这里也要求 4 根全非缺失。若这里图省事用 ``np.nanmax``（它会跳过缺失），两份实现就会在
    「窗口里夹着缺口」的格子上各说各话——那不是舍入差，是口径差，而它只在**样本真的带缺口**时
    才露出来。故下面两份样本里那份随机行情**故意带缺口**。
    """
    h = high.to_numpy(dtype=float)
    low_values = low.to_numpy(dtype=float)
    c = close.to_numpy(dtype=float)
    size = c.size

    hhv = np.full(size, np.nan)
    llv = np.full(size, np.nan)
    for i in range(3, size):
        window_high = h[i - 3 : i + 1]
        window_low = low_values[i - 3 : i + 1]
        if np.isnan(window_high).any() or np.isnan(window_low).any():
            continue  # 窗口里夹着缺口 ⇒ 算不出来，与 min_periods=4 同一口径
        hhv[i] = window_high.max()
        llv[i] = window_low.min()

    span = hhv - llv
    with np.errstate(invalid="ignore", divide="ignore"):
        var1 = (hhv - c) / span * 100.0 - 90.0
        var3 = (c - llv) / span * 100.0
    var1 = np.where(span > 0, var1, np.nan)
    var3 = np.where(span > 0, var3, np.nan)

    var2 = _literal_sma(var1, 4, 1) + 100.0
    var5 = _literal_sma(_literal_sma(var3, 6, 1), 6, 1) + 100.0
    var6 = var5 - var2

    # 照字面：IF(VAR6A>4, VAR6A−4, 0)；缺口单独处置，与库里同样取缺失。
    line = np.where(var6 > 4, var6 - 4, 0.0)
    return pd.Series(np.where(np.isnan(var6), np.nan, line), index=close.index)


#: 随机行情里造缺口的比例：一份「完全无缺口」的样本证明不了缺口口径，而真实行情是有停牌的。
GAP_RATE = 0.02


def random_panel(panel, *, symbols=3, bars=400, seed=20260918, gap_rate=GAP_RATE):
    """多标的随机行情：``high ≥ close ≥ low``，走势各异，**并带缺口**。

    缺口（停牌造成的缺失行）是刻意加的：库里的口径是「窗口里 4 根全有值才算得出来」，而
    ``np.nanmax`` 那类写法会跳过缺失；样本不带缺口时，两种口径的差别**看不出来**。
    """
    rng = np.random.default_rng(seed)
    fields: dict[str, dict[str, np.ndarray]] = {"high": {}, "low": {}, "close": {}}
    for i in range(symbols):
        name = f"sh60{i:04d}"
        walk = 10.0 * np.exp(np.cumsum(rng.normal(0.0, 0.02, bars)))
        high = walk * (1.0 + rng.uniform(0.0, 0.03, bars))
        low = walk * (1.0 - rng.uniform(0.0, 0.03, bars))
        halted = rng.random(bars) < gap_rate
        for column in (walk, high, low):
            column[halted] = np.nan
        fields["close"][name] = walk
        fields["high"][name] = high
        fields["low"][name] = low
    return panel(fields)


def fixture_panel(panel, symbol="sh600000"):
    """真实的日线切片（45 根），当作第二份对照样本。"""
    frame = TdxDataSource("tests/fixtures/tdx").daily(symbol)
    return panel({field: {symbol: frame[field].to_numpy()} for field in ("high", "low", "close")})


def _judgements(line: pd.DataFrame, size: pd.DataFrame):
    """四处判据的布尔读数。

    前三处是**数值层**的比较，第四处是**过滤条件的通过集**（前一根绿砖 + 当根红砖 + 红砖更大）。
    ``红砖是否大于绿砖`` 单列一条是本票据点名的判据，故它不并进过滤条件里——并进去之后，那些
    「前一根不是绿砖」的格子上的翻转就看不见了。
    """
    red = line > line.shift(1)
    green = line < line.shift(1)
    bigger = size > size.shift(1)
    previous_green = line.shift(1) < line.shift(2)
    return red, green, bigger, previous_green & red & bigger


def _literal_readings(high: pd.Series, low: pd.Series, close: pd.Series, name: str):
    """独立实现的 (line, size)，形状与库里一致，故能喂同一个 :func:`_judgements`。"""
    line = _literal_brick(high, low, close).to_frame(name)
    return line, line.diff().abs()


@pytest.mark.parametrize("pool", ["random", "fixture"])
def test_the_vectorised_readings_match_a_literal_per_column_implementation(panel, pool):
    """数值层：两处实现对得上——它只证明「没写错」，不证明「结论没变」（ADR-0015）。

    实测最大相对差记在 :data:`MEASURED_MAX_RELATIVE_DIFFERENCE` 里：这份证据要**报出数字**，
    不能只说「浮点误差可忽略」——那是一句主张。
    """
    sample = random_panel(panel) if pool == "random" else fixture_panel(panel)
    got = brick_line(sample)

    worst = 0.0
    compared = 0
    for name in sample.symbols:
        want = _literal_brick(sample["high"][name], sample["low"][name], sample["close"][name])
        mine = got.line[name]
        both = mine.notna() & want.notna()
        assert both.any(), f"{name}：两份实现没有一格同时有值，对照无从谈起"
        difference = (mine[both] - want[both]).abs()
        scale = want[both].abs().clip(lower=1e-9)
        worst = max(worst, float((difference / scale).max()))
        compared += int(both.sum())

    assert compared >= MIN_COMPARED[pool], f"对拍的有效格只有 {compared} 格，太少了"
    assert worst <= TOLERANCE, f"最大相对差 {worst:g} 超过了浮点求值路径该有的量级"
    # 与记录值同量级：数量级变了说明求值路径变了（换了零件、改了递推写法），那时该更新记录
    # 并重新解释，而不是顺手把容差放宽。
    assert worst <= MEASURED_MAX_RELATIVE_DIFFERENCE[pool] * 100, "与记录在案的量级差了两个数量级"


@pytest.mark.parametrize("pool", ["random", "fixture"])
def test_no_judgement_flips_against_the_literal_implementation(panel, pool):
    """判据层：四处比较**零翻转**——这才是「结论不变」的直接证据（ADR-0015）。

    只有数值差而无翻转统计，不知道会不会在别的数据上翻；故两者都要。
    """
    sample = random_panel(panel) if pool == "random" else fixture_panel(panel)
    got = brick_line(sample)
    mine = _judgements(got.line, got.size)

    labels = ("红砖方向", "绿砖方向", "红砖大于绿砖", "过滤条件")
    fired = False
    for name in sample.symbols:
        literal = _literal_readings(
            sample["high"][name], sample["low"][name], sample["close"][name], name
        )
        wants = _judgements(*literal)
        for label, mine_frame, want_frame in zip(labels, mine, wants, strict=True):
            flips = int((mine_frame[name].to_numpy() != want_frame[name].to_numpy()).sum())
            assert flips == 0, f"{name} 的「{label}」判据翻了 {flips} 格"
            fired = fired or bool(mine_frame[name].any())

    assert fired, "四处判据全假——那零翻转是空的"
