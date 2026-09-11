"""验证复权：标准除权除息公式、按评估日过滤、两种视图的锚点性质（票据 #3 切片 3–4）。

数值断言分两类：合成序列用于**手工验算**（数字可当场心算），真实 fixture 用于验证
「除权日假跳空确实被消除」这一端到端性质。
"""

import datetime as dt

import pandas as pd
import pytest

from mbt.data import (
    AdjustmentEvent,
    GbbqDataSource,
    MarketDataError,
    TdxDataSource,
    adjustment_factors,
    backward_adjusted,
    forward_adjusted,
)

SH = "sh600000"

#: 真实 fixture 里落在价格窗口内的唯一除权除息事件：2026-07-16 每 10 股派 4.20 元。
EX_DATE = pd.Timestamp("2026-07-16")
PREV_BAR = pd.Timestamp("2026-07-15")


def _frame(closes, start="2024-01-02"):
    """只含收盘价的极简宽表；adjust 不涉及撮合，故无需真实振幅。"""
    index = pd.bdate_range(start, periods=len(closes))
    return pd.DataFrame(
        {
            "open": list(closes),
            "high": list(closes),
            "low": list(closes),
            "close": list(closes),
            "volume": [1000] * len(closes),
            "amount": [1_000.0] * len(closes),
        },
        index=index,
    )


@pytest.fixture
def sh_prices(fixture_root):
    return TdxDataSource(fixture_root).daily(SH)


@pytest.fixture
def sh_events(gbbq_file):
    return GbbqDataSource(gbbq_file).events(SH)


# --- 除权除息公式：手工验算 ---


def test_reference_price_hand_verified_for_cash_dividend():
    """10 派 5：前收 10.00 → 参考价 9.50，除权因子 0.95。"""
    event = AdjustmentEvent(SH, dt.date(2024, 6, 1), cash_per_10=5.0)

    assert event.reference_price(10.0) == pytest.approx(9.5)
    assert event.factor(10.0) == pytest.approx(0.95)


def test_reference_price_hand_verified_for_bonus_shares():
    """10 送转 10：股本翻倍，前收 10.00 → 参考价 5.00，因子 0.5。"""
    event = AdjustmentEvent(SH, dt.date(2024, 6, 1), bonus_per_10=10.0)

    assert event.reference_price(10.0) == pytest.approx(5.0)
    assert event.factor(10.0) == pytest.approx(0.5)


def test_reference_price_hand_verified_for_rights_issue():
    """10 配 3、配股价 5.00，前收 10.00：

    ref = (10 − 0 + 5 × 0.3) / (1 + 0 + 0.3) = 11.5 / 1.3 = 8.846153846…
    """
    event = AdjustmentEvent(SH, dt.date(2024, 6, 1), rights_price=5.0, rights_per_10=3.0)

    assert event.reference_price(10.0) == pytest.approx(11.5 / 1.3)
    assert event.factor(10.0) == pytest.approx(11.5 / 1.3 / 10.0)


def test_reference_price_hand_verified_when_dividend_bonus_and_rights_combine():
    """10 派 2 送 5 配 2、配股价 4.00，前收 10.00：

    ref = (10 − 0.2 + 4 × 0.2) / (1 + 0.5 + 0.2) = 10.6 / 1.7 = 6.235294117…
    三项同时出现才检验得出分母是不是「1 + 送转 + 配股」。
    """
    event = AdjustmentEvent(
        SH,
        dt.date(2024, 6, 1),
        cash_per_10=2.0,
        rights_price=4.0,
        bonus_per_10=5.0,
        rights_per_10=2.0,
    )

    assert event.reference_price(10.0) == pytest.approx(10.6 / 1.7)
    assert event.factor(10.0) == pytest.approx(0.6235294117647059)


def test_a_zero_event_does_not_move_the_price():
    event = AdjustmentEvent(SH, dt.date(2024, 6, 1))

    assert event.factor(10.0) == 1.0


def test_factor_rejects_a_non_positive_previous_close():
    event = AdjustmentEvent(SH, dt.date(2024, 6, 1), cash_per_10=5.0)

    with pytest.raises(MarketDataError, match="前收盘价"):
        event.factor(0.0)


def test_factor_rejects_an_event_that_would_produce_a_negative_price():
    """分红高于股价加配股缴款时参考价为负——事件数据不成立，不得算出一个负价格。"""
    event = AdjustmentEvent(SH, dt.date(2024, 6, 1), cash_per_10=200.0)

    with pytest.raises(MarketDataError, match="除权因子"):
        event.factor(10.0)


# --- 因子：常规行为 ---


def test_no_events_means_no_adjustment():
    prices = _frame([10.0, 11.0, 12.0])

    assert (adjustment_factors(prices, []) == 1.0).all()
    pd.testing.assert_frame_equal(backward_adjusted(prices, []), prices)
    pd.testing.assert_frame_equal(forward_adjusted(prices, []), prices)


def test_events_compound_multiplicatively():
    """两个事件各自的 ``1/r`` 相乘——因子是累积乘积，不是最后一个事件的因子。"""
    prices = _frame([10.0] * 4)  # 01-02, 01-03, 01-04, 01-05
    first = AdjustmentEvent(SH, dt.date(2024, 1, 3), cash_per_10=5.0)  # r = 0.95
    second = AdjustmentEvent(SH, dt.date(2024, 1, 5), cash_per_10=10.0)  # r = 0.90

    factors = adjustment_factors(prices, [first, second])

    assert factors.iloc[0] == 1.0
    assert factors.iloc[1] == pytest.approx(1 / 0.95)
    assert factors.iloc[2] == pytest.approx(1 / 0.95)
    assert factors.iloc[3] == pytest.approx(1 / (0.95 * 0.90))


def test_events_before_the_series_are_not_applied():
    """序列之前的事件算不出 r（缺前收盘价），且只贡献一个常数倍数，故略去。"""
    prices = _frame([10.0, 11.0, 12.0])
    ancient = AdjustmentEvent(SH, dt.date(2020, 1, 1), cash_per_10=5.0)

    assert (adjustment_factors(prices, [ancient]) == 1.0).all()


def test_events_after_the_last_bar_are_not_applied():
    """落不到任何 K 线上的事件进不了因子。"""
    prices = _frame([10.0, 11.0, 12.0])
    future = AdjustmentEvent(SH, dt.date(2024, 2, 1), cash_per_10=5.0)

    assert (adjustment_factors(prices, [future]) == 1.0).all()


# --- 评估日过滤（ADR-0006） ---


def test_as_of_excludes_events_that_are_inside_the_series_but_not_yet_effective():
    """评估日之后的事件不得参与——即使它就落在价格窗口内。

    这正是 ADR-0006 的场景：用一段到 01-08 的行情回溯评估 01-04，01-05 才除权的事件
    在当时尚不可知，把它算进去就是把未来信息注入过去。
    """
    prices = _frame([10.0] * 5)  # 01-02 … 01-08
    event = AdjustmentEvent(SH, dt.date(2024, 1, 5), cash_per_10=5.0)

    as_of = dt.date(2024, 1, 4)
    assert (adjustment_factors(prices, [event], as_of=as_of) == 1.0).all()

    # 不提评估日时它才会生效——证明上一条不是因别的原因恒为 1
    assert not (adjustment_factors(prices, [event]).iloc[3:] == 1.0).all()


def test_a_future_event_does_not_rescale_the_forward_view():
    """「已公告、未除权」的未来事件不得改变前复权视图。

    前复权以序列末根为锚，故未落到任何 K 线上的事件既进不了因子也改不了锚点。
    若把锚点写成「最新一次除权」，这里就会提前用上未来的事件——本测试即钉住该错误。
    """
    prices = _frame([10.0, 11.0, 12.0])
    announced = AdjustmentEvent(SH, dt.date(2024, 1, 20), cash_per_10=5.0)

    pd.testing.assert_frame_equal(
        forward_adjusted(prices, [announced]), forward_adjusted(prices, [])
    )


# --- 两种视图的锚点性质（真实 fixture） ---


def test_backward_view_keeps_the_first_bar_at_its_raw_price(sh_prices, sh_events):
    adjusted = backward_adjusted(sh_prices, sh_events)

    assert adjusted["close"].iloc[0] == sh_prices["close"].iloc[0]


def test_forward_view_keeps_the_latest_price_at_its_raw_price(sh_prices, sh_events):
    """验收标准之一：前复权序列**最新价不变**。"""
    adjusted = forward_adjusted(sh_prices, sh_events)

    assert adjusted["close"].iloc[-1] == sh_prices["close"].iloc[-1]
    assert adjusted["high"].iloc[-1] == sh_prices["high"].iloc[-1]


def test_backward_view_removes_the_fake_gap_on_the_ex_date(sh_prices, sh_events):
    """除权日的「跳空」大部分是除权本身造成的假象。

    原始价在 2026-07-16 跌 4.94%，而当天只是每 10 股派 4.20 元——后复权后应只剩真实
    波动（不足 1%）。这正是本票要消除的假跳空。
    """
    adjusted = backward_adjusted(sh_prices, sh_events)

    raw_gap = sh_prices["close"].loc[EX_DATE] / sh_prices["close"].loc[PREV_BAR] - 1
    hfq_gap = adjusted["close"].loc[EX_DATE] / adjusted["close"].loc[PREV_BAR] - 1

    assert raw_gap == pytest.approx(-0.0494, abs=5e-4), "原始价带假跳空"
    assert abs(hfq_gap) < 0.01, f"后复权后不应有 {hfq_gap:.2%} 的跳空"


def test_backward_view_is_continuous_by_construction(sh_prices, sh_events):
    """后复权在除权日的收益 = 原始收益 ÷ 除权因子 —— 即跳空被恰好抵消。"""
    adjusted = backward_adjusted(sh_prices, sh_events)
    event = next(e for e in sh_events if pd.Timestamp(e.ex_date) == EX_DATE)
    factor = event.factor(sh_prices["close"].loc[PREV_BAR])

    raw_ratio = sh_prices["close"].loc[EX_DATE] / sh_prices["close"].loc[PREV_BAR]
    hfq_ratio = adjusted["close"].loc[EX_DATE] / adjusted["close"].loc[PREV_BAR]

    assert hfq_ratio == pytest.approx(raw_ratio / factor)


def test_forward_view_on_the_day_before_the_ex_date_equals_the_reference_price(
    sh_prices, sh_events
):
    """除权前一日的**前复权价恰好等于除权除息参考价**。

    这不是巧合而是恒等式：``qfq(前一日) = 前收盘 × r = 参考价``。实盘公布的参考价为
    8.890（见 ``docs/research/tdx-halt-and-limit-representation.md``），故这条同时是
    在真实数据上对公式的手工验算。
    """
    adjusted = forward_adjusted(sh_prices, sh_events)

    assert adjusted["close"].loc[PREV_BAR] == pytest.approx(8.89, rel=1e-7)


def test_forward_view_is_the_backward_view_rescaled(sh_prices, sh_events):
    """两视图只差一个常数倍数——后复权的绝对尺度任意，前复权把它钉在最新价上。"""
    hfq = backward_adjusted(sh_prices, sh_events)
    qfq = forward_adjusted(sh_prices, sh_events)

    scale = sh_prices["close"].iloc[-1] / hfq["close"].iloc[-1]

    assert qfq["close"].iloc[-2] == pytest.approx(hfq["close"].iloc[-2] * scale)
    assert qfq["close"].iloc[0] == pytest.approx(hfq["close"].iloc[0] * scale)


def test_only_price_columns_are_adjusted(sh_prices, sh_events):
    """成交量与成交额不是价格口径，复权不得改动它们。"""
    adjusted = backward_adjusted(sh_prices, sh_events)

    pd.testing.assert_series_equal(adjusted["volume"], sh_prices["volume"])
    pd.testing.assert_series_equal(adjusted["amount"], sh_prices["amount"])


def test_the_input_frame_is_not_mutated(sh_prices, sh_events):
    before = sh_prices.copy()

    backward_adjusted(sh_prices, sh_events)
    forward_adjusted(sh_prices, sh_events)

    pd.testing.assert_frame_equal(sh_prices, before)


def test_an_empty_frame_is_returned_unchanged():
    empty = _frame([])

    assert len(backward_adjusted(empty, [])) == 0


# --- 入参校验 ---


def test_factors_require_a_close_column():
    prices = _frame([10.0, 11.0]).drop(columns=["close", "open", "high", "low"])

    with pytest.raises(ValueError, match="close"):
        adjustment_factors(prices, [])


def test_factors_require_a_sorted_index():
    prices = _frame([10.0, 11.0, 12.0])[::-1]

    with pytest.raises(ValueError, match="升序"):
        adjustment_factors(prices, [])


def test_as_of_must_be_a_date():
    prices = _frame([10.0, 11.0])

    with pytest.raises(TypeError, match="as_of"):
        adjustment_factors(prices, [], as_of="2024-01-03")
