"""估值信号：动态PE、PE 百分位、PEG（票据 #37）。

本文件的重心是**时点正确性**与**口径的边界**，两者都靠手算断言：

- 一季报在**公告日**当天才生效，而不是报告期（3 月 31 日）当天——差额近一个月，而那段
  时间里用它就是把未来的信息搬进历史（ADR-0006）；
- 百分位在窗口恒定时**返回缺失**，不是 0 或 1；
- 同比的分母取绝对值，否则「由亏转盈」的符号会反过来。
"""

from __future__ import annotations

import datetime as dt

import pandas as pd
import pytest

from mbt.data.errors import MarketDataError
from mbt.data.fundamental import FinancialRecord
from mbt.data.valuation import (
    PE,
    PE_PERCENTILE,
    PEG,
    VALUATION_FIELDS,
    annualisation_factor,
    build_valuation,
    pe_percentile,
    price_earnings_growth,
    price_earnings_ratio,
)


class FakeFinancials:
    """只提供 ``records(symbol)`` 的最小替身——本模块唯一依赖的那一个方法。"""

    def __init__(self, **by_symbol):
        self._by_symbol = {symbol: tuple(records) for symbol, records in by_symbol.items()}

    def records(self, symbol):
        return tuple(sorted(self._by_symbol.get(symbol, ()), key=lambda r: r.report_period))


def record(symbol, period, announced, **values):
    """造一条 ``FinancialRecord``；未给的字段补 0。"""
    filled = {
        "eps_ytd": 0.0,
        "net_profit_ytd": 0.0,
        "bvps": 0.0,
        "revenue_ytd": 0.0,
        "revenue_quarter": 0.0,
        "net_profit_quarter": 0.0,
    }
    filled.update(values)
    return FinancialRecord(
        symbol=symbol,
        report_period=period,
        announcement_date=announced,
        values=filled,
    )


def frame(values, symbol="sh600000", start="2023-03-01"):
    """一列收盘价（由值序列构造）。"""
    return pd.DataFrame(
        {symbol: [float(v) for v in values]},
        index=pd.bdate_range(start, periods=len(values)),
    )


# --- 年化系数 -----------------------------------------------------------------


@pytest.mark.parametrize(
    ("period", "expected"),
    [
        (dt.date(2024, 3, 31), 4.0),
        (dt.date(2024, 6, 30), 2.0),
        (dt.date(2024, 9, 30), 4.0 / 3.0),
        (dt.date(2024, 12, 31), 1.0),
    ],
)
def test_the_annualisation_factor_follows_the_report_period(period, expected):
    """通达信口径：Q1×4、H1×2、Q3×4/3、年报×1。"""
    assert annualisation_factor(period) == pytest.approx(expected)


def test_a_period_that_is_not_a_quarter_end_is_rejected():
    """不是季末就**报错**，不猜系数——猜错会让整个 PE 序列偏一个倍数，且不报错。"""
    with pytest.raises(MarketDataError, match="不是季末"):
        annualisation_factor(dt.date(2024, 5, 31))


# --- 动态PE -------------------------------------------------------------------


def test_pe_divides_price_by_annualised_eps():
    close = frame([100.0])
    earnings = frame([4.0])

    assert price_earnings_ratio(close, earnings).iloc[0, 0] == pytest.approx(25.0)


def test_a_zero_eps_yields_a_missing_pe_not_an_infinity():
    """EPS 为 0 → 缺失。除零给出 ``inf`` 会让「PE > 0」之类的比较得到看似正常的结果。"""
    close = frame([100.0])
    earnings = frame([0.0])

    assert pd.isna(price_earnings_ratio(close, earnings).iloc[0, 0])


def test_a_negative_eps_yields_a_negative_pe():
    """负 EPS 照常算出负 PE——「PE > 0」是过滤条件的事，不是计算的事。

    多一个隐含分支就多一处会漂移的地方：若这里静默返回缺失，过滤条件与计算之间就有了两套
    「亏损」的表示。
    """
    close = frame([100.0])
    earnings = frame([-4.0])

    assert price_earnings_ratio(close, earnings).iloc[0, 0] == pytest.approx(-25.0)


def test_mismatched_shapes_are_rejected():
    with pytest.raises(MarketDataError, match="columns 不一致"):
        price_earnings_ratio(frame([1.0]), frame([1.0], symbol="sz000001"))


# --- 百分位 -------------------------------------------------------------------


def test_the_percentile_is_hand_computable():
    """``[2, 1, 3, 2]``、窗口 4：逐日手算 ``(PE − LLV) / (HHV − LLV)``。

    第 1 天只有 1 个点（窗口恒定）→ 缺失；其后 ``(1−1)/(2−1)=0``、``(3−1)/(3−1)=1``、
    ``(2−1)/(3−1)=0.5``。
    """
    got = pe_percentile(frame([2.0, 1.0, 3.0, 2.0]), window=4)

    values = got.iloc[:, 0].tolist()
    assert pd.isna(values[0])
    assert values[1:] == pytest.approx([0.0, 1.0, 0.5])


def test_a_constant_window_yields_missing_not_zero_or_one():
    """窗口内 PE 恒定 → ``HHV == LLV`` → **缺失**。

    硬给 0 或 1 会让「百分位 < 8%」凭空通过或凭空拦下——两种都是无依据的答案。
    """
    got = pe_percentile(frame([5.0, 5.0, 5.0]), window=3)

    assert got.iloc[:, 0].isna().all()


def test_the_window_uses_available_history_when_shorter_than_requested():
    """不足 ``window`` 根时按**可得历史**算（通达信 ``LLV``/``HHV`` 语义）。

    故 4 根数据配 1000 的窗口，与配 4 的窗口结果相同——次新股的「4 年百分位」实际是短窗口
    上的百分位，这是刻意保留的口径（见模块文档）。
    """
    series = [2.0, 1.0, 3.0, 2.0]

    assert pe_percentile(frame(series), window=1000).equals(pe_percentile(frame(series), window=4))


def test_the_window_includes_the_current_day():
    """含当日：第 2 天 ``[2, 1]`` 里当日是 1，故百分位是 0（窗口最低）而不是 0.5。"""
    got = pe_percentile(frame([2.0, 1.0]), window=2)

    assert got.iloc[1, 0] == pytest.approx(0.0)


def test_a_non_positive_window_is_rejected():
    with pytest.raises(ValueError, match="窗口至少为 1"):
        pe_percentile(frame([1.0]), window=0)


# --- PEG ----------------------------------------------------------------------


def test_peg_divides_pe_by_the_growth_percentage():
    """PE 15、增长 30% → PEG 0.5。增长率是**百分数**（30 而不是 0.3）。"""
    pe = frame([15.0])
    growth = frame([30.0])

    assert price_earnings_growth(pe, growth).iloc[0, 0] == pytest.approx(0.5)


def test_a_zero_growth_yields_a_missing_peg():
    assert pd.isna(price_earnings_growth(frame([15.0]), frame([0.0])).iloc[0, 0])


# --- 时点正确性（本组最重要） -------------------------------------------------


def test_a_quarterly_report_takes_effect_on_its_announcement_date_not_the_period_end():
    """一季报在**公告日**当天生效，而不是报告期（3 月 31 日）。

    构造：年报（报告期 2022-12-31、公告 2023-03-15、累计 EPS 2.0 → 年化 2.0）与一季报
    （报告期 2023-03-31、**公告 2023-04-28**、累计 EPS 1.2 → 年化 4.8）。收盘价恒 100：

    - 2023-03-20 与 2023-04-01：只知年报 → PE = 100 / 2.0 = **50**
    - 2023-04-28 之后：一季报生效 → PE = 100 / 4.8 ≈ **20.83**

    若按报告期对齐，4 月 1 日就会用上一季报——比真实公告早了近一个月，而那段「提前知道」
    正是 ADR-0006 要挡的。
    """
    prices = frame([100.0] * 60, start="2023-03-01")
    financials = FakeFinancials(
        sh600000=[
            record(
                "sh600000",
                dt.date(2022, 12, 31),
                dt.date(2023, 3, 15),
                eps_ytd=2.0,
            ),
            record(
                "sh600000",
                dt.date(2023, 3, 31),
                dt.date(2023, 4, 28),
                eps_ytd=1.2,
            ),
        ]
    )

    pe = build_valuation(prices, financials).pe

    assert pd.isna(pe.loc[pd.Timestamp("2023-03-10"), "sh600000"]), "首份财报公告前无从算起"
    assert pe.loc[pd.Timestamp("2023-03-20"), "sh600000"] == pytest.approx(50.0)
    assert pe.loc[pd.Timestamp("2023-04-27"), "sh600000"] == pytest.approx(
        50.0
    ), "公告前一日仍是年报"
    assert pe.loc[pd.Timestamp("2023-04-28"), "sh600000"] == pytest.approx(100.0 / 4.8)


def test_an_unusable_record_does_not_enter_the_series():
    """公告日**等于**报告期的记录是占位符（2005 年前的特征），一律不进序列。

    若不挡它，那条记录会带着一个假公告日进入点取，让历史某段用上「当时还不知道」的财报。
    """
    prices = frame([100.0] * 10, start="2023-03-01")
    financials = FakeFinancials(
        sh600000=[
            record("sh600000", dt.date(2023, 3, 31), dt.date(2023, 3, 31), eps_ytd=1.0),
        ]
    )

    assert build_valuation(prices, financials).pe.isna().all().all()


def test_a_symbol_without_financials_gets_a_missing_series_not_zeros():
    """没有财务数据的标的整列为缺失——给 0 会让 PE 变成 ``inf`` 或 0，属凭空造数。"""
    prices = frame([100.0] * 5)

    valuation = build_valuation(prices, FakeFinancials())

    assert valuation.pe.isna().all().all()
    assert valuation.pe_percentile.isna().all().all()
    assert valuation.peg.isna().all().all()


# --- 同比增长率 ---------------------------------------------------------------


def test_the_growth_is_year_on_year_of_cumulative_net_profit():
    """累计同比：2022 年 100 → 2023 年 130，增长 30%。PEG 由 PE 15 得 0.5。"""
    prices = frame([15.0] * 20, start="2023-03-01")
    financials = FakeFinancials(
        sh600000=[
            record(
                "sh600000",
                dt.date(2021, 12, 31),
                dt.date(2022, 3, 15),
                net_profit_ytd=100.0,
                eps_ytd=1.0,
            ),
            record(
                "sh600000",
                dt.date(2022, 12, 31),
                dt.date(2023, 3, 15),
                net_profit_ytd=130.0,
                eps_ytd=1.0,
            ),
        ]
    )

    valuation = build_valuation(prices, financials)

    assert valuation.peg.loc[pd.Timestamp("2023-03-20"), "sh600000"] == pytest.approx(0.5)


def test_a_loss_turning_into_a_profit_gets_a_positive_growth():
    """**分母取绝对值**的意义：-100 → +50 是「增长 150%」，不是「下降 150%」。

    不带绝对值时 ``(50 − (−100)) / (−100) = −150%``——符号反了，于是 PEG 变负，而「0 < PEG」
    这条买入条件会因此把一家刚扭亏的公司挡在门外。
    """
    prices = frame([15.0] * 20, start="2023-03-01")
    financials = FakeFinancials(
        sh600000=[
            record(
                "sh600000",
                dt.date(2021, 12, 31),
                dt.date(2022, 3, 15),
                net_profit_ytd=-100.0,
                eps_ytd=1.0,
            ),
            record(
                "sh600000",
                dt.date(2022, 12, 31),
                dt.date(2023, 3, 15),
                net_profit_ytd=50.0,
                eps_ytd=1.0,
            ),
        ]
    )

    valuation = build_valuation(prices, financials)

    # 增长率 150% 时 PE=15 → PEG = 15 / 150 = 0.1（正数）
    assert valuation.peg.loc[pd.Timestamp("2023-03-20"), "sh600000"] == pytest.approx(0.1)


def test_a_missing_prior_year_record_yields_a_missing_peg():
    """去年同期的记录缺失（或不可用）→ 同比无从算起 → PEG 缺失，而不是拿别的期间凑。"""
    prices = frame([15.0] * 20, start="2023-03-01")
    financials = FakeFinancials(
        sh600000=[
            record(
                "sh600000",
                dt.date(2022, 12, 31),
                dt.date(2023, 3, 15),
                net_profit_ytd=130.0,
                eps_ytd=1.3,
            ),
        ]
    )

    valuation = build_valuation(prices, financials)

    assert valuation.pe.loc[pd.Timestamp("2023-03-20"), "sh600000"] == pytest.approx(15.0 / 1.3)
    assert pd.isna(valuation.peg.loc[pd.Timestamp("2023-03-20"), "sh600000"])


def test_a_zero_base_yields_a_missing_peg():
    """基数为 0 → 同比无定义（不是无穷大）。"""
    prices = frame([15.0] * 20, start="2023-03-01")
    financials = FakeFinancials(
        sh600000=[
            record(
                "sh600000",
                dt.date(2021, 12, 31),
                dt.date(2022, 3, 15),
                net_profit_ytd=0.0,
                eps_ytd=1.0,
            ),
            record(
                "sh600000",
                dt.date(2022, 12, 31),
                dt.date(2023, 3, 15),
                net_profit_ytd=130.0,
                eps_ytd=1.0,
            ),
        ]
    )

    assert pd.isna(build_valuation(prices, financials).peg.loc["2023-03-20", "sh600000"])


# --- 契约与形状 ---------------------------------------------------------------


def test_the_three_series_share_the_index_and_columns_of_the_prices():
    prices = frame([10.0] * 6)

    valuation = build_valuation(prices, FakeFinancials())

    for one in (valuation.pe, valuation.pe_percentile, valuation.peg):
        assert one.index.equals(prices.index)
        assert one.columns.equals(prices.columns)


def test_as_fields_exposes_the_documented_names():
    valuation = build_valuation(frame([10.0] * 3), FakeFinancials())

    assert tuple(valuation.as_fields()) == VALUATION_FIELDS
    assert (PE, PE_PERCENTILE, PEG) == VALUATION_FIELDS


def test_a_non_datetime_index_is_rejected():
    prices = pd.DataFrame({"sh600000": [1.0, 2.0]})

    with pytest.raises(MarketDataError, match="DatetimeIndex"):
        build_valuation(prices, FakeFinancials())


def test_an_unsorted_index_is_rejected():
    """按公告日点取依赖有序索引；无序会让 ``ffill`` 静默取错值。"""
    prices = frame([1.0, 2.0, 3.0]).iloc[::-1]

    with pytest.raises(MarketDataError, match="升序"):
        build_valuation(prices, FakeFinancials())
