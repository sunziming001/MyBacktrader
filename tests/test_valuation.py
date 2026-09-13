"""估值信号：动态PE、PE 百分位、PEG（票据 #37、#48）。

口径**逐项对齐通达信的市盈率公式**（历史分支），因为行情软件里的 PE/PEG 就是它算的：

    PE历史   = FINVALUE(238) * C / FINVALUE(276)      # 总股本 × 收盘价 ÷ 归母净利TTM
    增历史   = FINVALUE(184)                            # 当期累计同比(%)
    PEG      = IF(PE>0 AND ABS(增)>0.1, PE/增, NULL)
    PE百分位 = (PE − LLV(PE,DUR)) / (HHV(PE,DUR) − LLV(PE,DUR))

本文件的重点是**时点正确性**与**口径的边界**，两者都靠手算断言。另有一条测试把「自创口径
会造成假阳性」写成可核对的事实（对照 `sz002692` 的真实数字）。
"""

from __future__ import annotations

import datetime as dt

import pandas as pd
import pytest

from mbt.data.errors import MarketDataError
from mbt.data.fundamental import FinancialRecord
from mbt.data.valuation import (
    EQUITY,
    MARKET_CAP,
    PE,
    PE_PERCENTILE,
    PEG,
    ROE,
    VALUATION_FIELDS,
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


def record(period, announced, *, shares=1_000_000_000.0, ttm=50_000_000.0, growth=30.0, **extra):
    """造一条 ``FinancialRecord``；估值字段按名义值给出，其余补 0。"""
    values = {
        "eps_ytd": 0.0,
        "net_profit_ytd": 0.0,
        "bvps": 0.0,
        "revenue_ytd": 0.0,
        "revenue_quarter": 0.0,
        "net_profit_quarter": 0.0,
    }
    values.update(
        {
            "total_shares": shares,
            "profit_ttm": ttm,
            "growth_ytd": growth,
            # 默认权益使得 ROE = 50_000_000 ÷ 1_000_000_000 = 5%
            "equity": 1_000_000_000.0,
        }
    )
    values.update(extra)
    return FinancialRecord(
        symbol="sh600000",
        report_period=period,
        announcement_date=announced,
        values=values,
    )


def frame(values, symbol="sh600000", start="2024-01-02"):
    """一列收盘价（由值序列构造）。"""
    return pd.DataFrame(
        {symbol: [float(v) for v in values]},
        index=pd.bdate_range(start, periods=len(values)),
    )


def same_shape(value, symbol="sh600000", start="2024-01-02", periods=3):
    return pd.DataFrame(
        {symbol: [float(value)] * periods},
        index=pd.bdate_range(start, periods=periods),
    )


# --- 动态PE：总股本 × 收盘价 ÷ 归母净利TTM ------------------------------------


def test_pe_is_market_cap_over_ttm_profit():
    """PE = 总股本 × 收盘价 ÷ TTM 净利——与通达信 `FINVALUE(238)*C/FINVALUE(276)` 一致。

    手算：10 亿股 × 20 元 = 200 亿元市值 ÷ 5 亿元 TTM 净利 = **40**。
    """
    close = same_shape(20.0)
    shares = same_shape(1_000_000_000.0)
    ttm = same_shape(500_000_000.0)

    got = price_earnings_ratio(close, shares, ttm)

    assert got.iloc[0, 0] == pytest.approx(40.0)


def test_a_zero_ttm_yields_a_missing_pe_not_an_infinity():
    """TTM 净利为 0 → 缺失。除零给出 ``inf`` 会让「PE > 0」之类的比较得到看似正常的结果。"""
    got = price_earnings_ratio(same_shape(20.0), same_shape(1e9), same_shape(0.0))

    assert pd.isna(got.iloc[0, 0])


def test_a_negative_ttm_yields_a_negative_pe():
    """亏损照常算出负 PE——「PE > 0」是过滤条件的事，不是计算的事。"""
    got = price_earnings_ratio(same_shape(20.0), same_shape(1e9), same_shape(-5e8))

    assert got.iloc[0, 0] == pytest.approx(-40.0)


def test_mismatched_shapes_are_rejected():
    """形状不一致必须报错——逐格相乘/相除时静默错位是本项目最防的那类失败。"""
    with pytest.raises(MarketDataError, match="index 不一致|columns 不一致"):
        price_earnings_ratio(frame([1.0]), same_shape(1.0, symbol="sz000001"), same_shape(1.0))

    with pytest.raises(MarketDataError, match="index 不一致|columns 不一致"):
        price_earnings_ratio(frame([1.0, 2.0]), same_shape(1.0, periods=3), same_shape(1.0))


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
    assert pe_percentile(frame([5.0, 5.0, 5.0]), window=3).iloc[:, 0].isna().all()


def test_the_window_uses_available_history_when_shorter_than_requested():
    """不足 ``window`` 根时按**可得历史**算（通达信 ``LLV``/``HHV`` 语义）。"""
    series = [2.0, 1.0, 3.0, 2.0]

    assert pe_percentile(frame(series), window=1000).equals(pe_percentile(frame(series), window=4))


def test_the_window_includes_the_current_day():
    """含当日：第 2 天 ``[2, 1]`` 里当日是 1，故百分位是 0（窗口最低）而不是 0.5。"""
    assert pe_percentile(frame([2.0, 1.0]), window=2).iloc[1, 0] == pytest.approx(0.0)


def test_a_non_positive_window_is_rejected():
    with pytest.raises(ValueError, match="窗口至少为 1"):
        pe_percentile(frame([1.0]), window=0)


# --- PEG ----------------------------------------------------------------------


def test_peg_divides_pe_by_the_growth_percentage():
    """PE 15、增长 30% → PEG 0.5。增长率是**百分数**（30 而不是 0.3）。"""
    assert price_earnings_growth(same_shape(15.0), same_shape(30.0)).iloc[0, 0] == pytest.approx(
        0.5
    )


@pytest.mark.parametrize("growth", [0.0, 0.05, -0.05, 0.1, -0.1])
def test_a_tiny_growth_yields_a_missing_peg(growth):
    """``|增长率| ≤ 0.1`` → 缺失。

    这是通达信原式的守卫：增长率接近 0 时 PEG 是个巨大的数，那不是「贵」，是**没有意义**。
    边界取**严格大于**（``ABS(增)>0.1``），故恰好 0.1 也算缺失。
    """
    assert pd.isna(price_earnings_growth(same_shape(15.0), same_shape(growth)).iloc[0, 0])


@pytest.mark.parametrize("pe", [0.0, -15.0])
def test_a_non_positive_pe_yields_a_missing_peg(pe):
    """``PE ≤ 0`` → 缺失（通达信同一个守卫）。"""
    assert pd.isna(price_earnings_growth(same_shape(pe), same_shape(30.0)).iloc[0, 0])


def test_a_negative_growth_keeps_its_sign():
    """增长率为负且绝对值够大时 PEG 照常为负——那正是「增长在恶化」的信号。

    实测 `sz002692` 在 2025-09-02 的 PEG 是 **−36.57**（当期同比 −1.67%、PE 61.07），
    而买入条件要求 ``0 < PEG``，故它会被排除。这是**正确的**行为。
    """
    got = price_earnings_growth(same_shape(61.07), same_shape(-1.67))

    assert got.iloc[0, 0] == pytest.approx(61.07 / -1.67)


# --- 时点正确性 ---------------------------------------------------------------


def test_a_quarterly_report_takes_effect_on_its_announcement_date():
    """财报在**公告日**当天生效，而不是报告期当天。

    构造：两期财报。收盘恒 20 元、总股本恒 10 亿股：

    - 2023-03-15 公告那期 TTM = 5 亿 → PE = 200 亿 / 5 亿 = **40**
    - 2023-04-28 公告那期 TTM = 8 亿 → PE = 200 亿 / 8 亿 = **25**
    """
    prices = frame([20.0] * 60, start="2023-03-01")
    financials = FakeFinancials(
        sh600000=[
            record(dt.date(2022, 12, 31), dt.date(2023, 3, 15), ttm=500_000_000.0),
            record(dt.date(2023, 3, 31), dt.date(2023, 4, 28), ttm=800_000_000.0),
        ]
    )

    pe = build_valuation(prices, financials).pe

    assert pd.isna(pe.loc[pd.Timestamp("2023-03-10"), "sh600000"]), "首份财报公告前无从算起"
    assert pe.loc[pd.Timestamp("2023-03-20"), "sh600000"] == pytest.approx(40.0)
    assert pe.loc[pd.Timestamp("2023-04-27"), "sh600000"] == pytest.approx(40.0), "公告前一日未生效"
    assert pe.loc[pd.Timestamp("2023-04-28"), "sh600000"] == pytest.approx(25.0)


def test_an_unusable_record_does_not_enter_the_series():
    """公告日**等于**报告期的记录是占位符（2005 年前的特征），一律不进序列。"""
    prices = frame([20.0] * 10, start="2023-03-01")
    financials = FakeFinancials(
        sh600000=[record(dt.date(2023, 3, 31), dt.date(2023, 3, 31), ttm=500_000_000.0)]
    )

    assert build_valuation(prices, financials).pe.isna().all().all()


def test_a_symbol_without_financials_gets_a_missing_series_not_zeros():
    """没有财务数据的标的整列为缺失——给 0 会让 PE 变成 ``inf`` 或 0，属凭空造数。"""
    valuation = build_valuation(frame([20.0] * 5), FakeFinancials())

    for one in (valuation.pe, valuation.pe_percentile, valuation.peg):
        assert one.isna().all().all()


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
    assert (PE, PE_PERCENTILE, PEG, MARKET_CAP, ROE, EQUITY) == VALUATION_FIELDS


def test_equity_is_the_point_in_time_book_value_and_never_a_substitute_for_roe():
    """``equity`` 原样暴露「归母股东权益」（净资产）。

    它存在的理由是「ROE 拦不住净资产为负」：**净利与净资产同时为负时 ROE 是正的**。
    这条测试把这个反例钉死——若哪天有人把 ``equity`` 当成 ROE 的冗余而删掉，这里会先红。
    """
    financials = FakeFinancials(
        sh600000=[
            record(
                dt.date(2023, 12, 31),
                dt.date(2024, 1, 1),
                ttm=-500_000_000.0,  # 巨亏
                equity=-1_000_000_000.0,  # 资不抵债
            )
        ]
    )

    valuation = build_valuation(frame([20.0] * 3), financials)

    assert valuation.equity.iloc[0, 0] == pytest.approx(-1_000_000_000.0)
    assert valuation.roe.iloc[0, 0] == pytest.approx(50.0), "两负相除得正 ROE——正是要拦的情形"


def test_roe_is_ttm_profit_over_equity_in_percent():
    """``ROE = 归母净利润TTM ÷ 归母股东权益 × 100``，**单位是百分数**。

    手算：TTM 5000 万 ÷ 权益 5 亿 = 10%。
    """
    financials = FakeFinancials(
        sh600000=[
            record(
                dt.date(2023, 12, 31),
                dt.date(2024, 1, 1),
                ttm=50_000_000.0,
                equity=500_000_000.0,
            )
        ]
    )

    valuation = build_valuation(frame([20.0] * 3), financials)

    assert valuation.roe.iloc[0, 0] == pytest.approx(10.0)


def test_the_direct_roe_field_is_not_used_because_it_is_cumulative():
    """**不用 `gpcw` 那个现成的「净资产收益率」字段**——它是累计值，会把一季报的 ROE 报小。

    这里用同一期数据把差异摆出来：现成字段（累计）给 2.5，而 TTM 口径给 10.0。若拿前者做
    「ROE > 10%」的过滤，一季度报告期里几乎所有股票都会被排除——与 PE 那个「年化 ×4」的
    陷阱同源，只是方向相反。
    """
    financials = FakeFinancials(
        sh600000=[
            record(
                dt.date(2023, 3, 31),
                dt.date(2023, 4, 30),
                ttm=50_000_000.0,
                equity=500_000_000.0,
                roe_cum=2.5,  # 一季报的累计 ROE——**本模块刻意不用它**
            )
        ]
    )

    valuation = build_valuation(frame([20.0] * 3, start="2023-05-02"), financials)

    assert valuation.roe.iloc[0, 0] == pytest.approx(10.0), "TTM 口径"


def test_a_zero_equity_yields_a_missing_roe():
    """权益为 0 → 缺失（除零无意义）；负权益照常算出负 ROE。"""
    zero = build_valuation(
        frame([20.0] * 3),
        FakeFinancials(sh600000=[record(dt.date(2023, 12, 31), dt.date(2024, 1, 1), equity=0.0)]),
    )
    negative = build_valuation(
        frame([20.0] * 3),
        FakeFinancials(sh600000=[record(dt.date(2023, 12, 31), dt.date(2024, 1, 1), equity=-5e8)]),
    )

    assert pd.isna(zero.roe.iloc[0, 0])
    assert negative.roe.iloc[0, 0] < 0


def test_market_cap_is_shares_times_price():
    """市值 = 总股本 × 收盘价，**用原始价**（与 PE 同一理由：后复权价会把它放大）。

    手算：10 亿股 × 20 元 = **200 亿元**。
    """
    financials = FakeFinancials(
        sh600000=[record(dt.date(2023, 12, 31), dt.date(2024, 1, 1), shares=1_000_000_000.0)]
    )

    valuation = build_valuation(frame([20.0] * 3), financials)

    assert valuation.market_cap.iloc[0, 0] == pytest.approx(20.0 * 1_000_000_000.0)


def test_a_symbol_without_shares_data_gets_a_missing_market_cap():
    """取不到总股本 → 市值缺失（不给 0，那会让它在「市值 > 100 亿」里被静默排除，
    而「缺失」与「不足」是**两件事**，前者该报出来）。"""
    valuation = build_valuation(frame([20.0] * 3), FakeFinancials())

    assert valuation.market_cap.isna().all().all()


def test_a_symbol_without_financials_gets_a_missing_equity():
    """无可用财报 → 净资产缺失，而不是 0。

    这与「净资产为 0」必须区分：0 是「资不抵债到刚好归零」，缺失是「不知道」。过滤条件
    两者都拦，但把它填成 0 会让下游以为拿到了一个真实的账面值。
    """
    valuation = build_valuation(frame([20.0] * 3), FakeFinancials())

    assert valuation.equity.isna().all().all()


def test_equity_is_point_in_time_like_the_other_financial_fields():
    """公告日之前拿不到净资产——与 PE/ROE 同一道门（ADR-0006）。"""
    financials = FakeFinancials(
        sh600000=[
            record(dt.date(2023, 12, 31), dt.date(2024, 4, 1), equity=7e8),
        ]
    )

    valuation = build_valuation(frame([20.0] * 315, start="2023-01-02"), financials)

    before = valuation.equity.loc[:"2024-03-31", "sh600000"]
    after = valuation.equity.loc["2024-04-01":, "sh600000"]
    assert before.isna().all(), "公告前不可知"
    assert (after == pytest.approx(7e8)).all(), "公告当天起生效"


def test_a_non_datetime_index_is_rejected():
    with pytest.raises(MarketDataError, match="DatetimeIndex"):
        build_valuation(pd.DataFrame({"sh600000": [1.0, 2.0]}), FakeFinancials())


def test_an_unsorted_index_is_rejected():
    """按公告日点取依赖有序索引；无序会让 ``ffill`` 静默取错值。"""
    with pytest.raises(MarketDataError, match="升序"):
        build_valuation(frame([1.0, 2.0, 3.0]).iloc[::-1], FakeFinancials())


# --- 为什么必须照通达信的口径（票据 #48） --------------------------------------


def test_the_hand_rolled_annualisation_would_have_produced_a_false_positive():
    """**本组的核心**：自创的「年化 EPS」口径会让 `sz002692` 变成假阳性买入。

    真实数字（2025-09-02，`sz002692`）：

    - 通达信口径：PE **61.07**、百分位 **91.82%**、PEG **−36.57** → 「百分位 < 8%」不通过
    - 自创口径（年化 EPS + 滞后同比）：PE **50.60**、百分位 **2.55%**、PEG **+0.42** → 通过

    差别全在 PE 的算法上：自创口径用「累计 EPS × 年化系数」，而该股 2021 年半年报净利仅
    870 万元，年化后 PE 冲到 **931**；那个极值进入 1000 日窗口后把分母撑大，于是**之后整整
    4 年**的百分位都被压低。

    这里用简化数据把那条机制复现出来：同一段 PE 序列，其中一个「被年化放大的极值」会把
    后续分位压低。
    """
    # 末值 55、窗口最低 50；被放大的极值把窗口上界抬到 931，于是分位被压到 0.6%；
    # 换成真实的次高值 100 时是 10%——**跨越了 8% 这条买入门槛**。
    pe_with_spike = frame([931.25, 50.0, 60.0, 55.0])
    pe_without = frame([100.0, 50.0, 60.0, 55.0])

    with_spike = pe_percentile(pe_with_spike, window=4)
    without = pe_percentile(pe_without, window=4)

    last = pd.Timestamp(pe_with_spike.index[-1])
    assert with_spike.loc[last, "sh600000"] < 0.08, "被放大的极值让「< 8%」凭空通过"
    assert without.loc[last, "sh600000"] > 0.08, "换成真实的次高值就通不过"
    assert with_spike.loc[last, "sh600000"] < without.loc[last, "sh600000"]
