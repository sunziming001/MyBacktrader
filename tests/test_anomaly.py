"""异常检测：把越出涨跌停带的跳空区分为公司行为与坏数据（ADR-0005）。"""

from __future__ import annotations

import datetime as dt

import pandas as pd
import pytest

from mbt.data import AdjustmentEvent, MarketDataError, find_anomalies, require_no_anomalies
from mbt.rules import RuleTable, limit_band, limit_price, round_to_cent

#: 出厂规则表窗口内的普通标的：沪主板 10%，非 ST。
SYMBOL = "sh600000"
LIMIT = 0.10


@pytest.fixture(scope="module")
def rules(tmp_path_factory):
    """只含「沪主板 10%」的自足规则表，避免测试依赖出厂数值。"""
    path = tmp_path_factory.mktemp("rules") / "plain.toml"
    path.write_text(
        """
schema_version = 1

[[price_limit]]
board = "沪主板"
effective_from = 2015-01-01
limit = 0.10
""",
        encoding="utf-8",
    )
    return RuleTable.load(path)


def bars(rows: dict[str, tuple[float, float, float, float]]) -> pd.DataFrame:
    """由 ``{日期: (开, 高, 低, 收)}`` 造一张价格表。"""
    index = pd.DatetimeIndex([pd.Timestamp(day) for day in rows])
    frame = pd.DataFrame.from_dict(
        {
            day: dict(zip(("open", "high", "low", "close"), values, strict=True))
            for day, values in rows.items()
        },
        orient="index",
    )
    frame.index = index
    frame.index.name = "date"
    return frame.sort_index()


def dividend(ex_date: str, each_10: float, symbol: str = SYMBOL) -> AdjustmentEvent:
    """每 10 股派 ``each_10`` 元现金。"""
    return AdjustmentEvent(
        symbol=symbol, ex_date=dt.date.fromisoformat(ex_date), cash_per_10=each_10
    )


def bonus(ex_date: str, each_10: float, symbol: str = SYMBOL) -> AdjustmentEvent:
    """每 10 股送转 ``each_10`` 股。"""
    return AdjustmentEvent(
        symbol=symbol, ex_date=dt.date.fromisoformat(ex_date), bonus_per_10=each_10
    )


# --- 判据：比限价带，不比涨跌幅比率 ---


def test_bar_inside_the_band_is_clean(rules):
    prices = bars(
        {
            "2024-01-02": (10.00, 10.50, 9.50, 10.00),
            "2024-01-03": (10.10, 10.90, 9.10, 10.50),
        }
    )
    assert find_anomalies(prices, SYMBOL, rules) == []


def test_bar_exactly_at_the_band_edges_is_clean(rules):
    """涨停价 / 跌停价本身是**合法**成交价，不能因「正好触限」被判异常。"""
    prices = bars(
        {
            "2024-01-02": (10.00, 10.00, 10.00, 10.00),
            "2024-01-03": (11.00, 11.00, 9.00, 9.00),
        }
    )
    assert find_anomalies(prices, SYMBOL, rules) == []


def test_a_ratio_over_the_limit_but_inside_the_band_is_not_an_anomaly(rules):
    """11.26 → 12.39 是 +10.04%，但 12.39 正是涨停价——比比率会把它误报。

    实测本机数据里这类「比率略超名义限幅」的合法交易日有数百个，是本模块存在的理由。
    """
    assert round_to_cent(11.26 * 1.1) == 12.39
    prices = bars(
        {
            "2024-01-02": (11.26, 11.26, 11.26, 11.26),
            "2024-01-03": (12.30, 12.39, 12.20, 12.39),
        }
    )
    assert find_anomalies(prices, SYMBOL, rules) == []


def test_intraday_extremes_are_checked_not_only_the_close(rules):
    """收盘在带内、但盘中最高越界——比收盘价会漏掉它。"""
    prices = bars(
        {
            "2024-01-02": (10.00, 10.00, 10.00, 10.00),
            "2024-01-03": (10.50, 11.50, 10.40, 10.60),
        }
    )
    anomalies = find_anomalies(prices, SYMBOL, rules)
    assert len(anomalies) == 1
    assert anomalies[0].high == 11.50
    assert anomalies[0].upper == 11.00


# --- 归因：公司行为解释跳空 ---


def test_gap_without_any_event_is_unexplained(rules):
    prices = bars(
        {
            "2024-01-02": (10.00, 10.00, 10.00, 10.00),
            "2024-01-03": (8.50, 8.60, 8.40, 8.50),
        }
    )
    (anomaly,) = find_anomalies(prices, SYMBOL, rules)
    assert anomaly.kind == "unexplained"
    assert anomaly.events == ()
    assert anomaly.lower == 9.00
    assert "无权息事件" in anomaly.describe()


def test_ex_date_gap_is_explained_by_the_dividend(rules):
    """10 派 5 元 → 参考价 9.50，带 [8.55, 10.45]；从 10.00 跌到 8.60 落在带内。"""
    prices = bars(
        {
            "2024-01-02": (10.00, 10.00, 10.00, 10.00),
            "2024-01-03": (8.70, 10.40, 8.60, 9.00),
        }
    )
    assert find_anomalies(prices, SYMBOL, rules, [dividend("2024-01-03", 5.0)]) == []


def test_gap_beyond_what_the_event_explains_is_still_an_anomaly(rules):
    """有事件不等于豁免：按其重算的带仍被越出，就是坏数据。"""
    prices = bars(
        {
            "2024-01-02": (10.00, 10.00, 10.00, 10.00),
            "2024-01-03": (7.00, 7.10, 6.90, 7.00),
        }
    )
    (anomaly,) = find_anomalies(prices, SYMBOL, rules, [dividend("2024-01-03", 5.0)])
    assert anomaly.kind == "beyond_event"
    assert anomaly.lower == 8.55
    assert "事件" in anomaly.describe()


def test_event_during_a_suspension_is_attributed_across_the_gap(rules):
    """停牌期间的权息事件在复牌日才体现为跳空——只查当根 K 线会误判为坏数据。"""
    prices = bars(
        {
            "2024-01-02": (10.00, 10.00, 10.00, 10.00),
            # 中间停牌两周，期间 10 送 10 → 参考价 5.00，带 [4.50, 5.50]
            "2024-01-16": (5.20, 5.45, 5.10, 5.30),
        }
    )
    events = [bonus("2024-01-10", 10.0)]
    assert find_anomalies(prices, SYMBOL, rules, events) == []
    # 反向验证：若忽略停牌期间的事件，这根 K 线会被误报
    (anomaly,) = find_anomalies(prices, SYMBOL, rules)
    assert anomaly.kind == "unexplained"


def test_multiple_events_in_one_gap_are_chained_not_summed(rules):
    """跨日的多条事件要**逐个**折算（每一步的基数不同），不能求和后一次代入。

    10.00 --10送10--> 5.00 --10派2元--> 4.80，带 [4.32, 5.28]。
    若当成同日一次性代入，基数是 ``(10 − 0.2) / (1 + 1) = 4.90``、带 [4.41, 5.39]，
    就会把合法的 4.32 误报成异常。
    """
    prices = bars(
        {
            "2024-01-02": (10.00, 10.00, 10.00, 10.00),
            "2024-01-31": (4.80, 5.28, 4.32, 4.80),
        }
    )
    events = [bonus("2024-01-10", 10.0), dividend("2024-01-20", 2.0)]
    assert find_anomalies(prices, SYMBOL, rules, events) == []


def test_event_order_does_not_matter(rules):
    """事件按**日期**折算，与传入顺序无关——乱序的输入不该算出不同的限价带。"""
    prices = bars(
        {
            "2024-01-02": (10.00, 10.00, 10.00, 10.00),
            "2024-01-31": (4.80, 5.28, 4.32, 4.80),
        }
    )
    events = [bonus("2024-01-10", 10.0), dividend("2024-01-20", 2.0)]
    assert find_anomalies(prices, SYMBOL, rules, list(reversed(events))) == []


def test_same_day_events_are_summed_in_one_formula(rules):
    """同日多条记录要**求和后代入一次**：ref = (10 − 0.5) / (1 + 1) = 4.75，带 [4.28, 5.23]。"""
    prices = bars(
        {
            "2024-01-02": (10.00, 10.00, 10.00, 10.00),
            "2024-01-03": (4.80, 5.23, 4.28, 4.80),
        }
    )
    events = [dividend("2024-01-03", 5.0), bonus("2024-01-03", 10.0)]
    assert find_anomalies(prices, SYMBOL, rules, events) == []


# --- require_no_anomalies：缺口纪律的落点 ---


def test_require_no_anomalies_passes_when_clean(rules):
    prices = bars(
        {
            "2024-01-02": (10.00, 10.00, 10.00, 10.00),
            "2024-01-03": (10.00, 11.00, 9.00, 10.00),
        }
    )
    assert require_no_anomalies(prices, SYMBOL, rules) is None


def test_require_no_anomalies_raises_with_the_finding(rules):
    prices = bars(
        {
            "2024-01-02": (10.00, 10.00, 10.00, 10.00),
            "2024-01-03": (14.00, 14.00, 14.00, 14.00),
        }
    )
    with pytest.raises(MarketDataError, match="无公司行为可解释"):
        require_no_anomalies(prices, SYMBOL, rules)


def test_require_no_anomalies_reports_the_count_and_caps_the_list(rules):
    """连根越界时报告总数、只列前几条——避免价格表整体错位时刷屏。"""
    days = [f"2024-01-{day:02d}" for day in range(2, 20)]
    # 相邻两根交替 20.00 / 1.00：每一步都远超 ±10%，故每根都是异常
    rows = {
        day: (20.00, 20.00, 20.00, 20.00) if i % 2 else (1.00, 1.00, 1.00, 1.00)
        for i, day in enumerate(days)
    }
    prices = bars({"2024-01-01": (10.00, 10.00, 10.00, 10.00), **rows})
    with pytest.raises(MarketDataError) as excinfo:
        require_no_anomalies(prices, SYMBOL, rules)
    message = str(excinfo.value)
    assert f"有 {len(days)} 处" in message
    assert "…另有" in message


# --- 输入校验 ---


def test_missing_columns_are_rejected(rules):
    prices = pd.DataFrame(
        {"close": [1.0, 2.0]}, index=pd.DatetimeIndex(["2024-01-02", "2024-01-03"])
    )
    with pytest.raises(ValueError, match="缺少列"):
        find_anomalies(prices, SYMBOL, rules)


def test_unsorted_index_is_rejected(rules):
    prices = bars(
        {
            "2024-01-02": (10.0, 10.0, 10.0, 10.0),
            "2024-01-03": (10.0, 10.0, 10.0, 10.0),
        }
    ).iloc[::-1]
    with pytest.raises(ValueError, match="升序"):
        find_anomalies(prices, SYMBOL, rules)


def test_non_positive_price_is_rejected(rules):
    prices = bars(
        {
            "2024-01-02": (10.00, 10.00, 10.00, 10.00),
            "2024-01-03": (0.00, 0.00, 0.00, 0.00),
        }
    )
    with pytest.raises(MarketDataError, match="非正价格"):
        find_anomalies(prices, SYMBOL, rules)


# --- 限价算法（规则层，撮合与质检共用） ---


def test_limit_band_is_the_two_cent_rounded_edges():
    assert limit_band(10.00, 0.10) == (9.00, 11.00)
    assert limit_band(11.26, 0.10) == (10.13, 12.39)


def test_limit_price_uses_decimal_half_up():
    """15.25 × 0.9 = 13.725：十进制半进位取 13.73，二进制 round 给 13.72。"""
    assert limit_price(15.25, 0.10, -1) == 13.73
    assert limit_price(5.35, 0.10, +1) == 5.89


def test_round_to_cent_is_decimal_half_up():
    assert round_to_cent(13.725) == 13.73
    assert round_to_cent(13.715) == 13.72
