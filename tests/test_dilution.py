"""判别类别 1 事件是否**真的**稀释了价格（票据 #20）。

要修的是这样一件事：`gbbq` 把「不改变总股本」的公司行为（典型是 2005–2007 年**股改对价
送股**）也记为类别 1，而复权照它稀释，于是**后复权凭空插入约 +22% 的假跳空**——价格里没有
这件事。它会污染均线、动量与所有收益数字，且**不报错**。

判据是「比两个假说各自的涨跌停带」：带宽取自规则表（按板块与成交日），故创业板 20%、
北交所 30% 与主板 10% 各自正确。
"""

from __future__ import annotations

import pandas as pd
import pytest

from mbt.data import AdjustmentEvent, MarketDataError
from mbt.data.dilution import (
    BELOW_THRESHOLD,
    DILUTED,
    DILUTION_THRESHOLD,
    NOT_DILUTED,
    OUTSIDE_SERIES,
    resolve_dilution,
)
from mbt.rules import RuleTable

#: 「主板 10%」足以测判定逻辑；制度数值本身由出厂表的测试负责。
MAIN_BOARD_RULES = """
schema_version = 1

[[price_limit]]
board = "沪主板"
effective_from = 2015-01-01
limit = 0.10

[[stamp_duty]]
effective_from = 2015-01-01
sell_rate = 0.001
"""


@pytest.fixture
def main_board_rules(tmp_path):
    path = tmp_path / "main-board.toml"
    path.write_text(MAIN_BOARD_RULES, encoding="utf-8")
    return RuleTable.load(path)


#: 故意把限幅设得足够宽，使两个假说的带**必然重叠**——用来验证「就近取一个并标低置信」。
WIDE_LIMIT_RULES = """
schema_version = 1

[[price_limit]]
board = "沪主板"
effective_from = 2015-01-01
limit = 0.60

[[stamp_duty]]
effective_from = 2015-01-01
sell_rate = 0.001
"""


@pytest.fixture
def wide_limit_rules(tmp_path):
    path = tmp_path / "wide-limit.toml"
    path.write_text(WIDE_LIMIT_RULES, encoding="utf-8")
    return RuleTable.load(path)


def bars(closes, start="2024-01-02"):
    """只填 close 的极简价格表——判定只用收盘价。"""
    return pd.DataFrame(
        {"close": closes},
        index=pd.bdate_range(start, periods=len(closes)),
    )


def event(on, index, *, bonus=0.0, cash=0.0, rights=0.0, rights_price=0.0):
    """构造一条事件；``index`` 是价格表，用于把序号换成日期。"""
    return AdjustmentEvent(
        symbol="sh600000",
        ex_date=index[on].date(),
        cash_per_10=cash,
        bonus_per_10=bonus,
        rights_price=rights_price,
        rights_per_10=rights,
    )


# --- 真稀释：照旧参与复权 ------------------------------------------------------


def test_a_genuinely_diluting_event_is_kept(main_board_rules):
    """10送3 且价格**确实**按 1.3 稀释（收在参考价附近）→ 照旧复权。

    第 3 根收在 7.70 ≈ 前收 10.00 ÷ 1.3 = 7.69，落在已稀释带内且不在未稀释带内。
    """
    prices = bars([10.00, 10.00, 7.70])
    events = [event(2, prices.index, bonus=3.0)]

    effective, verdicts = resolve_dilution(prices, events, "sh600000", main_board_rules)

    assert len(effective) == 1
    assert effective[0].bonus_per_10 == 3.0
    assert verdicts[0].verdict == DILUTED
    assert verdicts[0].low_confidence is False


# --- 假稀释：股改对价送股，丢掉稀释成分 ----------------------------------------


def test_a_non_diluting_event_loses_only_its_dilution_component(main_board_rules):
    """**这就是本票要修的缺陷**：2006 年那种「10送3 但价格没稀释」的事件。

    第 3 根收在 9.40（≈ −6%，正常波动）→ 判为未稀释，故送转被置零。
    """
    prices = bars([10.00, 10.00, 9.40])
    events = [event(2, prices.index, bonus=3.0)]

    effective, verdicts = resolve_dilution(prices, events, "sh600000", main_board_rules)

    assert verdicts[0].verdict == NOT_DILUTED
    assert effective[0].bonus_per_10 == 0.0, "稀释成分必须丢掉"


def test_a_non_diluting_event_keeps_its_cash_dividend(main_board_rules):
    """同日既有现金分红又有（不稀释的）送转时，**只丢稀释成分、保留分红**。

    分红是真的——那天公司确实付了钱。整条事件作废会把这笔真实的分红也丢掉。
    """
    prices = bars([10.00, 10.00, 9.40])
    events = [event(2, prices.index, bonus=3.0, cash=5.0)]

    effective, _ = resolve_dilution(prices, events, "sh600000", main_board_rules)

    assert effective[0].bonus_per_10 == 0.0
    assert effective[0].cash_per_10 == 5.0, "现金分红不因送转不稀释而消失"


def test_the_rights_issue_component_is_dropped_too(main_board_rules):
    """配股同属稀释成分，判为未稀释时一并置零（连配股价一起）。

    配股价取 1.0（远低于市价）使参考价明显下移（7.92），两个带因此不重叠——
    这样测的才是「丢掉稀释成分」，而不是撞上重叠时的就近判定。
    """
    prices = bars([10.00, 10.00, 9.40])
    events = [event(2, prices.index, rights=3.0, rights_price=1.0)]

    effective, verdicts = resolve_dilution(prices, events, "sh600000", main_board_rules)

    assert verdicts[0].verdict == NOT_DILUTED
    assert verdicts[0].low_confidence is False
    assert effective[0].rights_per_10 == 0.0
    assert effective[0].rights_price == 0.0


def test_the_sh600000_fixture_case_is_judged_not_diluted(main_board_rules):
    """把 2006-05-12 的真实数字搬进来：10.86 → 10.21（−5.99%）。

    真实数据里这条事件被记为「10送3」，而复权照它稀释、在后复权序列里插进 +22% 的假跳空。
    这条用例把判定钉在这个具体数字上。
    """
    prices = bars([10.86, 10.86, 10.21])
    events = [event(2, prices.index, bonus=3.0)]

    effective, verdicts = resolve_dilution(prices, events, "sh600000", main_board_rules)

    assert verdicts[0].verdict == NOT_DILUTED
    assert verdicts[0].actual_ratio == pytest.approx(10.21 / 10.86, abs=1e-9)
    # 理论比值用的是**取整到分**的参考价（交易所公布的那种），不是未取整的 10.86/1.3。
    assert verdicts[0].diluted_ratio == pytest.approx(8.35 / 10.86, abs=1e-9)
    assert effective[0].bonus_per_10 == 0.0


# --- 阈值之下不判 --------------------------------------------------------------


def test_events_below_the_threshold_are_left_alone(main_board_rules):
    """稀释幅度太小则不判：噪声与真实事件在那个量级上分不开，而误判的代价也只有那个量级。"""
    prices = bars([10.00, 10.00, 9.00])
    events = [event(2, prices.index, bonus=DILUTION_THRESHOLD - 0.01)]

    effective, verdicts = resolve_dilution(prices, events, "sh600000", main_board_rules)

    assert verdicts[0].verdict == BELOW_THRESHOLD
    assert effective[0] == events[0], "未判定的事件原样保留"


def test_the_threshold_itself_is_judged(main_board_rules):
    """恰在阈值上要判（边界是闭的）——否则阈值附近的规则说不清。

    注意 ``bonus_per_10`` 是**每 10 股**，故阈值的 0.25 对应 ``bonus_per_10 = 2.5``。
    """
    prices = bars([10.00, 10.00, 7.70])
    events = [event(2, prices.index, bonus=DILUTION_THRESHOLD * 10)]

    _, verdicts = resolve_dilution(prices, events, "sh600000", main_board_rules)

    assert verdicts[0].dilution == pytest.approx(DILUTION_THRESHOLD)
    assert verdicts[0].verdict != BELOW_THRESHOLD


# --- 两个带都落不进 → 报错 ----------------------------------------------------


def test_a_gap_that_neither_hypothesis_explains_is_rejected(main_board_rules):
    """跳空既非正常波动、亦非事件所致（多半是长期停牌复牌）→ **报错**，不假装解释掉。

    前收 10.00、事件说 10送3，而实际收在 4.00（−60%）——两个带都进不去。
    """
    prices = bars([10.00, 10.00, 4.00])
    events = [event(2, prices.index, bonus=3.0)]

    with pytest.raises(MarketDataError, match="两个假说的带都落不进"):
        resolve_dilution(prices, events, "sh600000", main_board_rules)


def test_the_error_message_carries_both_hypotheses(main_board_rules):
    """消息要给全四样，否则读者无法判断该切片还是该记为存疑。"""
    prices = bars([10.00, 10.00, 4.00])
    events = [event(2, prices.index, bonus=3.0)]

    with pytest.raises(MarketDataError) as info:
        resolve_dilution(prices, events, "sh600000", main_board_rules)

    message = str(info.value)
    assert "sh600000" in message
    assert "实际比值" in message
    assert "未稀释假说期望" in message
    assert "已稀释假说期望" in message


# --- 两个带都落进 → 就近取一个并标低置信 --------------------------------------


def test_overlapping_bands_pick_the_nearest_hypothesis_and_flag_low_confidence(wide_limit_rules):
    """宽限幅板块上两个带会重叠，此时判据没有更多信息可用——就近取一个并标低置信。

    构造：限幅 0.60（足够宽，两个带必然重叠），前收 10、10送3 → 参考价 7.69。
    未稀释带 [4.00, 16.00]、已稀释带 [3.08, 12.31]。收在 6.00：距 7.69 为 1.69，
    距 10.00 为 4.00，故判**已稀释**。
    """
    prices = bars([10.00, 10.00, 6.00])
    events = [event(2, prices.index, bonus=3.0)]

    effective, verdicts = resolve_dilution(prices, events, "sh600000", wide_limit_rules)

    assert verdicts[0].low_confidence is True
    assert verdicts[0].verdict == DILUTED
    assert len(effective) == 1


def test_overlapping_bands_can_also_land_on_not_diluted(wide_limit_rules):
    """同一套判据在就近偏向「未稀释」时给出 NOT_DILUTED——方向由距离决定，不是写死的。"""
    prices = bars([10.00, 10.00, 9.80])  # 距 10.00 仅 0.20，距 7.69 为 2.11
    events = [event(2, prices.index, bonus=3.0)]

    _, verdicts = resolve_dilution(prices, events, "sh600000", wide_limit_rules)

    assert verdicts[0].low_confidence is True
    assert verdicts[0].verdict == NOT_DILUTED


# --- 边界与时点 ----------------------------------------------------------------


def test_an_event_on_a_suspension_day_is_judged_at_the_resumption_bar(main_board_rules):
    """除权日落在停牌区间内时，判定看**复牌首根**——那才是除权效果显现的地方。

    价格表第 2 行是复牌日（中间那天没有 K 线），事件日期落在停牌那天。
    """
    index = pd.bdate_range("2024-01-02", periods=3)
    prices = pd.DataFrame({"close": [10.0, 9.4]}, index=[index[0], index[2]])
    events = [AdjustmentEvent(symbol="sh600000", ex_date=index[1].date(), bonus_per_10=3.0)]

    effective, verdicts = resolve_dilution(prices, events, "sh600000", main_board_rules)

    assert verdicts[0].verdict == NOT_DILUTED, "看的是复牌首根（−6%），而非停牌那天"
    assert effective[0].bonus_per_10 == 0.0


def test_an_event_outside_the_series_is_passed_through(main_board_rules):
    """序列之外的事件本就不参与复权，判定也无从谈起——原样放过，不报错。"""
    prices = bars([10.00, 10.00], start="2024-06-03")
    events = [
        AdjustmentEvent(
            symbol="sh600000",
            ex_date=pd.Timestamp("2024-01-02").date(),
            bonus_per_10=3.0,
        )
    ]

    effective, verdicts = resolve_dilution(prices, events, "sh600000", main_board_rules)

    assert verdicts[0].verdict == OUTSIDE_SERIES
    assert effective[0].bonus_per_10 == 3.0


def test_multiple_events_on_the_same_day_are_judged_together(main_board_rules):
    """同日多条必须**合起来**判：参考价要把同日各量求和后一次代入。"""
    prices = bars([10.00, 10.00, 7.70])
    events = [
        event(2, prices.index, bonus=2.0),
        event(2, prices.index, bonus=1.0),
    ]

    _, verdicts = resolve_dilution(prices, events, "sh600000", main_board_rules)

    assert len(verdicts) == 1, "同日合为一组，故只有一条判定记录"
    assert verdicts[0].dilution == pytest.approx(0.3)
    assert verdicts[0].verdict == DILUTED


def test_the_judgement_is_causal(main_board_rules):
    """判定只看事件日及之前：截断到第 k 根重算，前 k 根内的判定必须逐条相同。

    ADR-0006 的硬约束。本票第一次让复权依赖**价格**（不只是事件本身），故显式钉住。
    """
    closes = [10.00, 10.00, 7.70, 8.20, 8.10]
    prices = bars(closes)
    events = [event(2, prices.index, bonus=3.0)]

    full_effective, full_verdicts = resolve_dilution(prices, events, "sh600000", main_board_rules)

    for rows in (3, 4, 5):
        truncated = bars(closes[:rows])
        effective, verdicts = resolve_dilution(truncated, events, "sh600000", main_board_rules)
        assert [v.verdict for v in verdicts] == [v.verdict for v in full_verdicts]
        assert effective[0].bonus_per_10 == full_effective[0].bonus_per_10


def test_a_price_table_without_close_is_rejected(main_board_rules):
    with pytest.raises(ValueError, match="必须含 close"):
        resolve_dilution(
            pd.DataFrame({"open": [1.0]}, index=pd.bdate_range("2024-01-02", periods=1)),
            [],
            "sh600000",
            main_board_rules,
        )


def test_a_date_outside_the_rule_table_window_fails_rather_than_guessing():
    """规则表不覆盖该日期时**报错**，不猜一个常数带宽——与「都落不进就报错」同一立场。

    1996-12-16 之前的沪深主板没有涨跌幅制度可言，硬套一个 10% 会判错。
    """
    from mbt.rules import RuleTableError

    rules = RuleTable.load()
    index = pd.bdate_range("1995-01-02", periods=3)
    prices = pd.DataFrame({"close": [10.0, 10.0, 9.4]}, index=index)
    events = [AdjustmentEvent(symbol="sh600000", ex_date=index[2].date(), bonus_per_10=3.0)]

    with pytest.raises(RuleTableError):
        resolve_dilution(prices, events, "sh600000", rules)
