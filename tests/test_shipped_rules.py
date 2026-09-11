"""验证出厂规则表：数值与出处一致，缺口报错而非猜，且已声明的局限真的成立。

这张表是**生产默认值**，与前几处的合成夹具不同——它的每条数值都对应一份可查证的
公告，故测试直接断言具体数值与**变更日边界**（变更日当天即生效、前一天仍用旧值）。
出处总表见 ``docs/research/a-share-trading-rules.md``。
"""

from __future__ import annotations

import datetime as dt

import pytest
import tomli

from mbt.backtest.engine import DEFAULT_RULES_PATH
from mbt.rules import RuleTable, RuleTableError


@pytest.fixture
def table():
    return RuleTable.load(DEFAULT_RULES_PATH)


# --- 加载 ---


def test_shipped_table_exists_and_loads(table):
    """出厂表就在默认路径上，且能加载——否则 run_backtest 的默认规则会全线报错。"""
    assert DEFAULT_RULES_PATH.is_file()
    assert table.price_limit("沪主板", dt.date(2026, 1, 5)) == 0.10


def test_run_backtest_uses_the_shipped_table_by_default(make_prices):
    """不传 ``rules`` 时走出厂表——且费用数值确实来自表里，不是零。

    沪主板 2022-04-29 起的过户费为 0.01‰、双向，故买入 21 元 1 股应收 21 × 0.00001。
    若默认规则没被用上（或表读错），这个数字不会对上。
    """
    import backtrader as bt

    from mbt.backtest import run_backtest

    class BuyOnce(bt.Strategy):
        def next(self):
            if not self.position:
                self.buy(size=1)

    result = run_backtest(
        make_prices([20.0, 21.0, 22.0]), symbol="sh600000", strategy=BuyOnce, cash=1000.0
    )

    assert len(result.trades) == 1
    assert result.trades.iloc[0]["commission"] == pytest.approx(21.0 * 0.00001, abs=1e-12)


# --- 日常涨跌幅 ---


def test_daily_limits_by_board(table):
    on = dt.date(2026, 1, 5)

    assert table.price_limit("沪主板", on) == 0.10
    assert table.price_limit("深主板", on) == 0.10
    assert table.price_limit("创业板", on) == 0.20
    assert table.price_limit("科创板", on) == 0.20
    assert table.price_limit("北交所", on) == 0.30


def test_chinext_limit_switches_on_the_reform_day(table):
    """创业板注册制改革：2020-08-24 当天起由 10% 变 20%。"""
    assert table.price_limit("创业板", dt.date(2020, 8, 23)) == 0.10
    assert table.price_limit("创业板", dt.date(2020, 8, 24)) == 0.20


def test_shanghai_main_board_limit_before_1996_is_not_covered(table):
    """沪主板限幅自 1996-12-16 起——更早的日期必须报错，不得默认一个限幅。"""
    with pytest.raises(RuleTableError):
        table.price_limit("沪主板", dt.date(1996, 12, 15))


# --- ST 涨跌幅 ---


def test_st_limits_would_be_correct_if_a_period_were_registered():
    """ST 限幅按板块且按日期取值——用注入的 ST 期间把这张子表验证到。

    出厂表本身不登记任何 ST 期间（见下一条测试），所以这里显式注入一个期间，
    才能走到 ``limit_for`` 的 ST 分支。
    """
    with DEFAULT_RULES_PATH.open("rb") as f:
        raw = tomli.load(f)
    raw["st_period"] = [
        {"symbol": "sh600243", "start": dt.date(1990, 1, 1)},
        {"symbol": "sz300001", "start": dt.date(1990, 1, 1)},
    ]
    table = RuleTable(raw)

    # 主板 ST：5% 自 1998-04-28，2026-07-06 起上调为 10%
    assert table.limit_for("sh600243", dt.date(2026, 7, 5)) == 0.05
    assert table.limit_for("sh600243", dt.date(2026, 7, 6)) == 0.10

    # 创业板 ST：改革前 5%，2020-08-24 起随板块跳到 20%
    assert table.limit_for("sz300001", dt.date(2020, 8, 23)) == 0.05
    assert table.limit_for("sz300001", dt.date(2020, 8, 24)) == 0.20


def test_shipped_table_registers_no_st_period_so_nothing_is_ever_st(table):
    """**已声明的局限**：出厂表不登记任何 ST 期间，故一律按非 ST 取值。

    本地价格数据不含股票名称，无法回溯历史 ST 状态（见
    ``docs/research/tdx-halt-and-limit-representation.md`` 的未决风险）。
    这条测试把「局限真的存在」也钉住：若日后有人填了 ST 期间数据，本测试会失败，
    从而提醒他更新该局限的表述——而不是让文档与实现悄悄分叉。
    """
    assert table.is_st("sh600243", dt.date(2025, 4, 23)) is False
    assert table.is_st("sz000609", dt.date(2026, 4, 23)) is False


# --- 印花税 ---


def test_stamp_duty_halves_on_2023_08_28(table):
    """印花税 0.1% → 0.05%，自 2023-08-28 起（财政部 税务总局公告 2023 年第 39 号）。"""
    assert table.stamp_duty_rate(dt.date(2023, 8, 27)) == 0.001
    assert table.stamp_duty_rate(dt.date(2023, 8, 28)) == 0.0005


def test_stamp_duty_at_the_start_of_the_window(table):
    """窗口起点（2015-01-01）适用的是 2008-09-19 起的 0.1% 卖出单边制度。"""
    assert table.stamp_duty_rate(dt.date(2015, 1, 5)) == 0.001


# --- 过户费 ---


def test_transfer_fee_is_charged_both_ways(table):
    """过户费双向收取——**不是** 2022-04-29 才开始双向的。

    日期按各板块可取的最早时间来选：科创板 2019-07-22 才开板、北交所 2021-11-15
    才有该费率，取更早的日期会（正确地）报错而非返回数值。
    """
    earliest = {
        "沪主板": dt.date(2015, 8, 1),
        "深主板": dt.date(2012, 6, 1),
        "创业板": dt.date(2012, 6, 1),
        "科创板": dt.date(2019, 7, 22),
        "北交所": dt.date(2021, 11, 15),
    }
    for board, since in earliest.items():
        assert table.transfer_fee_sides(board, since) == "both", board
        assert table.transfer_fee_sides(board, dt.date(2026, 1, 5)) == "both", board


def test_boards_are_not_covered_before_they_existed(table):
    """板块有史以来的边界：科创板 2019-07-22 开板、北交所 2021-11-15 开市。

    更早的日期必须报错——给一个尚未存在的板块返回限幅，等于凭空造出一条规则。
    """
    with pytest.raises(RuleTableError):
        table.price_limit("科创板", dt.date(2019, 7, 21))
    with pytest.raises(RuleTableError):
        table.transfer_fee_rate("科创板", dt.date(2019, 7, 21))
    with pytest.raises(RuleTableError):
        table.transfer_fee_rate("北交所", dt.date(2021, 11, 14))


def test_transfer_fee_cuts_on_2022_04_29(table):
    """2022-04-29 是**降费**（0.02‰ → 0.01‰），不是「由单边改双向」。"""
    assert table.transfer_fee_rate("沪主板", dt.date(2022, 4, 28)) == 0.00002
    assert table.transfer_fee_rate("沪主板", dt.date(2022, 4, 29)) == 0.00001


def test_transfer_fee_follows_the_market_not_only_the_board(table):
    """过户费按**市场**收，故科创板跟沪市、创业板跟深市。"""
    assert table.transfer_fee_rate("科创板", dt.date(2019, 7, 22)) == 0.00002
    assert table.transfer_fee_rate("创业板", dt.date(2015, 8, 1)) == 0.00002


def test_shanghai_transfer_fee_before_2015_08_01_is_not_covered(table):
    """沪市在 2015-08-01 前按**成交面额**计费，而面值不在数据里——故刻意留空。

    宁可报错让人看见缺口，也不凭「面值 = 1 元」硬算：面值不恒为 1 元
    （如紫金矿业为 0.1 元），硬算会给这类股票算错成本。
    """
    with pytest.raises(RuleTableError):
        table.transfer_fee_rate("沪主板", dt.date(2015, 7, 31))

    # 深市同期是按成交金额计的，故有据可查、可以给出
    assert table.transfer_fee_rate("深主板", dt.date(2015, 7, 31)) == 0.0000255


# --- 经手费与证管费 ---


def test_handling_fee_cuts_on_2015_08_01_and_2023_08_28(table):
    """经手费两度下调：0.0696‰ → 0.0487‰（2015-08-01）→ 0.0341‰（2023-08-28）。"""
    assert table.handling_fee_rate("沪主板", dt.date(2015, 7, 31)) == 0.0000696
    assert table.handling_fee_rate("沪主板", dt.date(2015, 8, 1)) == 0.0000487
    assert table.handling_fee_rate("沪主板", dt.date(2023, 8, 27)) == 0.0000487
    assert table.handling_fee_rate("沪主板", dt.date(2023, 8, 28)) == 0.0000341


def test_handling_fee_differs_on_the_bse(table):
    """北交所经手费与沪深不同，且有自己的三次费率（0.5‰→0.25‰→0.125‰）。"""
    assert table.handling_fee_rate("北交所", dt.date(2021, 11, 15)) == 0.0005
    assert table.handling_fee_rate("北交所", dt.date(2022, 12, 1)) == 0.00025
    assert table.handling_fee_rate("北交所", dt.date(2023, 8, 28)) == 0.000125


def test_regulatory_fee_is_flat_across_the_window(table):
    """证管费 0.02‰ 自 2012-07-01 起，窗口内未变。"""
    assert table.regulatory_fee_rate("沪主板", dt.date(2015, 1, 5)) == 0.00002
    assert table.regulatory_fee_rate("沪主板", dt.date(2026, 1, 5)) == 0.00002
    assert table.regulatory_fee_rate("北交所", dt.date(2021, 11, 15)) == 0.00002


def test_shanghai_converges_on_2023_08_28_across_fees(table):
    """2023-08-28 是多项费率同日变更日：印花税、经手费、过户费的口径都动。

    这条同时是「按成交日查表」的联合边界测试——若任何一项在变更日两侧取错，
    汇总出来的单边成本就会偏。
    """
    before = dt.date(2023, 8, 27)
    on = dt.date(2023, 8, 28)
    gross = 1_000_000.0

    def per_side(date, selling):
        fee = table.handling_fee_rate("沪主板", date) + table.regulatory_fee_rate("沪主板", date)
        fee += table.transfer_fee_rate("沪主板", date)
        if selling:
            fee += table.stamp_duty_rate(date)
        return fee * gross

    assert per_side(on, selling=True) < per_side(before, selling=True)
    assert per_side(on, selling=False) < per_side(before, selling=False)
