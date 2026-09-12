"""指标口径（票据 #10，AC 1–2）。

AC 要求「每个指标的计算口径以公式写进文档，读者不必猜」。故这里的断言一律写成**独立的
算式**（手算的常数或按定义直接写出的表达式），而不是照实现抄一遍——否则测试与实现同源，
改错了也发现不了。

三个口径最容易各算各的，各有一条专门钉住：年化用**交易日**、夏普的无风险利率按**几何**
折算、以及无波动时返回**缺失而非无穷**。
"""

from __future__ import annotations

import math

import pandas as pd
import pytest
from helpers import curve, trades_table

from mbt.metrics import Metrics, closed_trade_pnls, compute_metrics

# --- 年化：用交易日数、按几何 -----------------------------------------------


def test_annual_return_over_exactly_one_year_is_the_total_return():
    """一期恰好 252 个交易日时，年化就是总收益本身——年化不该在整年处引入偏差。

    净值按 ``1.10 ** (i/252)`` 构造，故总收益恰为 10%，且构造方式与实现无关。
    """
    equity = curve([100.0 * 1.10 ** (i / 252) for i in range(253)])

    got = compute_metrics(equity)

    assert got.total_return == pytest.approx(0.10)
    assert got.annual_return == pytest.approx(0.10)


def test_annual_return_compounds_a_half_year_return_to_a_full_year():
    """半年翻倍 → 年化 4 倍（几何复利），这是「几何」与「线性外推」的分野。

    126 期、总收益 100%：``(1+1)^(252/126) − 1 = 3``。若实现按线性外推会得到 2.0。
    """
    equity = curve([100.0] * 126 + [200.0])

    got = compute_metrics(equity)

    assert got.annual_return == pytest.approx(3.0)


def test_annual_return_is_missing_when_there_is_only_one_point():
    """只有一个点时没有期数可言，年化缺失而不是编一个数出来。"""
    got = compute_metrics(curve([100.0]))

    assert math.isnan(got.annual_return)


def test_a_total_loss_is_not_annualized_into_a_meaningless_number():
    """亏到 0 时年化无定义（``(1+r)`` 为 0 的分数次幂），返回缺失。"""
    got = compute_metrics(curve([100.0, 50.0, 0.0]))

    assert got.total_return == pytest.approx(-1.0)
    assert math.isnan(got.annual_return)


# --- 夏普：无风险利率按几何折算、样本标准差 --------------------------------


def test_sharpe_matches_the_hand_computed_ratio_when_not_annualized():
    """日收益为 1% 与 2% 时：均值 1.5%，样本标准差 ``0.5% × √2``，故比值为 ``3/√2``。

    取 ``periods_per_year=1`` 以去掉年化因子，期望值是可手算的 ``3/√2``。
    """
    equity = curve([100.0, 101.0, 103.02])

    got = compute_metrics(equity, periods_per_year=1)

    assert got.sharpe == pytest.approx(3.0 / math.sqrt(2.0))


def test_sharpe_annualizes_by_the_square_root_of_the_period_count():
    """年化就是把上一条的比值乘 ``√252``——这一条与上一条一起把公式完全钉死。"""
    equity = curve([100.0, 101.0, 103.02])

    assert compute_metrics(equity, periods_per_year=252).sharpe == pytest.approx(
        3.0 / math.sqrt(2.0) * math.sqrt(252.0)
    )


def test_the_risk_free_rate_is_converted_geometrically_not_by_dividing():
    """无风险利率按 ``(1+rf)^(1/252) − 1`` 折算，**不是** ``rf/252``。

    取 rf = 10%：几何日频为 ``1.1^(1/252) − 1``（约 0.0378%），而除以 252 给 0.0397%——
    两者在这个量级上可区分，故这条能真正区分实现。
    """
    equity = curve([100.0, 101.0, 103.02])
    geometric_daily = 1.10 ** (1.0 / 252.0) - 1.0
    returns = equity.pct_change().dropna()
    std = float(returns.std(ddof=1))
    expected = float((returns - geometric_daily).mean()) / std * math.sqrt(252.0)

    got = compute_metrics(equity, risk_free=0.10)

    assert got.sharpe == pytest.approx(expected)
    assert got.sharpe != pytest.approx(compute_metrics(equity).sharpe), "rf 没有生效"


def test_sharpe_is_missing_rather_than_infinite_when_there_is_no_volatility():
    """收益率全等时标准差为 0，比值无定义——返回**缺失**而不是无穷。

    ``inf`` 会污染任何比较（``inf > 3`` 恒真）与图表坐标，而「算不出来」与「好得没边」
    是两件事。
    """
    equity = curve([100.0, 101.0, 102.01])

    got = compute_metrics(equity)

    assert math.isnan(got.sharpe)
    assert got.sharpe != float("inf")
    assert any("无波动" in note for note in got.notes)


def test_sharpe_is_missing_with_fewer_than_two_returns():
    """样本不足两期时样本标准差无定义。"""
    assert math.isnan(compute_metrics(curve([100.0, 101.0])).sharpe)


def test_a_negative_sharpe_stays_negative():
    """亏损策略的夏普必须为负——正负号是它的主要用途，不能被取绝对值。"""
    equity = curve([100.0, 99.0, 98.01, 97.0299])

    assert compute_metrics(equity).sharpe < 0


# --- 最大回撤：正值 ----------------------------------------------------------


def test_max_drawdown_is_the_deepest_peak_to_trough_decline_as_a_positive_number():
    """[100, 120, 90, 130] 的最大回撤是 ``(120−90)/120 = 0.25``，报为**正**值。"""
    got = compute_metrics(curve([100.0, 120.0, 90.0, 130.0]))

    assert got.max_drawdown == pytest.approx(0.25)


def test_a_monotonically_rising_curve_has_no_drawdown():
    got = compute_metrics(curve([100.0, 110.0, 120.0]))

    assert got.max_drawdown == pytest.approx(0.0)


def test_max_drawdown_uses_the_running_peak_not_the_first_value():
    """回撤的参照是**运行最高点**：从 120 跌到 90 是 25%，不是从 100 算的 10%。"""
    one_peak = compute_metrics(curve([100.0, 120.0, 90.0])).max_drawdown
    from_start = compute_metrics(curve([100.0, 90.0])).max_drawdown

    assert one_peak == pytest.approx(0.25)
    assert from_start == pytest.approx(0.10)


# --- 平仓交易的配对（胜率与盈亏比的分母） ----------------------------------


def test_each_sell_produces_one_closed_trade_with_its_realised_pnl():
    """买入 100@10、卖出 100@12 → 一笔平仓，盈亏 200 元（无费用时）。"""
    pnls = closed_trade_pnls(trades_table([(0, 100, 10.0, 0.0), (1, -100, 12.0, 0.0)]))

    assert pnls == pytest.approx([200.0])


def test_commissions_are_deducted_from_the_realised_pnl():
    """费用计入已实现盈亏：买 5 元 + 卖 5 元 → 200 − 10 = 190。"""
    pnls = closed_trade_pnls(trades_table([(0, 100, 10.0, 5.0), (1, -100, 12.0, 5.0)]))

    assert pnls == pytest.approx([190.0])


def test_a_partial_sell_realises_proportionally_and_a_later_sell_finishes_it():
    """买 100、卖 40、再卖 60：两笔平仓，各按比例实现（FIFO 按股数）。"""
    pnls = closed_trade_pnls(
        trades_table([(0, 100, 10.0, 0.0), (1, -40, 12.0, 0.0), (2, -60, 14.0, 0.0)])
    )

    assert pnls == pytest.approx([80.0, 240.0])


def test_fifo_matches_the_oldest_lot_first():
    """两批买入价不同：卖出先扣最早那批。

    买 100@10（第 1 天）、买 100@20（第 2 天）、卖 100@30 → 匹配的是 @10 那批，
    故盈亏 ``(30−10)×100 = 2000``，而不是 ``(30−20)×100 = 1000``。
    """
    pnls = closed_trade_pnls(
        trades_table([(0, 100, 10.0, 0.0), (1, 100, 20.0, 0.0), (2, -100, 30.0, 0.0)])
    )

    assert pnls == pytest.approx([2000.0])


def test_an_open_position_is_not_counted_as_a_closed_trade():
    """只买未卖 → 没有平仓交易。拿浮动盈亏当一笔「赢」是两类不同的统计对象。"""
    pnls = closed_trade_pnls(trades_table([(0, 100, 10.0, 0.0)]))

    assert pnls == []


def test_selling_more_than_held_is_rejected():
    """卖出量超过持仓说明明细不可信——静默按可用量截断会让胜率的分母凭空变小。"""
    with pytest.raises(ValueError, match="超过当时持仓"):
        closed_trade_pnls(trades_table([(0, 100, 10.0, 0.0), (1, -200, 12.0, 0.0)]))


# --- 配对必须**按标的分别**做（多标的组合） ----------------------------------


def test_a_buy_in_one_symbol_is_not_paired_with_a_sell_in_another():
    """**本组核心**：组合里 A 的买入不得被当成 B 卖出的对手方。

    这两个标的一个买、一个卖，在时间上相邻。若共用一条队列，就会算出
    ``(30 − 10) × 100 = 2000`` 这样**无意义的盈亏**——而胜率与盈亏比都建在它上面。

    正确行为是：B 在没有持仓的情况下卖出，本身就是明细不可信，**报错并点名标的**。
    """
    a = trades_table([(0, 100, 10.0, 0.0)], symbol="sh600000")
    b = trades_table([(1, -100, 30.0, 0.0)], symbol="sz000001")
    mixed = pd.concat([a, b], ignore_index=True)

    with pytest.raises(ValueError, match="sz000001"):
        closed_trade_pnls(mixed)


def test_two_symbols_each_round_tripped_are_paired_separately():
    """两个标的各自完成一次往返 → 两笔平仓，盈亏**各归各的**。

    A 赚 200、B 亏 100。若按单一队列配对，A 的买入会与 B 的卖出（或反之）配错，
    ``2000``、``-1500`` 这类数字就会冒出来——故这里逐笔断言。
    """
    a = trades_table([(0, 100, 10.0, 0.0), (3, -100, 12.0, 0.0)], symbol="sh600000")
    b = trades_table([(1, 100, 20.0, 0.0), (2, -100, 19.0, 0.0)], symbol="sz000001")
    mixed = pd.concat([a, b], ignore_index=True)

    pnls = closed_trade_pnls(mixed)

    assert sorted(pnls) == pytest.approx([-100.0, 200.0])


def test_interleaved_symbols_do_not_leak_into_each_other():
    """交错下单：A 买、B 买、A 卖、B 卖——每一笔都只与自己标的的买入配对。

    A：100@10 → 100@11，赚 100；B：100@20 → 100@18，亏 200。
    单一队列会把「先买先配」错用到跨标的上，得出的数字与这两个都不符。
    """
    a = trades_table(
        [(0, 100, 10.0, 0.0), (2, -100, 11.0, 0.0)],
        symbol="sh600000",
    )
    b = trades_table(
        [(1, 100, 20.0, 0.0), (3, -100, 18.0, 0.0)],
        symbol="sz000001",
    )
    mixed = pd.concat([a, b], ignore_index=True)

    pnls = closed_trade_pnls(mixed)

    assert sorted(pnls) == pytest.approx([-200.0, 100.0])


def test_a_trades_frame_without_a_symbol_column_is_rejected():
    """缺 ``symbol`` 列 → 报错，而不是**退回**「混着配对」。

    退回去就等于把上面那个错悄悄算出来；而它不会报错，只会让胜率与盈亏比失真。
    """
    frame = trades_table([(0, 100, 10.0, 0.0), (1, -100, 12.0, 0.0)]).drop(columns=["symbol"])

    with pytest.raises(ValueError, match="缺少 symbol 列"):
        closed_trade_pnls(frame)


# --- 胜率与盈亏比 ------------------------------------------------------------


def test_win_rate_and_payoff_use_the_closed_trades():
    """两笔平仓：赚 200 与亏 100 → 胜率 1/2，盈亏比 200/100 = 2。"""
    trades = trades_table(
        [
            (0, 100, 10.0, 0.0),
            (1, -100, 12.0, 0.0),
            (2, 100, 10.0, 0.0),
            (3, -100, 9.0, 0.0),
        ]
    )

    got = compute_metrics(curve([1000.0, 1200.0, 1100.0, 1200.0, 1100.0]), trades)

    assert got.closed_trades == 2
    assert got.win_rate == pytest.approx(0.5)
    assert got.payoff_ratio == pytest.approx(2.0)


def test_payoff_ratio_is_missing_when_no_trade_lost():
    """没有亏损笔时盈亏比无定义——返回缺失，而不是无穷。"""
    trades = trades_table([(0, 100, 10.0, 0.0), (1, -100, 12.0, 0.0)])

    got = compute_metrics(curve([1000.0, 1200.0]), trades)

    assert got.win_rate == pytest.approx(1.0)
    assert math.isnan(got.payoff_ratio)
    assert any("没有亏损" in note for note in got.notes)


def test_flat_closed_trades_do_not_count_as_wins():
    """打平的平仓不算赢——胜率的分母包含它，分子不包含。"""
    trades = trades_table([(0, 100, 10.0, 0.0), (1, -100, 10.0, 0.0)])

    got = compute_metrics(curve([1000.0, 1000.0]), trades)

    assert got.win_rate == pytest.approx(0.0)


def test_no_trades_makes_the_trade_based_metrics_missing_with_a_reason():
    """没有成交时，依赖成交的指标缺失，且**注明原因**——否则读者会以为算错了。"""
    got = compute_metrics(curve([1000.0, 1100.0]))

    assert math.isnan(got.win_rate)
    assert math.isnan(got.payoff_ratio)
    assert got.turnover == pytest.approx(0.0)
    assert any("没有平仓交易" in note for note in got.notes)


# --- 换手率：单边、期间累计、不年化 ------------------------------------------


def test_turnover_is_one_when_the_whole_value_round_trips_once():
    """净值恒为 1000、买 1000 再卖 1000：``(1000+1000)/2/1000 = 1``。

    「单边」即除以 2：一次完整往返算**一轮**。若不除以 2 会得到 2.0。
    """
    trades = trades_table([(0, 100, 10.0, 0.0), (1, -100, 10.0, 0.0)])

    got = compute_metrics(curve([1000.0] * 3), trades)

    assert got.turnover == pytest.approx(1.0)


def test_turnover_uses_the_mean_equity_not_the_starting_equity():
    """分母是**平均**净值（资金规模在变），不是期初。"""
    trades = trades_table([(0, 100, 10.0, 0.0)])
    equity = curve([1000.0, 2000.0, 3000.0])  # 均值 2000

    got = compute_metrics(equity, trades)

    assert got.turnover == pytest.approx(1000.0 / 2.0 / 2000.0)


def test_turnover_is_scaled_by_the_actual_traded_value():
    """换手率随成交金额线性变化，故可用来比较不同资金的同一策略。"""
    small = compute_metrics(curve([1000.0] * 3), trades_table([(0, 100, 10.0, 0.0)])).turnover
    large = compute_metrics(curve([1000.0] * 3), trades_table([(0, 200, 10.0, 0.0)])).turnover

    assert large == pytest.approx(2.0 * small)


# --- 基准与超额 --------------------------------------------------------------


def test_the_benchmark_annualizes_on_the_same_convention():
    """基准与策略用同一套年化口径，否则「超额」没有可比性。"""
    equity = curve([100.0] * 126 + [200.0])
    benchmark = curve([100.0 * 1.10 ** (i / 252) for i in range(127)])

    got = compute_metrics(equity, benchmark_equity=benchmark)

    assert got.benchmark_annual_return == pytest.approx(0.10)
    assert got.excess_annual_return == pytest.approx(3.0 - 0.10)


def test_a_losing_strategy_can_have_a_worse_than_benchmark_excess():
    """超额为负是合法且重要的结论：涨了 20% 在基准涨 30% 的年份是失败（规格故事 53）。"""
    equity = curve([100.0 * 1.20 ** (i / 252) for i in range(253)])
    benchmark = curve([100.0 * 1.30 ** (i / 252) for i in range(253)])

    got = compute_metrics(equity, benchmark_equity=benchmark)

    assert got.excess_annual_return < 0


def test_without_a_benchmark_the_benchmark_fields_are_absent():
    got = compute_metrics(curve([100.0, 110.0]))

    assert got.benchmark_annual_return is None
    assert got.excess_annual_return is None


# --- 输入纪律 ----------------------------------------------------------------


def test_an_empty_equity_curve_is_rejected():
    """「没跑起来」与「跑了但没交易」是两回事：前者报错，后者是合法产物。"""
    with pytest.raises(ValueError, match="净值曲线为空"):
        compute_metrics(pd.Series(dtype=float))


def test_a_gap_in_the_equity_curve_is_rejected_not_filled():
    """净值空缺不能就地补——补一条等于凭空造出一天的资产，且不会在图上报异常。"""
    with pytest.raises(ValueError, match="含有缺失值"):
        compute_metrics(curve([100.0, float("nan"), 120.0]))


def test_every_missing_metric_is_accompanied_by_a_reason():
    """**每一条**缺失都要说明原因，否则读者会以为算错了。

    逐个覆盖会缺的项与不会缺的项：这里是「只有一个点」（年化与夏普都缺）。
    """
    got = compute_metrics(curve([100.0]))

    assert math.isnan(got.annual_return)
    assert math.isnan(got.sharpe)
    assert any("年化收益缺失" in note for note in got.notes)
    assert any("夏普缺失" in note for note in got.notes)


def test_a_total_loss_explains_why_the_annualisation_is_missing():
    """亏光时年化无定义，也要给出原因——这是最需要解释的一种缺失。"""
    got = compute_metrics(curve([100.0, 50.0, 0.0]))

    assert any("年化收益缺失" in note for note in got.notes)


def test_an_unusable_benchmark_is_explained_too():
    """基准只有一个点时，基准年化缺失也要有原因。"""
    got = compute_metrics(curve([100.0, 110.0]), benchmark_equity=curve([3000.0]))

    assert got.benchmark_annual_return is None
    assert any("基准年化缺失" in note for note in got.notes)


def test_metrics_are_json_serializable():
    """产物要落盘，故指标必须能序列化——``notes`` 是元组，故专门转一次。"""
    import json

    payload = compute_metrics(curve([100.0, 110.0])).as_dict()

    assert json.loads(json.dumps(payload))["total_return"] == pytest.approx(0.10)
    assert isinstance(payload["notes"], list)


def test_metrics_is_frozen_so_a_result_cannot_be_edited_after_the_fact():
    """指标是证据，不该在产出后被就地改掉。"""
    import dataclasses

    got = compute_metrics(curve([100.0, 110.0]))

    with pytest.raises(dataclasses.FrozenInstanceError):
        got.annual_return = 99.0  # type: ignore[misc]


def test_the_dataclass_exposes_every_metric_the_ac_asks_for():
    """AC 列的六项一个都不能少，且字段名可被 `as_dict` 直接落盘。"""
    got = compute_metrics(curve([100.0, 110.0]), trades_table([(0, 100, 10.0, 0.0)]))

    assert isinstance(got, Metrics)
    for field in (
        "annual_return",
        "sharpe",
        "max_drawdown",
        "win_rate",
        "payoff_ratio",
        "turnover",
    ):
        assert field in got.as_dict()
