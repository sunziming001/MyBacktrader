"""ST 期间的推断（票据 #53）。

本文件的重心是**判据的两条同时成立**：单看任何一条都会错，而两条错误方向相反——

- 只看「窗口内最大幅度 ≤ 5.4%」→ 把安静的大盘股全判成 ST（实测工行 1.97%）；
- 只看「打到过 ±5% 限价」→ 把大量正常股判成 ST（实测 79.5% 的主板股都有这种日子）。
"""

from __future__ import annotations

import datetime as dt

import pandas as pd
import pytest

from mbt.data.st import DEFAULT_WINDOW, infer_st_periods
from mbt.rules import RuleTable, limit_price

MAIN_BOARD_RULES = """
schema_version = 1

[[price_limit]]
board = "沪主板"
effective_from = 1990-01-01
limit = 0.10

[[price_limit]]
board = "创业板"
effective_from = 1990-01-01
limit = 0.20

[[stamp_duty]]
effective_from = 1990-01-01
sell_rate = 0.001
"""


@pytest.fixture
def rules(tmp_path):
    path = tmp_path / "rules.toml"
    path.write_text(MAIN_BOARD_RULES, encoding="utf-8")
    return RuleTable.load(path)


def series(values, start="2024-01-02"):
    """只填 close 的字段宽表。"""
    return pd.DataFrame(
        {"close": [float(v) for v in values]},
        index=pd.bdate_range(start, periods=len(values)),
    )


def walk(prices, steps, *, cap=0.05, start="2024-01-02"):
    """由价格序列与每日涨跌幅构造收盘价序列：每步都**恰好**走到给定幅度。

    ``cap`` 给 0.05 时，每步都精确落在 5% 限价上（用 ``limit_price`` 算，与实现同一口径）。
    """
    out = [prices]
    current = prices
    for step in steps:
        current = limit_price(current, step, 1) if step > 0 else limit_price(current, -step, -1)
        out.append(current)
    return out


def test_a_window_that_never_exceeds_5_percent_and_hits_the_limit_is_st(rules):
    """**判据的两条同时成立**才有结论：波动被封在 5% 内，且确实打过 5% 的板。"""
    # 20 次精确打到 +5% 限价，窗口内最大幅度恰好是 5%
    values = walk(10.0, [0.05] * (DEFAULT_WINDOW + 4))

    periods = infer_st_periods(series(values), "sh600000", rules)

    assert periods, "应当判出 ST 期间"
    assert periods[0].evidence_days >= 2


def test_a_quiet_stock_that_never_hits_the_limit_is_not_st(rules):
    """**只看「没波动」会误判**：安静的大盘股从不打板，故不是 ST。

    实测工商银行有 120 日窗口的最大幅度只有 1.97%，但它绝不是 ST。
    """
    values = walk(10.0, [0.002] * (DEFAULT_WINDOW + 4))

    assert infer_st_periods(series(values), "sh600000", rules) == ()


def test_a_stock_that_reaches_the_10_percent_limit_is_not_st(rules):
    """**只看「打过 5%」也会误判**：±10% 的带里，「恰好涨跌 5%」是正常事件。

    构造：多数日子恰好 +5%，但**每隔一小段就有一天走到 +9.9%**——于是不存在任何 40 日窗口
    是「干净被 5% 封顶」的，整段都判不出 ST。

    （若只在开头放一天大波动就不同了：那之后的窗口仍然干净，检测器会判「自那天起转为
    ST」——那实际上是**合理**的推断，故本测试要的是**贯穿**的大波动。）
    """
    steps = [0.05] * (DEFAULT_WINDOW + 4)
    for position in range(10, len(steps), 15):
        steps[position] = 0.099
    values = walk(10.0, steps)

    assert infer_st_periods(series(values), "sh600000", rules) == ()


def test_a_board_whose_limit_is_already_20_percent_cannot_be_judged(rules):
    """创业板 / 科创板的 ST 限幅与普通股**同为 20%**，价格里没有信号 → 返回空。

    这是「判不了」，不是「判为非 ST」——故这里断言的是返回空元组，而调用方不该把它
    当成「该股非 ST」的正面证据。
    """
    values = walk(10.0, [0.05] * (DEFAULT_WINDOW + 4))

    assert infer_st_periods(series(values), "sz300001", rules) == ()


def test_too_short_a_history_yields_nothing(rules):
    """历史短于一个窗口 → 判不了，返回空（不猜）。"""
    values = walk(10.0, [0.05] * 5)

    assert infer_st_periods(series(values), "sh600000", rules) == ()


def test_a_period_that_ends_well_before_the_data_end_is_closed(rules):
    """ST 结束后（其后有足够长的无证据期）→ 期间有 ``end``，不是延续到末端。"""
    # 先走 40+ 次 5% 板（触发），再走足够长的正常波动期（> 一个窗口）把它隔开
    steps = [0.05] * (DEFAULT_WINDOW + 4) + [0.03] * (DEFAULT_WINDOW * 3)
    values = walk(10.0, steps)

    periods = infer_st_periods(series(values), "sh600000", rules)

    assert periods, "前段应当被判为 ST"
    assert periods[0].end is not None, "它早已结束，不该延续到数据末端"


def test_a_missing_close_column_is_rejected(rules):
    frame = pd.DataFrame({"open": [1.0, 2.0, 3.0]})

    with pytest.raises(ValueError, match="close"):
        infer_st_periods(frame, "sh600000", rules)


def test_the_overlay_does_not_mutate_the_original_table(rules):
    """``with_st_periods`` 返回**新表**，原表不动——回测的可复现性依赖这一点。"""
    before = rules.st_period_count()

    augmented = rules.with_st_periods({"sh600000": [(dt.date(2024, 1, 2), None)]})

    assert rules.st_period_count() == before, "原表不该被改"
    assert augmented.st_period_count() == before + 1
    assert augmented.is_st("sh600000", dt.date(2024, 6, 3)) is True
    assert rules.is_st("sh600000", dt.date(2024, 6, 3)) is False


def test_the_overlay_tightens_the_limit_band(rules):
    """叠加之后 ``limit_for`` 真的**换了一条路**——这才是修好的实质：撮合用的带被收紧。

    本夹具只登记了普通限幅、没登记 ST 限幅，故叠加后查它会**报错**（而不再是静默返回 10%）。
    本项目对这个缺口的立场是「宁漏不错」：查不到 ST 限幅时宁可把问题暴露出来，也不退回
    一个偏宽的普通限幅——后者会放过本不该成交的交易。
    """
    from mbt.rules import RuleTableError

    on = dt.date(2024, 6, 3)
    assert rules.limit_for("sh600000", on) == pytest.approx(0.10)

    augmented = rules.with_st_periods({"sh600000": [(dt.date(2024, 1, 2), None)]})

    with pytest.raises(RuleTableError):
        augmented.limit_for("sh600000", on)
