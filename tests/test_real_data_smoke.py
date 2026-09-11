"""真实数据的冒烟测试：用本机通达信文件验证格式假设。

标记为 ``realmdata``，且在本机数据缺席时**自动跳过**而非失败——这样换机器后
测试套件仍然全绿，而格式假设（32 字节定长记录、gbbq 的 24+5 分栏）依然被真实文件
验证过。

路径可用环境变量覆盖：``MBT_TDX_ROOT``（行情 ``vipdoc``）、``MBT_TDX_GBBQ``（权息文件）。
"""

import datetime as dt
import os
from pathlib import Path

import backtrader as bt
import pandas as pd
import pytest

from mbt.backtest import run_backtest
from mbt.data import GbbqDataSource, TdxDataSource, backward_adjusted, find_anomalies
from mbt.data.tdx import DAY_RECORD_SIZE

DEFAULT_ROOT = Path(r"D:\Tools\tdx\vipdoc")
DEFAULT_GBBQ = Path(os.environ.get("MBT_TDX_GBBQ", r"D:\Tools\tdx\T0002\hq_cache\gbbq"))

pytestmark = pytest.mark.realmdata


@pytest.fixture
def real_root():
    root = Path(os.environ.get("MBT_TDX_ROOT", DEFAULT_ROOT))
    if not (root / "sh" / "lday").is_dir():
        pytest.skip(f"本机通达信数据不可用：{root}")
    return root


@pytest.fixture
def real_gbbq():
    if not DEFAULT_GBBQ.is_file():
        pytest.skip(f"本机通达信权息文件不可用：{DEFAULT_GBBQ}")
    return DEFAULT_GBBQ


class BuyAndHold(bt.Strategy):
    def next(self):
        if not self.position:
            self.buy()


def test_real_day_file_covers_expected_history(real_root):
    df = TdxDataSource(real_root).daily("sh600000")

    assert len(df) > 2000, "应覆盖多年日线"
    assert df.index.is_monotonic_increasing
    assert (df["close"] > 0).all()


def test_real_day_file_length_is_record_multiple(real_root):
    """真实文件的长度必须能被记录长度整除——这是格式假设的直接验证。"""
    source = TdxDataSource(real_root)

    assert source.path_for("sh600000").stat().st_size % DAY_RECORD_SIZE == 0


def test_backtest_runs_on_real_data(real_root, zero_cost_rules):
    """端到端在真实数据上跑通（结果本身尚不可用于策略判断，见票据 03/04）。"""
    prices = TdxDataSource(real_root).daily("sh600000")

    result = run_backtest(
        prices, symbol="sh600000", strategy=BuyAndHold, cash=100_000.0, rules=zero_cost_rules
    )

    assert len(result.equity_curve) == len(prices)
    assert len(result.trades) == 1
    assert result.final_value > 0


def test_real_gbbq_decrypts_the_whole_table(real_gbbq):
    """整表解密并通过结构自证——密钥表若错一位，这一步就会抛错。

    这里的价值不在于「读到很多条」，而在于 ``199,800`` 级别的真实密文**全部**满足
    market / code / date / category 的结构约束：这是密钥表正确性中不依赖 oracle 的那一半。
    """
    source = GbbqDataSource(real_gbbq)

    assert source.total_records > 190_000, "全市场权息表应有十万量级记录"

    by_date = {event.ex_date: event for event in source.events("sh600000")}
    assert dt.date(2026, 7, 16) in by_date
    assert by_date[dt.date(2026, 7, 16)].cash_per_10 == pytest.approx(4.2, abs=1e-5)


def test_real_backward_adjusted_removes_the_ex_date_gap(real_root, real_gbbq):
    """真实行情 + 真实权息：后复权把除权日的假跳空消除。

    ``sh600000`` 在 2026-07-16 每 10 股派 4.20 元，原始价当日跌约 4.9%；后复权后应只剩
    真实波动。这是本票目标（「除权日不再出现假跳空」）在真实数据上的端到端验证。
    """
    ex_date = pd.Timestamp("2026-07-16")
    prices = TdxDataSource(real_root).daily("sh600000")
    if ex_date not in prices.index or ex_date - pd.Timedelta(days=1) not in prices.index:
        pytest.skip("本机行情未覆盖 2026-07-16 及其前一交易日")

    events = GbbqDataSource(real_gbbq).events("sh600000")
    adjusted = backward_adjusted(prices, events)

    prev_bar = prices.index[prices.index.get_loc(ex_date) - 1]
    raw_gap = prices["close"].loc[ex_date] / prices["close"].loc[prev_bar] - 1
    hfq_gap = adjusted["close"].loc[ex_date] / adjusted["close"].loc[prev_bar] - 1

    assert abs(raw_gap) > 0.04, "原始价在除权日应有明显跳空，否则这条测试无从判别"
    assert abs(hfq_gap) < 0.01, f"后复权后不应有 {hfq_gap:.2%} 的跳空"
    assert adjusted["close"].iloc[0] == prices["close"].iloc[0]


def test_real_ratio_over_the_limit_but_at_the_band_is_not_an_anomaly(real_root, limit_rules):
    """``sh600004`` 2015-07-09：11.26 → 12.39 是 **+10.04%**，但 12.39 正是涨停价。

    这是「判异常必须比限价、不能比比率」在真实数据上的直接证据（研究结论五）：
    ``11.26 × 1.1 = 12.386`` 进位到 12.39，比率必然略超名义限幅。
    """
    from mbt.rules import RuleTable, limit_price

    prices = TdxDataSource(real_root).daily("sh600004")
    on = pd.Timestamp("2015-07-09")
    if on not in prices.index:
        pytest.skip("本机行情未覆盖 2015-07-09")

    prev_close = prices["close"].loc[:on].iloc[-2]
    assert limit_price(prev_close, 0.10, +1) == prices["close"].loc[on], "该日收在涨停价上"
    assert prices["close"].loc[on] / prev_close - 1 > 0.10, "比率确实略超 10%"

    rules = RuleTable.load(limit_rules)
    assert [a for a in find_anomalies(prices, "sh600004", rules) if a.date == on.date()] == []


def test_real_ex_date_gap_needs_the_event_to_be_explained(real_root, real_gbbq, limit_rules):
    """``sh600000`` 的除权跳空：有权息信息时不报，没有时就是坏数据。

    这条锁死的是本模块的核心归因——**公司行为把「越界跳空」解释掉**，而不是见到
    任何超限就报错。``without events: 2`` 也说明若不做归因，真实数据会被误报。
    """
    from mbt.rules import RuleTable

    prices = TdxDataSource(real_root).daily("sh600000")
    events = GbbqDataSource(real_gbbq).events("sh600000")
    rules = RuleTable.load(limit_rules)

    assert find_anomalies(prices, "sh600000", rules, events) == []
    without_events = find_anomalies(prices, "sh600000", rules)
    assert len(without_events) == 2
    assert all(a.kind == "unexplained" for a in without_events)
