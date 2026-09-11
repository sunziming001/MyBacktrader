"""真实数据的冒烟测试：用本机通达信文件验证格式假设。

标记为 ``realmdata``，且在本机数据缺席时**自动跳过**而非失败——这样换机器后
测试套件仍然全绿，而格式假设（32 字节定长记录、gbbq 的 24+5 分栏）依然被真实文件
验证过。

路径**只从环境变量取**：``MBT_TDX_ROOT``（行情 ``vipdoc``）、``MBT_TDX_GBBQ``（权息文件）。
刻意**不设写死的默认值**——曾经以 ``D:\\Tools\\tdx\\vipdoc`` 作为默认，而本机实际装在
``D:\\Tools\\new_tdx``，于是 7 条真实数据测试**全部静默跳过**，而「跳过」被读成了
「本机没有数据」。一个错误的默认路径比没有默认路径更糟：它让缺失看起来像正常。

因此本模块区分两种情形，且只在其中一种跳过：

- **未设置**环境变量 → 跳过（换机器后套件仍全绿）；
- **设置了但路径不存在** → **失败**。那是路径写错，不是数据不在；跳过会让它一直错下去。
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

ROOT_VARIABLE = "MBT_TDX_ROOT"
GBBQ_VARIABLE = "MBT_TDX_GBBQ"

pytestmark = pytest.mark.realmdata


def _from_environment(variable: str, what: str) -> Path:
    """按环境变量取路径。未设置即跳过；设置了却指错路径则**失败**（见模块说明）。"""
    value = os.environ.get(variable)
    if not value:
        pytest.skip(f"未设置 {variable}（{what}），跳过真实数据测试")

    path = Path(value)
    if not path.exists():
        pytest.fail(
            f"{variable}={path} 不存在。已设置却指错路径时**失败而不跳过**："
            f"跳过会让「路径写错」看起来像「本机没有数据」。"
        )
    return path


@pytest.fixture
def real_root():
    root = _from_environment(ROOT_VARIABLE, "通达信 vipdoc 根目录")
    if not (root / "sh" / "lday").is_dir():
        pytest.fail(f"{ROOT_VARIABLE}={root} 下没有 sh/lday，路径似非 vipdoc 根目录")
    return root


@pytest.fixture
def real_gbbq():
    path = _from_environment(GBBQ_VARIABLE, "权息文件 gbbq")
    if not path.is_file():
        pytest.fail(f"{GBBQ_VARIABLE}={path} 不是文件")
    return path


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

    **切片到 2015 年起**：本机数据现已回溯到 1999-11-10，而这条用例的数字是按
    「出厂规则表与费用口径覆盖 2015 起」的窗口校准的。全历史跑会多出 2006 年的一处
    ``beyond_event``——那是**另一个已知缺陷**（股改对价送股被当作除权除息），
    由下面那条 ``xfail`` 单独记录，不混进本用例。
    """
    from mbt.rules import RuleTable

    prices = TdxDataSource(real_root).daily("sh600000").loc["2015":]
    events = GbbqDataSource(real_gbbq).events("sh600000")
    rules = RuleTable.load(limit_rules)

    assert find_anomalies(prices, "sh600000", rules, events) == []
    without_events = find_anomalies(prices, "sh600000", rules)
    assert len(without_events) == 2
    assert all(a.kind == "unexplained" for a in without_events)


@pytest.mark.xfail(
    strict=True,
    reason="gbbq 把股改对价送股记为类别 1，复权照它稀释，而后复权因此凭空插入约 +22% 的"
    "假跳空。见票据 #20。修好后本用例转为通过——strict=True 会让它在那时提示我们。",
)
def test_real_share_reform_event_does_not_create_a_fake_jump(real_root, real_gbbq):
    """股改对价送股**不改变总股本**，故不除权——复权不该在它那里插入跳空。

    实测 ``sh600000`` 2006-05-12：gbbq 记有类别 1 的「10送3」，但价格并未按 1.3 稀释——
    原始价 10.86 → 10.21（−5.99%）落在**不稀释**的正常 ±10% 带 ``[9.77, 11.95]`` 内；
    若真稀释，除权参考价应是 8.35。而后复权把这一区间变成 **+22.22%**，即凭空造出跳空。

    这条是**既有缺陷的记录**，不是本票要修的：它以 ``xfail(strict=True)`` 挂着，修好后
    会由红转绿并提示改成正向断言（见票据 #20）。
    """
    prices = TdxDataSource(real_root).daily("sh600000")
    events = GbbqDataSource(real_gbbq).events("sh600000")

    before, on = pd.Timestamp("2006-03-20"), pd.Timestamp("2006-05-12")
    if before not in prices.index or on not in prices.index:
        pytest.skip("本机行情未覆盖 2006-03-20 与 2006-05-12")

    raw_gap = float(prices["close"].loc[on]) / float(prices["close"].loc[before]) - 1
    adjusted = backward_adjusted(prices, events)
    adjusted_gap = float(adjusted["close"].loc[on]) / float(adjusted["close"].loc[before]) - 1

    # 原始价是正常波动，而复权后成了大涨——那 27 个百分点就是复权插进去的。
    assert raw_gap == pytest.approx(-0.06, abs=0.01)
    assert abs(adjusted_gap) < 0.02, f"复权插入了 {adjusted_gap:+.2%} 的假跳空"
