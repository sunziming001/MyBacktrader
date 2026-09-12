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
from mbt.data import (
    GbbqDataSource,
    SecurityMasterDataSource,
    TdxDataSource,
    backward_adjusted,
    find_anomalies,
)
from mbt.data.tdx import DAY_RECORD_SIZE

ROOT_VARIABLE = "MBT_TDX_ROOT"
GBBQ_VARIABLE = "MBT_TDX_GBBQ"
MASTER_VARIABLE = "MBT_TDX_MASTER"

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


@pytest.fixture
def real_master():
    """证券主表 `base.dbf`。它**不在** `vipdoc` 之内，故单独一个环境变量。"""
    path = _from_environment(MASTER_VARIABLE, "证券主表 base.dbf")
    if not path.is_file():
        pytest.fail(f"{MASTER_VARIABLE}={path} 不是文件")
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


def test_real_share_reform_event_does_not_create_a_fake_jump(real_root, real_gbbq, limit_rules):
    """股改对价送股**不改变总股本**，故不除权——复权不该在它那里插入跳空。

    实测 ``sh600000`` 2006-05-12：gbbq 记有类别 1 的「10送3」，但价格并未按 1.3 稀释——
    原始价 10.86 → 10.21（−5.99%）落在**不稀释**的正常 ±10% 带 ``[9.77, 11.95]`` 内；
    若真稀释，除权参考价应是 8.35。而**未判定就复权**会把这一区间变成 **+22.22%**。

    这条锁定本票的修复（票据 #20）：判定后复权不再插进那个跳空。
    """
    from mbt.data.dilution import NOT_DILUTED, resolve_dilution
    from mbt.rules import RuleTable

    prices = TdxDataSource(real_root).daily("sh600000")
    raw_events = GbbqDataSource(real_gbbq).events("sh600000")
    rules = RuleTable.load(limit_rules)

    before, on = pd.Timestamp("2006-03-20"), pd.Timestamp("2006-05-12")
    if before not in prices.index or on not in prices.index:
        pytest.skip("本机行情未覆盖 2006-03-20 与 2006-05-12")

    # 未判定就复权 → 假跳空。这一条是「修复前确实有这个问题」的现场证据。
    naive = backward_adjusted(prices, raw_events)["close"]
    naive_gap = float(naive.loc[on]) / float(naive.loc[before]) - 1
    assert naive_gap > 0.20, "未判定时没有假跳空——那这条测试就无从判别"

    # 判定后 → 该事件被认出「未稀释」，故不复权。
    effective, verdicts = resolve_dilution(prices, raw_events, "sh600000", rules)
    judged = [v for v in verdicts if v.ex_date == on.date()]
    assert judged and judged[0].verdict == NOT_DILUTED, "该事件应被认出未稀释"

    fixed = backward_adjusted(prices, effective)["close"]
    raw_gap = float(prices["close"].loc[on]) / float(prices["close"].loc[before]) - 1
    fixed_gap = float(fixed.loc[on]) / float(fixed.loc[before]) - 1
    assert fixed_gap == pytest.approx(raw_gap, abs=1e-9), "判定后应回到原始价的真实变动"
    assert raw_gap == pytest.approx(-0.06, abs=0.01)


def test_real_base_dbf_satisfies_the_frame_identity(real_master):
    """**在真实的 3.95 MB 主表上**验帧恒等式。

    fixture 是手工重组的切片（记录数被改写），故它的帧恒等式是**因构造而真**的——这条测试
    才是对「本模块的布局假设在本机真实文件上成立」的正面证据。
    """
    import struct

    raw = real_master.read_bytes()
    count = struct.unpack_from("<I", raw, 4)[0]
    header_length = struct.unpack_from("<H", raw, 8)[0]
    record_length = struct.unpack_from("<H", raw, 10)[0]

    assert raw[0] == 0x03, "dBase III"
    assert raw[-1] == 0x1A, "EOF 标记"
    assert len(raw) == header_length + count * record_length + 1
    assert count > 8_000, "本机主表应有八千量级记录"


def test_real_listing_dates_agree_with_the_first_day_bar(real_master, real_root):
    """**上市日与行情首根逐一吻合**——这是「SSDATE 是真实上市日」最硬的证据。

    对**窗口内上市**（首根晚于 2015-06）的标的，本地 ``.day`` 恰好从上市日开始，故两者相差
    0 天。抽样若干只验证；若哪天不符，说明主表或行情的口径变了。
    """
    import datetime as dt

    from mbt.data import TdxDataSource, instrument_type

    source = TdxDataSource(real_root)
    master = SecurityMasterDataSource(
        real_master,
        symbols=[s for s in source.symbols() if instrument_type(s) == "股票"],
    )
    dates = master.listing_dates()

    checked = 0
    for symbol in sorted(dates):
        if checked >= 30:
            break
        try:
            index = source.daily(symbol).index
        except Exception:  # noqa: BLE001
            continue
        first = index[0].date()
        if first <= dt.date(2015, 6, 1):
            continue  # 窗口前上市的老股，首根会被窗口截断
        assert first == dates[symbol], f"{symbol}: 首根 {first} 与上市日 {dates[symbol]} 不符"
        checked += 1

    assert checked >= 10, f"抽样样本不足（{checked}）"


def test_real_master_has_a_material_share_of_unknown_listing_dates(real_master, real_root):
    """**把「5.2% 的股票没有上市日」这件事钉住**，并确认它们几乎全是已停更的。

    这条的作用是让「回退到行情根数」的**规模**可见：若哪天这个比例暴涨，说明主表变了。
    """
    from mbt.data import TdxDataSource, instrument_type

    source = TdxDataSource(real_root)
    stocks = [s for s in source.symbols() if instrument_type(s) == "股票"]
    known = SecurityMasterDataSource(real_master, symbols=stocks).listing_dates()

    unknown = [s for s in stocks if s not in known]
    assert len(unknown) > 0, "本机确实有一批标的没有上市日"
    ratio = len(unknown) / len(stocks)
    assert 0.01 < ratio < 0.15, f"无上市日的比例是 {ratio:.1%}，与实测的 5.2% 差得太多"

    # 它们应当**几乎全是**已停更的（末根早于全市场最新交易日）。
    latest = source.daily("sh600000").index[-1]
    stale = 0
    for symbol in unknown[:60]:
        try:
            if source.daily(symbol).index[-1] < latest:
                stale += 1
        except Exception:  # noqa: BLE001
            continue
    assert stale >= 55, f"抽样的 {len(unknown[:60])} 只里只有 {stale} 只已停更——与预期不符"


def test_real_dilution_judgement_does_not_bankrupt_the_universe(real_root, real_gbbq):
    """判定不该把大多数标的挡在门外——实测约 79% 的含事件标的判定成功。

    这条是**诚实的护栏**：判据的价值取决于它还能让人用。若哪天改动让通过率崩下去，
    这里会失败，而不是等到跑批时才发现。
    """
    from mbt.data.dilution import DILUTION_THRESHOLD, resolve_dilution
    from mbt.data.errors import MarketDataError
    from mbt.rules import RuleTable, RuleTableError

    source = TdxDataSource(real_root)
    gbbq = GbbqDataSource(real_gbbq)
    # 用**出厂**规则表：夹具表只覆盖 2015 起，会把大量早期事件算成「窗口外」而低估通过率。
    rules = RuleTable.load()

    sample = source.symbols()[::40]  # 约 300 个标的，够看比例
    judged = failed = 0
    for symbol in sample:
        try:
            prices = source.daily(symbol)
            events = gbbq.events(symbol)
        except Exception:  # noqa: BLE001
            continue
        if not any(e.bonus_per_share + e.rights_per_share >= DILUTION_THRESHOLD for e in events):
            continue
        try:
            resolve_dilution(prices, events, symbol, rules)
        except (MarketDataError, RuleTableError):
            failed += 1
        else:
            judged += 1

    total = judged + failed
    if total < 20:
        pytest.skip(f"本机样本中可判事件太少（{total} 个标的），不足以谈通过率")

    assert (
        judged / total >= 0.7
    ), f"判定通过率仅 {judged}/{total} = {judged / total:.0%}——判据正在把太多标的挡在门外"
