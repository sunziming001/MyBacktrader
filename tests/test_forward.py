"""估值的前瞻分支：原式的选择规则、成对切换、以及「只走非回溯入口」（票据 #50）。

本模块分三层来测：

1. **纯规则**：``用前瞻``、``PE前瞻``、``增预期`` 三小段原式，边界值逐一钉住（0、负数、
   缺失各是什么意思）。
2. **成对切换**：这是票据特意点名的要求——``选用增`` 必须跟着 ``选用PE`` 一起换。本模块把
   两个值放进同一个 :class:`~mbt.data.forward.Chosen`，故「只换一个」在类型上就写不出来；
   测试再从**数值**上确认两个真的都换了。
3. **入口纪律**：``forward`` 没有 ``as_of`` 之类的日期参数（不给回溯的口子），且
   ``mbt.data.valuation``（回测那条路）**不引用**本模块。这两条是 ADR-0006 修订二在代码里
   的落点，用 ``ast`` 读源文件来钉，免得被文档字符串里的提法骗过。
"""

import ast
from pathlib import Path

import pytest

from mbt.data import Chosen, ForwardReading, choose, reading, readings_frame, uses_forward
from mbt.data.forward import FIELD_EPS_T, FIELD_FISCAL_YEAR, FIELD_NET_PROFIT_T, FIELD_PE_EXPECTED
from mbt.data.fundamental import FinancialRecord
from mbt.data.gpone import GponeDataSource

GPONE_FIXTURE = Path(__file__).parent / "fixtures" / "gpone" / "gpshone.dat"
VALUATION_SOURCE = Path(__file__).parent.parent / "src" / "mbt" / "data" / "valuation.py"


class FakeFinancials:
    """只提供 ``records(symbol)`` 的最小替身——``forward`` 唯一依赖的那一个方法。"""

    def __init__(self, **by_symbol):
        self._by_symbol = {symbol: tuple(records) for symbol, records in by_symbol.items()}

    def records(self, symbol):
        return tuple(sorted(self._by_symbol.get(symbol, ()), key=lambda r: r.report_period))


def annual(period_year, *, net_profit_yuan, announced=None, usable=True):
    """造一条**年末**财报。``net_profit_yuan`` 是全年归母净利（元）。"""
    import datetime as dt

    period = dt.date(period_year, 12, 31)
    stamp = announced or dt.date(period_year + 1, 4, 20)
    if not usable:
        # 公告日退化成「等于报告期」——那是 2005 年之前的占位符形状，usable 判否
        stamp = period
    return FinancialRecord(
        symbol="sh600519",
        report_period=period,
        announcement_date=stamp,
        values={"net_profit_ytd": net_profit_yuan},
    )


@pytest.fixture
def gpone():
    return GponeDataSource(GPONE_FIXTURE.parent)


# --- 一、纯规则 -------------------------------------------------------------


@pytest.mark.parametrize(
    ("fiscal_year", "bar_year", "expected"),
    [
        (2026.0, 2026, True),
        (2026.0, 2025, False),  # 预期落在别的年份
        (2026.0, 2027, False),  # 到了次年，T 还是去年那个
        (0.0, 2026, False),  # 没有一致预期（原式：空值显示为 0）
        (-1.0, 2026, False),
    ],
)
def test_uses_forward_needs_a_positive_fiscal_year_matching_the_bar_year(
    fiscal_year, bar_year, expected
):
    """``用前瞻 := 财年T>0 AND K线年=财年T``——两条缺一不可。"""
    assert uses_forward(fiscal_year, bar_year) is expected


def test_pe_forward_falls_back_to_the_expected_pe_when_eps_is_not_positive():
    """``PE前瞻 := IF(EPS_T>0, C/EPS_T, IF(PE预期>0, PE预期, 0))``。

    三个分支各测一次：``EPS_T`` 为正时用 ``C/EPS_T``（连 ``PE预期`` 都不看）；``EPS_T`` 非正
    但 ``PE预期`` 为正时退回它；两者都不行时取 0。
    """
    usable = ForwardReading("sh600519", 2026.0, 50.0, 1.0, 999.0, 1.0)
    assert usable.pe_forward(close=1000.0) == 20.0

    fallback = ForwardReading("sh600519", 2026.0, 0.0, 1.0, 18.5, 1.0)
    assert fallback.pe_forward(close=1000.0) == 18.5

    nothing = ForwardReading("sh600519", 2026.0, 0.0, 1.0, 0.0, 1.0)
    assert nothing.pe_forward(close=1000.0) == 0.0


def test_growth_expected_uses_zero_when_either_end_is_not_positive():
    """``增预期 := IF(基期净利万>0 AND 净利T>0, (净利T-基期净利万)/基期净利万*100, 0)``。

    注意是取 **0** 而不是缺失——原式如此。拿 0 当「没增长」会偏乐观，故下一组的「成对切换」
    测试专门确认它只在 ``用前瞻`` 成立时才会被选中。
    """
    grows = ForwardReading("sh600519", 2026.0, 1.0, 12000.0, 1.0, 10000.0)
    assert grows.growth_expected == pytest.approx(20.0)

    # 基期为负：取 0，不是「算出个负数」
    negative_base = ForwardReading("sh600519", 2026.0, 1.0, 12000.0, 1.0, -5000.0)
    assert negative_base.growth_expected == 0.0

    # 预测为负：取 0
    negative_forecast = ForwardReading("sh600519", 2026.0, 1.0, -12000.0, 1.0, 10000.0)
    assert negative_forecast.growth_expected == 0.0

    # 拿不到基期年报：取 0
    missing_base = ForwardReading("sh600519", 2026.0, 1.0, 12000.0, 1.0, None)
    assert missing_base.growth_expected == 0.0


# --- 二、成对切换 -----------------------------------------------------------


def test_choose_switches_both_pe_and_growth_to_forward():
    """财年匹配 → **两个**值都换成前瞻口径（而不是只换 PE）。

    期望值刻意取成与历史口径**都不同**的三组数，否则「换没换」看不出来。
    """
    item = ForwardReading(
        "sh600519",
        2026.0,
        eps_t=50.0,
        net_profit_t=12000.0,
        pe_expected=18.5,
        base_net_profit=10000.0,
    )

    chosen = choose(item, close=1000.0, bar_year=2026, trailing_pe=6.0, trailing_growth=-7.5)

    assert chosen == Chosen(pe=20.0, growth=20.0, used_forward=True)
    assert chosen.pe != 6.0, "PE 没换"
    assert chosen.growth != -7.5, "增长率没跟着换"


def test_choose_falls_back_to_both_trailing_values():
    """财年不匹配 → **两个**值都是历史口径，一个都不许用前瞻的。"""
    item = ForwardReading(
        "sh600519",
        2026.0,
        eps_t=50.0,
        net_profit_t=12000.0,
        pe_expected=18.5,
        base_net_profit=10000.0,
    )

    chosen = choose(item, close=1000.0, bar_year=2025, trailing_pe=6.0, trailing_growth=-7.5)

    assert chosen == Chosen(pe=6.0, growth=-7.5, used_forward=False)


def test_choose_falls_back_when_there_is_no_consensus():
    """``财年T = 0``（这只票没有一致预期）→ 退回历史，且**不能**因为 0 而算出 PE 为 0。"""
    item = ForwardReading(
        "sh600690", 0.0, eps_t=0.0, net_profit_t=0.0, pe_expected=0.0, base_net_profit=None
    )

    chosen = choose(item, close=9.1, bar_year=2026, trailing_pe=5.2, trailing_growth=3.3)

    assert chosen == Chosen(pe=5.2, growth=3.3, used_forward=False)


# --- 三、入口纪律 -----------------------------------------------------------


def test_the_forward_entry_points_take_no_date():
    """前瞻入口**没有**日期参数——不是漏了，是不给回溯的口子。

    若哪天有人加了 ``as_of``，这条会红：本地的数据根本不支持按日期取一致预期。
    """
    import inspect

    from mbt.data import readings as readings_function

    banned = {"as_of", "date", "when", "day", "timestamp"}
    for function in (reading, readings_function):
        parameters = set(inspect.signature(function).parameters)
        assert not (parameters & banned), f"{function.__name__} 收了日期参数：{parameters & banned}"


def test_valuation_does_not_reach_for_the_forward_branch():
    """``mbt.data.valuation``（回测那条路）**不引用**本模块。

    用 ``ast`` 读源文件，不看文档字符串——否则正文里提到 ``mbt.data.forward`` 就会误判。
    这条钉的是「最新一致预期不得混进按公告日点取的估值表」。
    """
    tree = ast.parse(VALUATION_SOURCE.read_text(encoding="utf-8"))

    imported: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            imported.add(node.module or "")
            imported.update(alias.name for alias in node.names)

    assert not any("forward" in name for name in imported), f"valuation 引用了前瞻模块：{imported}"


# --- 四、真实 fixture 上走一遍 ----------------------------------------------


def test_reading_reads_the_four_ordinals_from_the_snapshot(gpone):
    """原式那四个字段从真实文件里读出来——茅台：财年 2026、EPS 67.372、净利 842.68 亿、PE 20.014。"""
    item = reading(
        "sh600519",
        gpone=gpone,
        financials=FakeFinancials(sh600519=[annual(2025, net_profit_yuan=86_000_000_000.0)]),
    )

    assert item.fiscal_year == 2026.0
    assert item.eps_t == pytest.approx(67.372, rel=1e-3)
    assert item.net_profit_t == pytest.approx(8_426_775.0, rel=1e-6)
    assert item.pe_expected == pytest.approx(20.014, rel=1e-3)
    # 基期年报 860 亿 = 8,600,000 万元 → 增长 (842.68−860)/860 = −2.01%
    assert item.base_net_profit == pytest.approx(8_600_000.0, rel=1e-6)
    assert item.growth_expected == pytest.approx(-2.014, rel=1e-2)


def test_the_base_annual_profit_ignores_non_annual_reports(gpone):
    """基期净利**只认年末那一期**——``net_profit_ytd`` 是累计数，季度报不能当基期。

    否则增长率会变成「与全年无关的数」：这里季报的 200 亿若被误用，增长率会是个荒谬的大数。
    """
    import datetime as dt

    quarter = FinancialRecord(
        symbol="sh600519",
        report_period=dt.date(2026, 6, 30),
        announcement_date=dt.date(2026, 8, 20),
        values={"net_profit_ytd": 20_000_000_000.0},
    )
    item = reading(
        "sh600519",
        gpone=gpone,
        financials=FakeFinancials(
            sh600519=[annual(2025, net_profit_yuan=86_000_000_000.0), quarter]
        ),
    )

    assert item.base_net_profit == pytest.approx(8_600_000.0, rel=1e-6)


def test_the_base_annual_profit_skips_records_whose_announcement_is_not_trustworthy(gpone):
    """公告日不可信（2005 年之前的占位符）的年末财报**不能**当基期——那是提前数月知道未公布的数。"""
    item = reading(
        "sh600519",
        gpone=gpone,
        financials=FakeFinancials(
            sh600519=[
                annual(2003, net_profit_yuan=1_000_000.0, usable=False),
                annual(2025, net_profit_yuan=86_000_000_000.0),
            ]
        ),
    )

    assert item.base_net_profit == pytest.approx(8_600_000.0, rel=1e-6)


def test_the_snapshot_frame_is_one_row_per_symbol(gpone):
    """摊成表：一行一标的，列就是原式那几项（便于盘后筛一遍）。"""
    from mbt.data import reading as read_one

    items = {
        symbol: read_one(
            symbol,
            gpone=gpone,
            financials=FakeFinancials(**{symbol: [annual(2025, net_profit_yuan=86_000_000_000.0)]}),
        )
        for symbol in ("sh600519", "sh600000")
    }

    frame = readings_frame(items)

    assert list(frame.index) == ["sh600519", "sh600000"]
    assert set(frame.columns) == {
        "fiscal_year",
        "eps_t",
        "net_profit_t",
        "pe_expected",
        "base_net_profit",
        "growth_expected",
    }
    assert frame.loc["sh600519", "fiscal_year"] == 2026.0


def test_the_field_ordinals_are_the_ones_the_formula_names():
    """四个字段号写死在常量里，与帮助原文对齐——改错了会读到另一个字段。"""
    assert (FIELD_FISCAL_YEAR, FIELD_EPS_T, FIELD_NET_PROFIT_T, FIELD_PE_EXPECTED) == (4, 5, 8, 23)
