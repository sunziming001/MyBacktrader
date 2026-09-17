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
from mbt.data.forward import (
    FIELD_EPS_T,
    FIELD_FISCAL_YEAR,
    FIELD_NET_PROFIT_T,
    FIELD_PE_EXPECTED,
    chosen_valuation,
    forward_available,
)
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


# --- 五、可用性判据：从文件写下那天起才许用（票据 #50 修订） -----------------


def test_forward_is_available_from_the_day_the_file_was_written_onward(gpone):
    """``评估日 ≥ 文件最后写入日`` ⇒ 可用；早一天就不行。

    早于该日的评估日上，我们手上这份内容**可能已经不是**那天的内容——不可知，故不许用。
    这一条把「前瞻只在盘后选股生效」从约定变成了可执行的判据：日常任务收盘后下载、
    随即选股，评估日正好等于写入日。
    """
    import datetime as dt

    written = gpone.updated_on("sh600519")

    assert forward_available("sh600519", gpone=gpone, as_of=written) is True
    assert forward_available("sh600519", gpone=gpone, as_of=written - dt.timedelta(days=1)) is False


def test_forward_is_unavailable_once_the_bar_year_leaves_the_forecast_year(gpone):
    """写入日之后也**不一定**可用：原式还有 ``K线年 = 财年T`` 这条。

    fixture 里一致预期指的财年是 2026，故到了 2027 年就不能再拿它当「今年的预期」——
    那是去年的预测，而原式特意拦住了这一脚。
    """
    import datetime as dt

    written = gpone.updated_on("sh600519")
    after_the_forecast_year = dt.date(written.year + 1, 1, 5)

    assert forward_available("sh600519", gpone=gpone, as_of=after_the_forecast_year) is False


def test_a_symbol_without_a_consensus_falls_back_to_trailing(gpone):
    """``sh600006`` 在文件里、也有别的字段，但**没有一致预期**（财年T = 0）→ 退回历史。

    这正是「有前瞻就用前瞻，没有就用历史」里那个「没有」。
    """
    assert forward_available("sh600006", gpone=gpone, as_of=gpone.updated_on("sh600006")) is False


def test_a_market_without_a_file_cannot_answer_the_question(gpone):
    """文件读不到时 ``forward_available`` **报错**，而不是回答 ``False``。

    「这只票没有一致预期」和「我把目录指错了」是两件事，判据本身分不出来，故把分类留给
    调用方：:func:`chosen_valuation` 会接住它、退回历史，并把读不到的标的**点名列出来**。
    """
    import datetime as dt

    from mbt.data.errors import MarketDataError

    with pytest.raises(MarketDataError):
        forward_available("sz000001", gpone=gpone, as_of=dt.date(2026, 9, 17))


# --- 六、把选择落到表上 -----------------------------------------------------


def _frame(values_by_symbol, columns=("sh600519", "sh600006", "sz000001")):
    """把 ``{标的: [四天的 PE]}`` 摊成日期 × 标的的表；列序**显式**给出，便于断言。"""
    import pandas as pd

    index = pd.bdate_range("2026-09-14", periods=4)
    return pd.DataFrame(values_by_symbol, index=index)[list(columns)]


def _day(gpone):
    """fixture 文件的写入日，**转成 Timestamp**——索引是 DatetimeIndex，用 ``date`` 恒不命中。"""
    import pandas as pd

    return pd.Timestamp(gpone.updated_on("sh600519"))


def test_the_chosen_pe_switches_only_the_evaluation_day(gpone):
    """``选用PE`` 只有**评估日那一行**换成前瞻值，更早的行一格不动。

    期望值手算：茅台一致预期 EPS 是 67.372，取收盘 1010.58 使 ``PE前瞻 = 15``（整数好核对）。
    前三天按历史口径是 10 / 20 / 10，评估日那行则由 10 变成 15。
    """
    trailing = _frame(
        {
            "sh600519": [10.0, 20.0, 10.0, 10.0],
            "sh600006": [5.0, 30.0, 5.0, 30.0],
            "sz000001": [8.0, 8.0, 20.0, 8.0],
        }
    )
    close = _frame(
        {
            "sh600519": [1010.58] * 4,
            "sh600006": [10.0] * 4,
            "sz000001": [10.0] * 4,
        }
    )
    as_of = _day(gpone)

    chosen = chosen_valuation(close, trailing, gpone=gpone, as_of=as_of)

    assert chosen.pe.loc[as_of, "sh600519"] == pytest.approx(15.0, rel=1e-6)
    assert list(chosen.pe["sh600519"][:-1]) == [10.0, 20.0, 10.0], "评估日之前的行不许动"
    assert chosen.forward_symbols == ("sh600519",)


def test_a_symbol_without_a_consensus_keeps_its_trailing_pe(gpone):
    """没一致预期、或连文件都没有的标的 → 逐格保持历史口径，且**点名**记进 ``unreadable``。

    只有茅台那只该走前瞻；另两只一个是「没有预期」，一个是「整个市场没有文件」。
    """
    trailing = _frame(
        {
            "sh600519": [10.0, 20.0, 10.0, 10.0],
            "sh600006": [5.0, 30.0, 5.0, 30.0],
            "sz000001": [8.0, 8.0, 20.0, 8.0],
        }
    )
    close = _frame({symbol: [10.0] * 4 for symbol in trailing.columns})

    chosen = chosen_valuation(close, trailing, gpone=gpone, as_of=_day(gpone))

    assert chosen.pe["sh600006"].tolist() == [5.0, 30.0, 5.0, 30.0]
    assert chosen.pe["sz000001"].tolist() == [8.0, 8.0, 20.0, 8.0]
    assert chosen.forward_symbols == ("sh600519",), "有预期的那只仍该走前瞻"
    assert chosen.unreadable == ("sz000001",), "文件读不到的要点名，否则指错目录看不出来"


def test_a_market_without_the_file_family_is_not_called_unreadable(gpone):
    """北交所是**已知覆盖缺口**，不是「读不到」。两者必须分开计数。

    合在一起会天天报「--forward-root 指对了吗」：实测本机 5,375 个标的里有 276 只北交所，
    每天都会顶上来。**天天喊就等于不喊**——真正指错目录的那天反而淹在噪声里。故这里断言：
    bj 进 ``unsupported``，一个也不进 ``unreadable``。
    """
    trailing = _frame(
        {
            "sh600519": [10.0, 20.0, 10.0, 10.0],
            "bj920001": [7.0, 7.0, 7.0, 7.0],
            "bj430047": [9.0, 9.0, 9.0, 9.0],
        },
        columns=("sh600519", "bj920001", "bj430047"),
    )
    close = _frame({symbol: [10.0] * 4 for symbol in trailing.columns}, columns=trailing.columns)

    chosen = chosen_valuation(close, trailing, gpone=gpone, as_of=_day(gpone))

    assert chosen.unsupported == ("bj920001", "bj430047")
    assert chosen.unreadable == (), "北交所有没有 gpbjone.dat 与目录指错无关，不该混进来"
    assert chosen.pe["bj920001"].tolist() == [7.0, 7.0, 7.0, 7.0], "照历史口径逐格不动"


def test_the_percentile_is_recomputed_from_the_chosen_pe(gpone):
    """百分位必须从 ``选用PE`` **重算**——否则「按前瞻选便宜」只换了分子、没换尺子。

    手算：茅台窗口内 ``选用PE`` 是 10 / 20 / 10 / 15，最低 10、最高 20，故评估日为
    ``(15 − 10) ÷ 10 = 0.5``；若漏掉重算，它会停在历史口径的 ``(10 − 10) ÷ 10 = 0``。
    """
    trailing = _frame(
        {
            "sh600519": [10.0, 20.0, 10.0, 10.0],
            "sh600006": [5.0, 30.0, 5.0, 30.0],
            "sz000001": [8.0, 8.0, 20.0, 8.0],
        }
    )
    close = _frame(
        {
            "sh600519": [1010.58] * 4,
            "sh600006": [10.0] * 4,
            "sz000001": [10.0] * 4,
        }
    )
    as_of = _day(gpone)

    chosen = chosen_valuation(close, trailing, gpone=gpone, as_of=as_of)

    assert chosen.pe_percentile.loc[as_of, "sh600519"] == pytest.approx(0.5, rel=1e-6)
    assert chosen.pe_percentile.loc[as_of, "sh600006"] == pytest.approx(1.0, rel=1e-6)
    assert chosen.pe_percentile.loc[as_of, "sz000001"] == pytest.approx(0.0, abs=1e-9)


def test_an_evaluation_day_before_the_file_was_written_changes_nothing(gpone):
    """把评估日挪到写入日之前，整张表**逐格相同**——那条闸门就是在这里兑现的。

    这正是「回测里前瞻永不生效」的机理：历史 bar 全都早于写入日，故全部退回历史口径。
    """
    import datetime as dt

    trailing = _frame(
        {
            "sh600519": [10.0, 20.0, 10.0, 10.0],
            "sh600006": [5.0, 30.0, 5.0, 30.0],
            "sz000001": [8.0, 8.0, 20.0, 8.0],
        }
    )
    close = _frame({symbol: [1010.58] * 4 for symbol in trailing.columns})
    earlier = _day(gpone) - dt.timedelta(days=3)

    chosen = chosen_valuation(close, trailing, gpone=gpone, as_of=earlier)

    assert chosen.forward_symbols == ()
    assert chosen.pe.equals(trailing), "评估日早于写入日却动了数值"


def test_a_close_frame_that_does_not_line_up_is_rejected(gpone):
    """两份表的标的集合不一致 → 报错，不许按标签各对齐各的（那会静默错位）。"""
    import pandas as pd

    trailing = _frame(
        {
            "sh600519": [10.0, 20.0, 10.0, 10.0],
            "sh600006": [5.0, 30.0, 5.0, 30.0],
            "sz000001": [8.0, 8.0, 20.0, 8.0],
        }
    )
    close = trailing.copy()
    close["sh600999"] = pd.Series([1.0] * 4, index=trailing.index)

    with pytest.raises(ValueError, match="标的"):
        chosen_valuation(close, trailing, gpone=gpone, as_of=_day(gpone))
