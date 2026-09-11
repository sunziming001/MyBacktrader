"""验证板块判定与 ST 覆盖。

**板块**回答「适用哪套规则」，与「品种」（这是什么）不同层——CONTEXT.md 中
「品种」的 _Avoid_ 本就列了「板块」，故二者分列。

ST 的必要性在于：ST 股的限幅与普通股不同（主板 5% vs 10%）。若不建模，
撮合会把一根 5% 的涨跌停一字误判为「未到限」，从而允许一笔现实中不可能成交的交易。
而「创业板 ST 仍是 20%」这类例外说明 ST 限幅本身也是**按板块取值的数据**。
"""

from datetime import date

import pytest

from mbt.rules import RuleTable, RuleTableError


def test_board_is_determined_from_market_prefix_and_code(synthetic_rules):
    """市场前缀给出交易所，代码前缀给出该交易所内的分段。"""
    table = RuleTable.load(synthetic_rules)

    assert table.board_of("sh600000") == "沪主板"
    assert table.board_of("sz000609") == "深主板"
    assert table.board_of("sz002415") == "深主板"  # 中小板已并入深主板
    assert table.board_of("sz300750") == "创业板"
    assert table.board_of("sh688981") == "科创板"


def test_non_stock_symbols_have_no_board(synthetic_rules):
    """指数等非股票标的没有板块——报错而不是硬套一个，免得污染规则取值。"""
    table = RuleTable.load(synthetic_rules)

    with pytest.raises(RuleTableError):
        table.board_of("sh000001")  # 上证指数


def test_st_status_is_confined_to_the_registered_period(synthetic_rules):
    table = RuleTable.load(synthetic_rules)

    assert table.is_st("sz000609", on=date(2020, 5, 6))
    assert table.is_st("sz000609", on=date(2021, 4, 30))
    assert not table.is_st("sz000609", on=date(2021, 5, 1))
    assert not table.is_st("sz000609", on=date(2019, 12, 31))


def test_st_period_without_end_runs_on(synthetic_rules):
    """未填 end 表示仍在 ST，不是「未登记」。"""
    table = RuleTable.load(synthetic_rules)

    assert table.is_st("sh600243", on=date(2025, 4, 1))
    assert table.is_st("sh600243", on=date(2030, 1, 1))
    assert not table.is_st("sh600243", on=date(2025, 3, 31))


def test_st_symbol_not_registered_at_all_is_not_st(synthetic_rules):
    """未登记即视为非 ST——这是必须显式声明的局限，不是隐式假设。"""
    table = RuleTable.load(synthetic_rules)

    assert not table.is_st("sh600000", on=date(2024, 1, 1))


def test_st_overrides_the_board_limit(synthetic_rules):
    """主板 ST 的限幅是 5%，而非主板的 10%。"""
    table = RuleTable.load(synthetic_rules)

    assert table.limit_for("sz000609", on=date(2019, 12, 31)) == 0.10
    assert table.limit_for("sz000609", on=date(2020, 5, 6)) == 0.05


def test_st_on_the_creation_board_keeps_the_board_limit(synthetic_rules):
    """例外：创业板 ST 的限幅仍与创业板普通股相同，不是 5%。

    这条把「ST 一律 5%」这个过简假设钉死为错误。
    """
    table = RuleTable.load(synthetic_rules)

    assert table.limit_for("sz300001", on=date(2023, 12, 31)) == 0.20
    assert table.limit_for("sz300001", on=date(2024, 1, 1)) == 0.20


def test_st_without_an_st_rule_for_its_board_raises(synthetic_rules):
    """是 ST 却查不到该板块的 ST 限幅时，报错而不是退回普通限幅。

    静默退回会给出一个偏大的限幅，从而放过本不该成交的交易。
    """
    table = RuleTable.load(synthetic_rules)

    with pytest.raises(RuleTableError, match="科创板"):
        table.limit_for("sh688981", on=date(2024, 1, 1))
