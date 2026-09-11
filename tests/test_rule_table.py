"""验证规则表：加载可配置数据文件，并按**成交日**取当日生效的规则。

测试缝 S3：规则表文件路径由调用方注入，测试指向 ``tests/fixtures/rules/`` 下的合成规则表，
因此本文件的测试只验证**查表机制**，不依赖出厂规则表的具体数值。

「按成交日取当日生效的规则」是 ADR-0002 明文要求：数据窗口横跨 2015–2026，
其间规则多次变更，写成常量必然让早年收益系统性偏高。
"""

from datetime import date

import pytest

from mbt.rules import RuleTable, RuleTableError

# --- 按成交日取规则：变更日当天即生效，前一天仍用旧规则 ---


def test_price_limit_switches_on_effective_date(synthetic_rules):
    """变更日**当天**就用新规则，前一天仍用旧规则——边界必须闭合。"""
    table = RuleTable.load(synthetic_rules)

    assert table.price_limit("沪主板", on=date(2023, 5, 31)) == 0.10
    assert table.price_limit("沪主板", on=date(2023, 6, 1)) == 0.12
    assert table.price_limit("沪主板", on=date(2023, 6, 2)) == 0.12


def test_price_limit_lookup_does_not_leak_across_boards(synthetic_rules):
    """板块之间互不串用：沪主板变更不影响创业板自己的生效日期。"""
    table = RuleTable.load(synthetic_rules)

    assert table.price_limit("创业板", on=date(2023, 6, 1)) == 0.20
    assert table.price_limit("创业板", on=date(2020, 8, 21)) == 0.10
    assert table.price_limit("创业板", on=date(2020, 8, 24)) == 0.20


def test_price_limit_before_earliest_rule_raises(synthetic_rules):
    """规则表未覆盖的日期必须报错——缺口纪律：宁可报错，不可默认一个限幅。"""
    table = RuleTable.load(synthetic_rules)

    with pytest.raises(RuleTableError, match="2014-12-31"):
        table.price_limit("沪主板", on=date(2014, 12, 31))


def test_unknown_board_raises(synthetic_rules):
    table = RuleTable.load(synthetic_rules)

    with pytest.raises(RuleTableError, match="不存在的板块"):
        table.price_limit("不存在的板块", on=date(2023, 6, 1))


# --- 费用：同一套查表机制，作用在不同字段上 ---


def test_stamp_duty_follows_effective_date(synthetic_rules):
    """印花税仅卖出方缴纳，故只有一个卖出费率。"""
    table = RuleTable.load(synthetic_rules)

    assert table.stamp_duty_rate(on=date(2023, 8, 25)) == 0.001
    assert table.stamp_duty_rate(on=date(2023, 8, 28)) == 0.0005


def test_transfer_fee_varies_by_board_and_date(synthetic_rules):
    """过户费按板块取值：深主板由「不收」变为「双向收」。"""
    table = RuleTable.load(synthetic_rules)

    assert table.transfer_fee_rate("深主板", on=date(2022, 4, 28)) == 0.0
    assert table.transfer_fee_rate("深主板", on=date(2022, 4, 29)) == 0.00001
    assert table.transfer_fee_rate("沪主板", on=date(2022, 4, 28)) == 0.00002


def test_transfer_fee_reports_whether_it_is_charged(synthetic_rules):
    """「不收」与「费率为 0」在计算上等价，但要能区分，便于解释费用构成。"""
    table = RuleTable.load(synthetic_rules)

    assert table.transfer_fee_sides("深主板", on=date(2022, 4, 28)) == "none"
    assert table.transfer_fee_sides("深主板", on=date(2022, 4, 29)) == "both"


# --- 加载 ---


def test_missing_rule_file_raises(tmp_path):
    with pytest.raises(RuleTableError, match="规则表"):
        RuleTable.load(tmp_path / "not-there.toml")
