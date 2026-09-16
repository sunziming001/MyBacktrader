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


# --- 上市初期不设涨跌幅的天数（price_limit.new_listing） --------------------------


@pytest.fixture
def listing_rules(tmp_path):
    """含「新股上市初期不设涨跌幅」三档的自足规则表。

    数值与出处见 ``docs/research/a-share-trading-rules.md`` 1.3：
    科创板 2019-07-22、创业板 2020-08-24、沪深主板 2023-04-10（全面注册制首批新股上市日）
    起，上市后**前 5 个交易日不设涨跌幅**；北交所 2021-11-15 起为上市**首日**。
    """
    path = tmp_path / "listing.toml"
    path.write_text(
        """
schema_version = 1

[[price_limit]]
board = "沪主板"
effective_from = 2015-01-01
limit = 0.10

[[new_listing_no_limit]]
board = "科创板"
effective_from = 2019-07-22
days = 5

[[new_listing_no_limit]]
board = "沪主板"
effective_from = 2023-04-10
days = 5

[[new_listing_no_limit]]
board = "北交所"
effective_from = 2021-11-15
days = 1
""",
        encoding="utf-8",
    )
    return RuleTable.load(path)


def test_new_listing_window_is_keyed_on_the_listing_date_not_the_bar_date(listing_rules):
    """查表的键是**上市日**，不是成交日——这是它与本表其它参数的根本差别。

    同一个板块、同一个成交日，上市日不同则窗口不同：2023-04-10 上市的有 5 天，
    而 2023-04-09 上市的（仍是老制度）是 0 天。
    """
    assert listing_rules.new_listing_no_limit_days("沪主板", date(2023, 4, 10)) == 5
    assert listing_rules.new_listing_no_limit_days("沪主板", date(2023, 4, 9)) == 0
    assert listing_rules.new_listing_no_limit_days("沪主板", date(2026, 1, 5)) == 5


def test_new_listing_window_does_not_leak_across_boards(listing_rules):
    """板块之间互不串味：科创板 2019-07-22 起的窗口不该被沪主板或北交所的日期影响。"""
    assert listing_rules.new_listing_no_limit_days("科创板", date(2019, 7, 22)) == 5
    assert listing_rules.new_listing_no_limit_days("科创板", date(2019, 7, 21)) == 0
    assert listing_rules.new_listing_no_limit_days("北交所", date(2021, 11, 15)) == 1


def test_new_listing_window_returns_zero_rather_than_raising(listing_rules):
    """**查不到适用条目时返回 0，而不是报错**——这是与 ``price_limit`` 等处的**有意差别**。

    别处「查不到」意味着「我们不知道那天的制度」，故报错（ADR-0005）。这里「查不到」
    本身就是答案：**该上市日没有这条制度**，故不豁免。把「不知道」与「没有」混成同一种
    处置，会让早年的新股全部无法加载。
    """
    # 该板块整段没有条目
    assert listing_rules.new_listing_no_limit_days("创业板", date(2026, 1, 5)) == 0
    # 有该板块，但上市日早于最早的条目
    assert listing_rules.new_listing_no_limit_days("沪主板", date(2001, 6, 1)) == 0
