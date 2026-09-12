"""股票池：按**调仓日**的当日状态动态计算（票据 #6）。

股票池是 boolean 标的宽表（`日期 × 标的`，True = 在池，ADR-0009）。选这个形状是因为
ADR-0001 说「行 = 截面」，而「今天哪些可买」正是一行截面；也因此「每个调仓日动态计算」
自然表达为「每一行都算」。调仓日由**策略**决定，引擎只把这张帧交给它。

本文件里最重要的一条是 `test_a_new_stock_enters_only_once_it_has_enough_bars`：它把
**时点正确性**钉在股票池上。用整段行情的长度去判「够不够 60 根」会引入前视偏差——
一只后来涨到 70 根的股票，会在它只有 3 根的时候就获准进入。这不会报错，只会让结论偏乐观
（ADR-0006）。
"""

from __future__ import annotations

import datetime as dt

import pandas as pd
import pytest

from mbt.rules import RuleTable
from mbt.universe import (
    MAIN_BOARD,
    UniverseRules,
    build_universe,
    why_not_in_universe,
)


def closes(n, value=10.0):
    """n 根等值 K 线。等值让「够不够根数」成为唯一的变量。"""
    return [value] * n


def test_only_stocks_enter_the_universe(make_prices, make_market):
    """指数、可转债、基金一律不进池（ADR-0004）——AC 明确要求它们不污染截面统计。"""
    markets = [
        make_market("sh600000", make_prices(closes(80))),  # 股票
        make_market("sh000001", make_prices(closes(80))),  # 指数
        make_market("sh510300", make_prices(closes(80))),  # 基金
        make_market("sh110043", make_prices(closes(80))),  # 可转债
    ]

    got = build_universe(markets, rules=UniverseRules())

    assert bool(got["sh600000"].iloc[-1]) is True
    for symbol in ("sh000001", "sh510300", "sh110043"):
        assert not got[symbol].any(), symbol


def test_default_boards_include_all_four_concepts(make_prices, make_market):
    """默认纳入主板、创业板、科创板、北交所（AC 3）。

    注意「主板」是一个**概念**，规则表内部把它拆成沪主板 / 深主板两行（过户费沪深不同），
    故这里同时验证两个交易所的主板都在内。
    """
    markets = [
        make_market("sh600000", make_prices(closes(80))),  # 沪主板
        make_market("sz000001", make_prices(closes(80))),  # 深主板
        make_market("sz300750", make_prices(closes(80))),  # 创业板
        make_market("sh688981", make_prices(closes(80))),  # 科创板
        make_market("bj920819", make_prices(closes(80))),  # 北交所
    ]

    got = build_universe(markets, rules=UniverseRules())

    for symbol in ("sh600000", "sz000001", "sz300750", "sh688981", "bj920819"):
        assert bool(got[symbol].iloc[-1]) is True, symbol


def test_a_board_can_be_excluded_by_configuration(make_prices, make_market):
    """准入规则可配置（AC 3）：只留主板时，创业板与科创板被排除。"""
    markets = [
        make_market("sh600000", make_prices(closes(80))),
        make_market("sz300750", make_prices(closes(80))),
        make_market("sh688981", make_prices(closes(80))),
    ]

    got = build_universe(markets, rules=UniverseRules(boards=frozenset({MAIN_BOARD})))

    assert bool(got["sh600000"].iloc[-1]) is True
    assert not got["sz300750"].any()
    assert not got["sh688981"].any()


def test_a_new_stock_enters_only_once_it_has_enough_bars(make_prices, make_market):
    """次新股规则，且**不得用整段长度判定**——那会让它在只有 3 根时就获准进入。

    样本共 70 根，``min_bars=60``：第 1..59 根为假，第 60 根起为真。
    若实现写成「整段长度 >= 60」，前 59 行会**全为真**，这条即失败。
    """
    market = make_market("sh600000", make_prices(closes(70)))

    got = build_universe([market], rules=UniverseRules(min_bars=60))["sh600000"]

    assert not got.iloc[:59].any()
    assert bool(got.iloc[59]) is True
    assert got.iloc[59:].all()


def test_min_bars_is_configurable_down_to_zero(make_prices, make_market):
    """把门槛调成 0 即「不排除次新」，从第一根起就在池内（AC 3 要求可配置）。"""
    market = make_market("sh600000", make_prices(closes(3)))

    got = build_universe([market], rules=UniverseRules(min_bars=0))["sh600000"]

    assert got.all()


def test_a_suspended_stock_stays_in_the_universe(make_prices, make_market):
    """停牌不改「交易资格」，只是当日**不可成交**——那是另一张掩码的事。

    `CONTEXT.md`：停牌是「标的仍在交易资格之内，但当日无成交。这是一个正常状态」。
    故股票池照常包含它，撮合由可交易掩码拦下。
    """
    frame = make_prices(closes(5)).drop(pd.Timestamp("2024-01-03"))
    market = make_market("sh600000", frame)
    other = make_market("sz000001", make_prices(closes(5)))

    got = build_universe([market, other], rules=UniverseRules(min_bars=0))

    assert bool(got.loc["2024-01-03", "sh600000"]) is True
    assert bool(got.loc["2024-01-03", "sz000001"]) is True


def test_the_frame_is_the_union_of_dates_by_symbols_and_is_boolean(make_prices, make_market):
    """形状与 dtype 是契约：行 = 各标的日期的并集（与引擎时钟一致），列 = 标的。"""
    a = make_market("sh600000", make_prices(closes(3), start="2024-01-02"))
    b = make_market("sz000001", make_prices(closes(2), start="2024-01-03"))

    got = build_universe([a, b], rules=UniverseRules(min_bars=0))

    assert list(got.columns) == ["sh600000", "sz000001"]
    assert list(got.index) == [pd.Timestamp(d) for d in ("2024-01-02", "2024-01-03", "2024-01-04")]
    assert all(dtype.kind == "b" for dtype in got.dtypes)


def test_the_frames_index_matches_the_union_of_trading_days(make_prices, make_market):
    """并集而非交集——某标的停牌的那天，其它标的照样要能被考虑。"""
    a = make_market("sh600000", make_prices(closes(4), start="2024-01-02"))
    b = make_market("sz000001", make_prices(closes(2), start="2024-01-02"))

    got = build_universe([a, b], rules=UniverseRules(min_bars=0))

    assert len(got.index) == 4
    assert bool(got.loc["2024-01-05", "sh600000"]) is True


def test_a_stock_stays_eligible_after_its_data_stops(make_prices, make_market):
    """行情结束之后仍留在池内，只要根数门槛已满足。

    这是一处**刻意的语义选择**：股票池表达**交易资格**，不是「当日有没有数据」。依据是
    `CONTEXT.md` 把**停牌**定义为「标的仍在交易资格之内，但当日无成交」，而**退市**与之
    的区别只在「不会恢复」——那是**事后**才知道的，逐行计算时无从区分，也不该去猜。

    选这一边的三个理由：

    1. 若改成「当日必须有 K 线」，股票池就与**可交易掩码**几乎重合，而词汇表把这两件事分开；
    2. 持仓的标的若因当日停牌就「掉出池子」，策略便无法分辨「它出池了」与「它今天没成交」
       ——而 AC 要求策略能察觉持仓出池；
    3. 包含它是**无害**的：能不能成交由可交易掩码与撮合层独立把关，池内不等于可成交。

    最后一条的代价是明确的：退市标的也会留在池内，故**消费股票池时必须同时看可交易掩码**。
    """
    a = make_market("sh600000", make_prices(closes(4), start="2024-01-02"))
    b = make_market("sz000001", make_prices(closes(2), start="2024-01-02"))

    got = build_universe([a, b], rules=UniverseRules(min_bars=0))

    assert bool(got.loc["2024-01-04", "sz000001"]) is True
    assert bool(got.loc["2024-01-05", "sz000001"]) is True


def test_a_stock_whose_data_ends_before_the_bar_threshold_stays_out(make_prices, make_market):
    """与上一条的边界：门槛未满足时，行情结束也不会让它「因为不再有数据」而进入。

    累计根数在缺数据的日子不增长，故它停在 2，永远够不到 3。
    """
    b = make_market("sz000001", make_prices(closes(2), start="2024-01-02"))
    a = make_market("sh600000", make_prices(closes(4), start="2024-01-02"))

    got = build_universe([a, b], rules=UniverseRules(min_bars=3))

    assert not got["sz000001"].any()
    assert bool(got.loc["2024-01-04", "sh600000"]) is True


def test_an_empty_market_list_yields_an_empty_frame():
    """没有标的时返回空帧而不是报错——「今天没有合格的标的」是合法结果。"""
    got = build_universe([], rules=UniverseRules())

    assert got.empty


# --- ST 接缝（AC：留出接缝但默认不生效） ---------------------------------------


def test_st_periods_in_the_rule_table_are_honoured(make_prices, make_market, synthetic_rules):
    """接缝是通的：规则表登记了 ST 期间，股票池就按它排除。

    用 ``sh600243``：合成表登记其自 2025-04-01 起为 ST（无结束日）。
    """
    table = RuleTable.load(synthetic_rules)
    frame = make_prices(closes(10), start="2025-03-25")
    market = make_market("sh600243", frame)

    got = build_universe([market], rules=UniverseRules(min_bars=0), rule_table=table)["sh600243"]

    assert bool(got.loc["2025-03-31"]) is True  # ST 生效前一天仍在池内
    assert bool(got.loc["2025-04-01"]) is False  # 生效当天即排除
    assert not got.loc["2025-04-01":].any()


def test_without_a_rule_table_no_st_exclusion_happens(make_prices, make_market):
    """不传规则表时不做 ST 判定——没有可查的期间，就无从判定。

    这条与下一条一起表达「接缝存在但不生效」：机制在，数据不在。
    """
    market = make_market("sh600243", make_prices(closes(10), start="2025-03-25"))

    got = build_universe([market], rules=UniverseRules(min_bars=0))["sh600243"]

    assert got.all()


def test_the_shipped_rule_table_registers_no_st_period_so_nothing_is_ever_excluded(
    make_prices, make_market
):
    """**出厂表**不登记任何 ST 期间，故 ST 排除目前永不生效。

    这不是遗漏，而是一条必须显式登记的局限：本地数据不含股票名称，无法回溯历史 ST 状态
    （ADR-0002 的修订一节）。这条测试把局限**本身**钉住——日后 ST 期间数据到位、有人填了
    出厂表，它会失败，从而提醒更新文档与 README 的表述。
    """
    from mbt.rules import RuleTable

    table = RuleTable.load()
    markets = [
        make_market("sh600243", make_prices(closes(10), start="2025-03-25")),
        make_market("sz000609", make_prices(closes(10), start="2025-03-25")),
    ]

    got = build_universe(markets, rules=UniverseRules(min_bars=0), rule_table=table)

    assert got.all().all(), "出厂表开始登记 ST 期间了——请同步更新 README 与 ADR-0002 的表述"


def test_the_board_mapping_covers_every_board_the_rule_layer_can_report(make_prices, make_market):
    """两套板块命名的漂移会被立刻发现。

    规则层按交易所分列主板（沪主板 / 深主板），词汇表的**板块**只有四个概念名；映射只此一处。
    这条把每个可判定的板块各取一个样本跑一遍——若哪天真加了板块而没更新映射，这里会报错，
    而不是让某只标的悄悄进不了池。
    """
    samples = {
        "沪主板": "sh600000",
        "深主板": "sz000001",
        "创业板": "sz300750",
        "科创板": "sh688981",
        "北交所": "bj920819",
    }
    markets = [make_market(symbol, make_prices(closes(80))) for symbol in samples.values()]

    got = build_universe(markets, rules=UniverseRules())

    for label, symbol in samples.items():
        assert bool(got[symbol].iloc[-1]) is True, label


def test_non_stock_instruments_are_excluded_even_when_the_rule_table_knows_them(
    make_prices, make_market, synthetic_rules
):
    """品种过滤先于一切：``sh688981`` 是**股票**（科创板），故它能进池；
    而指数 ``sh000001`` 即使板块层面无从判定也不该进池。"""
    from mbt.rules import RuleTable

    table = RuleTable.load(synthetic_rules)
    markets = [
        make_market("sh688981", make_prices(closes(10))),
        make_market("sh000001", make_prices(closes(10))),
    ]

    got = build_universe(markets, rules=UniverseRules(min_bars=0), rule_table=table)

    assert not got["sh000001"].any()
    # sh688981 在合成表里自 2024-01-01 起被登记为 ST，故应被排除
    assert not got["sh688981"].any()


# --- 真实上市日（票据 #25） ---------------------------------------------------


def test_a_stock_is_excluded_by_its_real_listing_date_even_with_plenty_of_bars(
    make_prices, make_market
):
    """**本票的核心**：次新股门槛按**真实上市日**算，而不是按本地行情根数。

    构造：某标的在窗口内有 **200 根**（远超门槛 60），但它的**真实上市日**就在窗口开始后不久
    ——故它在窗口前段是次新、应当被排除。按行情根数的近似口径会**放行**它（200 ≫ 60），
    因为那只标的的数据恰好从上市日开始，「数据够多」被误当成「上市很久了」。
    """
    market = make_market("sh600000", make_prices(closes(200)))
    first = market.prices.index[0].date()
    listing = first  # 真实上市日 = 窗口首根（数据恰好从上市日开始，正是要挡的那种）

    got = build_universe(
        [market], rules=UniverseRules(min_trading_days=60), listing_dates={"sh600000": listing}
    )["sh600000"]

    assert not got.iloc[:59].any(), "上市不足 60 个交易日 → 前段应被排除"
    assert bool(got.iloc[59]) is True, "第 60 个交易日起在池内"
    assert got.iloc[59:].all()


def test_the_real_listing_date_and_the_bar_count_agree_for_an_old_stock(make_prices, make_market):
    """对**窗口之前**上市的老股，两个口径结论相同（都不是次新）——故这次改动不会误伤它们。"""
    market = make_market("sh600000", make_prices(closes(200)))
    listing = market.prices.index[0].date() - dt.timedelta(days=365 * 20)

    by_listing = build_universe(
        [market], rules=UniverseRules(min_trading_days=60), listing_dates={"sh600000": listing}
    )["sh600000"]
    by_bars = build_universe([market], rules=UniverseRules(min_trading_days=60))["sh600000"]

    assert by_listing.equals(by_bars)


def test_an_unknown_listing_date_falls_back_to_the_bar_count(make_prices, make_market):
    """**上市日未知 → 回退到「本地行情根数」**，而不是整列排除。

    这是票据 AC 明确要求保留的那条退路。若改成整列排除，会把**任何**主表里查不到的标的
    无声丢掉——而实测本机 304 只没有上市日的股票**全部已停更**（多为退市股），它们的本地
    首根就是上市日，故回退结果对它们本来就准。

    构造：70 根、上市日不在字典里，门槛 60 → 前 59 根不在池、第 60 根起在池。
    """
    market = make_market("sh600000", make_prices(closes(70)))

    got = build_universe([market], rules=UniverseRules(min_trading_days=60), listing_dates={})[
        "sh600000"
    ]

    assert not got.iloc[:59].any(), "回退口径：不足 60 根时不在池"
    assert bool(got.iloc[59]) is True


def test_a_known_date_and_an_unknown_one_are_treated_differently_in_one_call(
    make_prices, make_market
):
    """同一次调用里，**有上市日的**按真实口径、**没有的**按回退口径——两列并存。

    这条钉住「回退是**逐标的**的」，而不是「整批都给了才用真实口径」。
    """
    fresh = make_market("sz000001", make_prices(closes(200)))
    unknown = make_market("sh600000", make_prices(closes(200)))

    got = build_universe(
        [fresh, unknown],
        rules=UniverseRules(min_trading_days=60),
        listing_dates={"sz000001": fresh.prices.index[0].date()},
    )

    assert not got["sz000001"].iloc[:59].any(), "真实口径：窗口内上市 → 前段排除"
    assert got["sh600000"].iloc[59:].all(), "回退口径：200 根远够 60"


def test_listing_dates_absent_falls_back_to_the_bar_count(make_prices, make_market):
    """不给上市日（或没传该字典）→ 回退到「本地行情根数」的近似口径。

    主表只是**细化**规则，不该让库因为缺一份可有可无的 TDX 文件而不可用。
    """
    market = make_market("sh600000", make_prices(closes(70)))

    got = build_universe([market], rules=UniverseRules(min_trading_days=60))["sh600000"]

    assert not got.iloc[:59].any()
    assert bool(got.iloc[59]) is True


def test_a_listing_date_after_the_data_leaves_the_symbol_out(make_prices, make_market):
    """上市日晚于数据末根 → 整列排除（它还没上市，或数据里没有它）。"""
    market = make_market("sh600000", make_prices(closes(200)))
    listing = market.prices.index[-1].date() + dt.timedelta(days=30)

    got = build_universe(
        [market], rules=UniverseRules(min_trading_days=0), listing_dates={"sh600000": listing}
    )["sh600000"]

    assert not got.any()


def test_the_reason_agrees_with_the_mask_when_the_date_is_unknown(make_prices, make_market):
    """上市日未知时，**理由说的口径必须与掩码实际用的口径一致**。

    这条防的是「理由说回退、掩码却在排除」这类自相矛盾——评审正是在这里抓到过一个真缺陷
    （文档承诺回退、实现却整列排除）。
    """
    market = make_market("sh600000", make_prices(closes(10)))
    rules = UniverseRules(min_trading_days=60)

    reason = why_not_in_universe([market], rules=rules, listing_dates={})["sh600000"]
    mask = build_universe([market], rules=rules, listing_dates={})["sh600000"]

    assert not mask.any()
    assert "行情只有" in reason, f"理由应与回退口径一致，而实际是：{reason}"


def test_the_reason_says_which_rule_was_used(make_prices, make_market):
    """「为什么被排除」的措辞要分得清**用的是哪个口径**（票据 #25）。

    旧措辞「行情只有 N 根」在有了真实上市日之后会**误导**——读者会以为规则还是按行情根数算的。
    """
    market = make_market("sh600000", make_prices(closes(10)))
    listing = market.prices.index[0].date()

    with_real = why_not_in_universe(
        [market], rules=UniverseRules(min_trading_days=60), listing_dates={"sh600000": listing}
    )["sh600000"]
    without = why_not_in_universe([market], rules=UniverseRules(min_trading_days=60))["sh600000"]
    unknown = why_not_in_universe(
        [market], rules=UniverseRules(min_trading_days=60), listing_dates={}
    )["sh600000"]

    assert "上市仅" in with_real and "个交易日" in with_real
    assert "行情只有" in without
    assert "上市日未知" in unknown
    assert len({with_real, without, unknown}) == 3, "三种情形的措辞必须互不相同"


# --- `min_bars` 的旧名（改名后的兼容） ----------------------------------------


def test_the_old_name_still_works(make_prices, make_market):
    """`min_bars` 是旧名，仍可用——40 处既有调用点与测试因此不受影响。"""
    rules = UniverseRules(min_bars=3)

    assert rules.threshold == 3
    assert rules.min_trading_days == 3


def test_giving_both_names_the_same_value_is_fine():
    assert UniverseRules(min_trading_days=5, min_bars=5).threshold == 5


def test_giving_both_names_different_values_is_an_error():
    """同时给两个名却是不同的值 → 报错，而不是替人挑一个（那会静默改变口径）。"""
    with pytest.raises(ValueError, match="不能给出不同的值"):
        UniverseRules(min_trading_days=5, min_bars=10)


# --- 附加过滤（基本面）：与结构性准入**取与**，失败原因分得开（票据 #8） --------


def test_an_extra_mask_is_intersected_with_the_structural_rules(make_prices, make_market):
    """基本面过滤与品种/板块/次新**取与**，不是替代。"""
    markets = [
        make_market("sh600000", make_prices(closes(80))),
        make_market("sz000001", make_prices(closes(80))),
    ]
    prices = pd.DataFrame({market.symbol: market.prices["close"] for market in markets})
    extra = pd.DataFrame(True, index=prices.index, columns=prices.columns)
    extra["sz000001"] = False  # 它「亏损」

    got = build_universe(markets, rules=UniverseRules(min_bars=0, extra_mask=extra))

    assert got["sh600000"].all()
    assert not got["sz000001"].any()


def test_a_symbol_absent_from_the_extra_mask_is_excluded_not_passed(make_prices, make_market):
    """掩码里没有这个标的 → 它**没被评估过**，故不放行。

    与「没有可用财报就排除」同一方向：入场过滤宁可少收，不可把没评估过的当合格。
    """
    markets = [make_market("sh600000", make_prices(closes(80)))]
    prices = pd.DataFrame({market.symbol: market.prices["close"] for market in markets})
    extra = pd.DataFrame(False, index=prices.index, columns=["sh601999"])  # 另一只

    got = build_universe(markets, rules=UniverseRules(min_bars=0, extra_mask=extra))

    assert not got["sh600000"].any()


def test_without_an_extra_mask_nothing_changes(make_prices, make_market):
    """不给附加掩码时行为与从前一致——既有调用方不受影响。"""
    markets = [make_market("sh600000", make_prices(closes(80)))]

    got = build_universe(markets, rules=UniverseRules(min_bars=0))

    assert got.all().all()


def test_why_not_in_universe_separates_fundamental_from_structural_reasons(
    make_prices, make_market
):
    """「为什么不在池内」要分得清是**结构性**原因还是**基本面**原因。

    这正是把基本面过滤做成 `extra_mask` 而不是塞进品种/板块那几条的理由——排查时最需要的
    就是这个区别。
    """
    from mbt.universe import why_not_in_universe

    markets = [
        make_market("sh600000", make_prices(closes(80))),
        make_market("sz000001", make_prices(closes(80))),
        make_market("sz000002", make_prices(closes(10))),  # 次新（不足 60 根）
    ]
    prices = pd.DataFrame({market.symbol: market.prices["close"] for market in markets})
    extra = pd.DataFrame(True, index=prices.index, columns=prices.columns)
    extra["sz000001"] = False

    reasons = why_not_in_universe(markets, rules=UniverseRules(min_bars=60, extra_mask=extra))

    assert reasons["sh600000"] == "在池内"
    assert "非亏损" in reasons["sz000001"]
    assert "不足" in reasons["sz000002"]
    assert reasons["sz000001"] != reasons["sz000002"], "两类原因必须分得开"


# --- combine_masks ------------------------------------------------------------


def test_combine_masks_ands_the_frames(make_prices, make_market):
    from mbt.universe import combine_masks

    markets = [make_market("sh600000", make_prices(closes(3)))]
    prices = pd.DataFrame({market.symbol: market.prices["close"] for market in markets})
    pool = pd.DataFrame(True, index=prices.index, columns=prices.columns)
    fundamental = pd.DataFrame(True, index=prices.index, columns=prices.columns)
    fundamental.iloc[0] = False

    got = combine_masks(pool, fundamental)

    assert got.iloc[0].eq(False).all()
    assert got.iloc[1:].all().all()


def test_combine_masks_rejects_misaligned_frames(make_prices, make_market):
    """形状不一致要报错并指出是哪一份——`&` 会静默按标签对齐，得到一张看着正常的表。"""
    from mbt.universe import combine_masks

    markets = [make_market("sh600000", make_prices(closes(3)))]
    prices = pd.DataFrame({market.symbol: market.prices["close"] for market in markets})
    good = pd.DataFrame(True, index=prices.index, columns=prices.columns)
    bad = pd.DataFrame(True, index=prices.index[:2], columns=prices.columns)

    with pytest.raises(ValueError, match="第 2 份掩码"):
        combine_masks(good, bad)


def test_combine_masks_needs_at_least_one(make_prices):
    from mbt.universe import combine_masks

    with pytest.raises(ValueError, match="至少要给一份"):
        combine_masks()
