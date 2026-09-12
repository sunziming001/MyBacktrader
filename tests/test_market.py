"""数据层入口 :func:`mbt.data.load_market_data` 的测试。

这个入口的全部价值在于**构造即质检**：拿到 :class:`mbt.data.MarketData` 就证明越界
检查跑过了。故测试围绕两件事——检查确实被接上了（坏数据让加载失败），以及检查确实
用到了权息事件（除权跳空不被误报）。
"""

import datetime as dt
import struct

import pytest

from mbt.data import MarketDataError, TdxDataSource, load_market_data
from mbt.rules import RuleTable

#: 沪主板夹具规则表：限幅用真实制度值（10%）、费用为零。
LIMIT_RULES = "tests/fixtures/rules/limit-fixture.toml"


def _day_file(records):
    """把 ``(date, o, h, l, c, amount, volume)`` 编成 ``.day`` 字节（价格 ×100 的整数）。"""
    out = bytearray()
    for date, o, h, low, c, amount, volume in records:
        out += struct.pack("<IIIIIfII", date, o, h, low, c, amount, volume, 0)
    return bytes(out)


def _write(tmp_path, symbol, records):
    """在临时目录搭一个只含 ``symbol`` 的数据源根，返回根路径。"""
    root = tmp_path / "vipdoc"
    target = root / symbol[:2] / "lday"
    target.mkdir(parents=True)
    (target / f"{symbol}.day").write_bytes(_day_file(records))
    return root


def test_load_returns_raw_prices_and_events(fixture_root, gbbq_file):
    """入口返回的是**原始价**（未复权），以及全量权息事件。"""
    market = load_market_data(
        "sh600000", tdx_root=fixture_root, gbbq_path=gbbq_file, rules=LIMIT_RULES
    )

    assert market.symbol == "sh600000"
    assert market.prices.equals(TdxDataSource(fixture_root).daily("sh600000")), "必须是原始价"
    assert market.events, "权息事件应被带上"
    assert market.backward_adjusted().index.equals(market.prices.index)


def test_load_carries_the_dilution_verdicts(fixture_root, gbbq_file):
    """入口把**判定记录**一并交出，且事件已按判定结果处理（票据 #20）。

    这条锁的是「判定接在正门上」——否则调用方拿到的是一份未判定的事件集，
    而复权与质检读到的就不是同一份东西了。

    判定记录的单位是**复牌 bar**，不是单条事件：停牌期间横跨多个除权日时，那一根复牌 K 线
    的价格反映的是它们**叠加**的结果（票据 #45）。本夹具上记录数恰好等于事件数（每条事件
    各占一根 bar），但**那不是契约**——契约是「每条事件都被某个判定覆盖」，合并的情形另有
    专门测试（``tests/test_dilution.py``）。
    """
    market = load_market_data(
        "sh600000", tdx_root=fixture_root, gbbq_path=gbbq_file, rules=LIMIT_RULES
    )

    assert market.verdicts, "判定记录应随行情交出"

    event_dates = sorted(event.ex_date for event in market.events)
    verdict_dates = sorted(verdict.ex_date for verdict in market.verdicts)
    assert verdict_dates == sorted(set(verdict_dates)), "判定记录不应重复"
    assert verdict_dates, "每条事件都该被覆盖，故至少有一条记录"
    assert set(verdict_dates) <= set(event_dates), "判定的日期必须来自事件"
    assert verdict_dates[0] == event_dates[0], "最早的判定应对应最早的事件"
    assert len(verdict_dates) <= len(event_dates), "合并只会减少记录数，不会凭空多出"


def test_load_leaves_a_below_threshold_event_untouched(fixture_root, gbbq_file):
    """夹具里那笔「每 10 股派 4.20 元」是纯现金分红：稀释为零，故判定不介入、事件原样。

    纯现金分红即使误判也只有 1% 量级，判据在那个量级上分不开噪声与真实事件——
    故不判，如实留痕即可。
    """
    from mbt.data.dilution import BELOW_THRESHOLD

    market = load_market_data(
        "sh600000", tdx_root=fixture_root, gbbq_path=gbbq_file, rules=LIMIT_RULES
    )

    verdict = next(v for v in market.verdicts if v.ex_date == dt.date(2026, 7, 16))
    assert verdict.verdict == BELOW_THRESHOLD
    assert verdict.low_confidence is False
    event = next(e for e in market.events if e.ex_date == dt.date(2026, 7, 16))
    assert event.cash_per_10 == pytest.approx(4.2, abs=1e-5)
    assert event.bonus_per_10 == 0.0


def test_load_rejects_a_gap_that_no_event_explains(tmp_path, gbbq_file):
    """10.00 → 20.00 且无权息事件：越界跳空无公司行为可解释，加载必须失败。"""
    root = _write(
        tmp_path,
        "sz000002",  # 权息夹具里没有它的记录，故事件为空
        [
            (20240102, 1000, 1000, 1000, 1000, 1.0, 1000),
            (20240103, 2000, 2000, 2000, 2000, 1.0, 1000),
        ],
    )

    with pytest.raises(MarketDataError, match="越出涨跌停带"):
        load_market_data("sz000002", tdx_root=root, gbbq_path=gbbq_file, rules=LIMIT_RULES)


def test_load_accepts_a_gap_that_the_event_explains(tmp_path, gbbq_file):
    """同一根 K 线，只要权息事件在场就合法——证明入口确实把事件接进了检查。

    sh600000 于 2026-07-16 每 10 股派 4.20 元：前收盘 8.50 的参考价是 8.08，当日限价带
    [7.27, 8.89]。最低 7.50 落在带内，但**若忽略事件**、按 8.50 算带 [7.65, 9.35]，
    7.50 就会越界。故这一例的结果直接取决于事件有没有被用上。
    """
    root = _write(
        tmp_path,
        "sh600000",
        [
            (20260714, 860, 860, 860, 860, 1.0, 1000),
            (20260715, 850, 850, 850, 850, 1.0, 1000),
            (20260716, 805, 810, 750, 780, 1.0, 1000),
            (20260717, 790, 790, 790, 790, 1.0, 1000),
        ],
    )

    market = load_market_data("sh600000", tdx_root=root, gbbq_path=gbbq_file, rules=LIMIT_RULES)

    assert dt.date(2026, 7, 16) in market.prices.index.date
    event = next(e for e in market.events if e.ex_date == dt.date(2026, 7, 16))
    assert event.cash_per_10 == pytest.approx(4.2, abs=1e-5)


def test_load_rejects_a_symbol_the_rules_do_not_cover(tmp_path, gbbq_file):
    """指数没有涨跌停制度，规则表按板块查不到它——报错而不是硬套股票的带。"""
    from mbt.rules.errors import RuleTableError

    root = _write(
        tmp_path,
        "sh000001",
        [
            (20240102, 3000, 3000, 3000, 3000, 1.0, 1000),
            (20240103, 3010, 3010, 3010, 3010, 1.0, 1000),
        ],
    )

    with pytest.raises(RuleTableError):
        load_market_data("sh000001", tdx_root=root, gbbq_path=gbbq_file, rules=LIMIT_RULES)


def test_loaded_data_feeds_the_backtest_end_to_end(fixture_root, gbbq_file):
    """端到端：入口 → 后复权 → 回测。这是「正门」这条路径的冒烟。"""
    import backtrader as bt

    from mbt.backtest import run_backtest

    class BuyAndHold(bt.Strategy):
        def next(self):
            if not self.position:
                self.buy()

    market = load_market_data(
        "sh600000", tdx_root=fixture_root, gbbq_path=gbbq_file, rules=LIMIT_RULES
    )
    rules = RuleTable.load(LIMIT_RULES)
    result = run_backtest(
        market.backward_adjusted(),
        symbol=market.symbol,
        strategy=BuyAndHold,
        cash=100_000.0,
        rules=rules,
    )

    assert len(result.equity_curve) == len(market.prices)
    assert result.final_value > 0


# --- 越界校验只在回测区间内做（票据 #45） --------------------------------------


def test_a_bad_bar_before_the_window_does_not_condemn_the_whole_symbol(tmp_path, gbbq_file):
    """**本票的核心改动**：区间之外的一根坏 K 线不该让整只标的被拒收。

    构造：一根无法解释的十倍跳空（无任何权息事件）之后是一段平稳行情。不传区间时正门必须
    报错（那是 ADR-0005 的纪律）；而把区间设在跳空**之后**时，那根 K 线不在回测里，标的
    就完全可用。

    真实世界的对应物是 ``sh600519``：它的 2006-05-25（股改复牌首日**不设涨跌幅**）越出
    涨跌幅带，而本地数据无从得知那一点。实测全市场抽样里 5.7% 的标的栽在这类「历史早期
    一处的说不清」上，其中九成的坏日子在 2015 之前。
    """
    root = tmp_path / "vipdoc"
    target = root / "sh" / "lday"
    target.mkdir(parents=True)
    # 2024-01-02 收 10.00 → 01-03 收 100.00（十倍跳空，无事件可解释）→ 其后平稳。
    target.joinpath("sh600000.day").write_bytes(
        _day_file(
            [
                (20240102, 1000, 1000, 1000, 1000, 0.0, 1000),
                (20240103, 10000, 10000, 10000, 10000, 0.0, 1000),
                (20240104, 10000, 10000, 10000, 10000, 0.0, 1000),
                (20240105, 10000, 10000, 10000, 10000, 0.0, 1000),
            ]
        )
    )

    with pytest.raises(MarketDataError):
        load_market_data("sh600000", tdx_root=root, gbbq_path=gbbq_file, rules=LIMIT_RULES)

    market = load_market_data(
        "sh600000",
        tdx_root=root,
        gbbq_path=gbbq_file,
        rules=LIMIT_RULES,
        start="2024-01-04",
    )

    assert len(market.prices) == 4, "返回的仍是**完整历史**，切片由调用方负责"
    assert market.prices.index[0].date() == dt.date(2024, 1, 2)


def test_a_bad_bar_inside_the_window_is_still_rejected(tmp_path, gbbq_file):
    """反过来：坏 K 线**落在区间内**时照样报错——区间限制不会变成放水开关。"""
    root = tmp_path / "vipdoc"
    target = root / "sh" / "lday"
    target.mkdir(parents=True)
    target.joinpath("sh600000.day").write_bytes(
        _day_file(
            [
                (20240102, 1000, 1000, 1000, 1000, 0.0, 1000),
                (20240103, 10000, 10000, 10000, 10000, 0.0, 1000),
                (20240104, 10000, 10000, 10000, 10000, 0.0, 1000),
            ]
        )
    )

    with pytest.raises(MarketDataError):
        load_market_data(
            "sh600000",
            tdx_root=root,
            gbbq_path=gbbq_file,
            rules=LIMIT_RULES,
            start="2024-01-03",
        )
