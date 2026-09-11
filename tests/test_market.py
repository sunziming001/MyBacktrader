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
    """
    market = load_market_data(
        "sh600000", tdx_root=fixture_root, gbbq_path=gbbq_file, rules=LIMIT_RULES
    )

    assert market.verdicts, "判定记录应随行情交出"
    assert len(market.verdicts) == len(market.events), "每条参与复权的事件都应有判定记录"


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
