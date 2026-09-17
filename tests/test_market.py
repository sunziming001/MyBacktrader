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


def test_backward_adjusted_markets_is_the_view_screening_runs_on(fixture_root, gbbq_file):
    """一批行情的**后复权**视图——筛选、股票池与撮合都该落在它上面（票据 #73）。

    返回的是**同一个标的的另一条价格序列**，不是一份新数据：`symbol` / 判定记录 / ST 期间
    照旧，只有价格换成后复权，且 `events` 清空（它们已经折进价格，留着会被二次折算，见
    :class:`mbt.data.MarketData` 的说明）。

    夹具上 `sh600000` 在 2026-07-16 每 10 股派 4.20 元：原始价当日从 9.31 跌到 8.85
    （一个 −4.94% 的假跳空），而后复权后应当只剩真实涨跌——按除权因子
    ``前收盘 ÷ (前收盘 − 每股现金)`` 折算，当日应落在 ``8.85 × 9.31 ÷ 8.89 ≈ 9.268``。
    """
    from mbt.data import backward_adjusted_markets

    market = load_market_data(
        "sh600000", tdx_root=fixture_root, gbbq_path=gbbq_file, rules=LIMIT_RULES
    )
    ex_date = "2026-07-16"

    adjusted = backward_adjusted_markets([market])

    assert len(adjusted) == 1
    view = adjusted[0]
    assert view.symbol == market.symbol
    assert view.events == (), "事件已折进价格，留着会被再折算一次"
    assert view.st_periods == market.st_periods, "ST 期间是标的自己的事，与价格尺度无关"
    assert not view.prices.equals(market.prices), "返回的必须是后复权价，不是原始价"
    # 假跳空被抹平：原始价当日跌了 4.94%，后复权后回到除权前的水平附近。
    assert market.prices["close"].loc[ex_date] == pytest.approx(8.85)
    assert view.prices["close"].loc[ex_date] == pytest.approx(8.85 * 9.31 / 8.89, rel=1e-5)
    assert view.prices["close"].loc[ex_date] > market.prices["close"].loc[ex_date]
    # 原对象不被就地改动（视图是新的，行情是共享的）。
    assert market.events, "原始行情上的权息事件不该被清掉"
    assert market.prices["close"].loc[ex_date] == pytest.approx(8.85)


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


# --- quality_bars：越界检查只看窗口末端 N 根（ADR-0014）--------------------------


def _spike_then_calm(root, *, tail=3):
    """造一只「开头一根无法解释的十倍跳空、其后平稳」的标的，返回 ``root``。

    跳空在 **2024-01-03**（第 2 根），其后 ``tail`` 根平稳——故「最后 N 根」不含跳空，
    而「全部历史」含。这正是 ``sh600547`` 的形态（越界在十几年前，选股只看最近）。
    """
    target = root / "sh" / "lday"
    target.mkdir(parents=True)
    records = [
        (20240102, 1000, 1000, 1000, 1000, 0.0, 1000),
        (20240103, 10000, 10000, 10000, 10000, 0.0, 1000),
    ]
    day = 4
    for _ in range(tail):
        records.append((20240100 + day, 10000, 10000, 10000, 10000, 0.0, 1000))
        day += 1
    target.joinpath("sh600000.day").write_bytes(_day_file(records))
    return root


def test_quality_bars_narrows_the_check_to_the_tail(tmp_path, gbbq_file):
    """``quality_bars`` 把越界检查收窄到最后 N 根，于是**远古那根跳空不再拒收整只标的**。

    构造：2024-01-03 一根十倍跳空（无事件可解释），其后 3 根平稳。不给 ``quality_bars``
    时必须报错（ADR-0005 的纪律）；给 ``quality_bars=2`` 时检查的是最后 2 根（另往前借一根
    作限价带基数，见 ``market._window_of`` 同一手法），跳空不在其中，标的就可用——而返回的
    ``prices`` 仍是**完整历史**（切不切由调用方定）。

    真实世界的对应物是 ``sh600547``：2006-03-31 的除权判定让它被拒，而它在 2026-09-15
    过得了全部 8 道 B1 门。全市场实测这一类多拒了 990 只（16.8%）。
    """
    root = _spike_then_calm(tmp_path / "vipdoc")

    with pytest.raises(MarketDataError):
        load_market_data("sh600000", tdx_root=root, gbbq_path=gbbq_file, rules=LIMIT_RULES)

    market = load_market_data(
        "sh600000",
        tdx_root=root,
        gbbq_path=gbbq_file,
        rules=LIMIT_RULES,
        quality_bars=2,
    )

    assert len(market.prices) == 5, "返回的仍是**完整历史**，切片由调用方负责"
    assert market.prices.index[0].date() == dt.date(2024, 1, 2)


def test_quality_bars_still_rejects_a_bad_bar_inside_the_tail(tmp_path, gbbq_file):
    """反过来：跳空**落在那 N 根之内**时照样报错——它不是放水开关。

    同一只标的，``quality_bars=4`` 时「最近 4 根」是 [01-03 … 01-06]，跳空那根**在其中**
    （它前面还有一根作基数），必须报错。这条与上一条成对：上一条证明收窄生效，这一条证明
    收窄只按位置、不看别的。
    """
    root = _spike_then_calm(tmp_path / "vipdoc")

    with pytest.raises(MarketDataError):
        load_market_data(
            "sh600000",
            tdx_root=root,
            gbbq_path=gbbq_file,
            rules=LIMIT_RULES,
            quality_bars=4,
        )


def test_quality_bars_anchors_on_the_end_not_the_data_end(tmp_path, gbbq_file):
    """给了 ``end`` 时，那 N 根是**相对 ``end``** 的末端，不是相对数据末端。

    这一条是选股能用的关键：``mbt screen`` 已经把 ``end=--as-of`` 传给取数，于是
    ``quality_bars`` 自动变成「评估日往前 N 根」，不需要它去反推日期。

    构造：跳空在 01-03，数据到 01-06 共 5 根。``end="2024-01-04"`` + ``quality_bars=2``
    时窗口是 [01-03, 01-04]，跳空**在**其中 → 报错；而同一组参数去掉 ``end``
    时窗口是 [01-05, 01-06] → 通过。两者只差 ``end``。
    """
    root = _spike_then_calm(tmp_path / "vipdoc")

    with pytest.raises(MarketDataError):
        load_market_data(
            "sh600000",
            tdx_root=root,
            gbbq_path=gbbq_file,
            rules=LIMIT_RULES,
            end="2024-01-04",
            quality_bars=2,
        )

    market = load_market_data(
        "sh600000",
        tdx_root=root,
        gbbq_path=gbbq_file,
        rules=LIMIT_RULES,
        quality_bars=2,
    )
    assert len(market.prices) == 5


def test_quality_bars_must_be_positive_or_none(tmp_path, gbbq_file):
    """``0`` 与负数不是「更宽」而是无意义：``0`` 会让窗口空掉，负数会从头部切。

    调用方要「不限制」就传 ``None``（CLI 那边把 ``--anomaly-bars 0`` 换成 ``None``）。
    """
    root = _spike_then_calm(tmp_path / "vipdoc")

    for bad in (0, -1):
        with pytest.raises(ValueError, match="至少为 1"):
            load_market_data(
                "sh600000",
                tdx_root=root,
                gbbq_path=gbbq_file,
                rules=LIMIT_RULES,
                quality_bars=bad,
            )


def test_quality_bars_also_narrows_the_dilution_judgment(tmp_path, gbbq_file):
    """收窄的是**整段质检窗口**，故稀释判定也一起被收窄——这才是 ``sh600547`` 的成因。

    先做「只收窄越界检查」是错的：实测 ``sh600547`` **仍被拒**，因为拦它的是
    :func:`~mbt.data.dilution.resolve_dilution` 的稀释判定（消息是「除权事件判不动」），
    而那一处吃的是同一个 ``window``。故本测试盯住那条路径。

    构造用夹具里**真实存在**的事件：``sh600000`` 于 **2006-05-12** 每 10 股送 3 股
    （稀释 0.30 ≥ 阈值 0.25）。前收 10.00 时两个假说各给一个带：

    - 未稀释：带 [9.00, 11.00]
    - 已稀释：参考价 10.00 ÷ 1.30 = 7.69，带 [6.92, 8.46]

    当日收 8.50——**两个带都落不进**，故判定无从做出。不传 ``quality_bars`` 时它报错
    （这正是 ``sh600547`` 收到的消息）；传 ``quality_bars=3`` 时窗口收窄到最近 4 根
    （2026 年那几根），那条 2006 的事件落在序列之外、**不再判定**，标的就可用。
    """
    root = _write(
        tmp_path,
        "sh600000",
        [
            (20060511, 1000, 1000, 1000, 1000, 1.0, 1000),
            (20060512, 850, 850, 850, 850, 1.0, 1000),  # 事件日，8.50 两个带都落不进
            (20260909, 1000, 1000, 1000, 1000, 1.0, 1000),
            (20260910, 1000, 1000, 1000, 1000, 1.0, 1000),
            (20260911, 1000, 1000, 1000, 1000, 1.0, 1000),
            (20260914, 1000, 1000, 1000, 1000, 1.0, 1000),
            (20260915, 1000, 1000, 1000, 1000, 1.0, 1000),
        ],
    )

    with pytest.raises(MarketDataError, match="两个假说的带都落不进"):
        load_market_data("sh600000", tdx_root=root, gbbq_path=gbbq_file, rules=LIMIT_RULES)

    market = load_market_data(
        "sh600000",
        tdx_root=root,
        gbbq_path=gbbq_file,
        rules=LIMIT_RULES,
        quality_bars=3,
    )

    assert len(market.prices) == 7, "返回的仍是**完整历史**"
    assert dt.date(2006, 5, 12) in market.prices.index.date
