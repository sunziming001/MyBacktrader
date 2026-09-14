"""选股规则：过滤器 + 排序 + 取前 N（票据 #7）。

`CONTEXT.md` 里**选股规则**产出一个候选标的集，回答「今天买哪些」，**不回答「买多少」**——
后者的边界在本文件里也是测试对象：规则不知道你持有什么。

最重要的一条是 `test_the_result_is_causal_so_it_can_be_computed_in_one_pass`。选股只在**一行
之内**比较（排序取前 N 不跨行），信号又只回看，故「整张帧一次算完」与「逐日各算一次」结果
必然相同。这条性质让引擎可以算一次、按日期取行，而不必逐 tick 重跑整条链——但它必须被钉住，
否则哪天有人写出跨行比较（例如「与昨日入选者比较」），一次算完就会静默给出逐日计算不会有的
结果。
"""

from __future__ import annotations

import backtrader as bt
import pandas as pd
import pytest

from mbt.data import MarketData
from mbt.screen import Screen
from mbt.signals import above_ma, momentum, new_high


class BuyAll(bt.Strategy):
    """第一根 K 线后，对每个标的各下一张买单。"""

    def next(self):
        if len(self) == 1:
            for data in self.datas:
                self.buy(data=data)


def flat_bars(n, value, start="2024-01-02"):
    """n 根等值 K 线——组合集成测试用，价与量全为整数便于手算。"""
    return pd.DataFrame(
        {
            "open": [value] * n,
            "high": [value] * n,
            "low": [value] * n,
            "close": [value] * n,
            "volume": [1000] * n,
        },
        index=pd.bdate_range(start, periods=n),
    )


CLOSES = {
    "sh600000": [10.0, 20.0, 30.0, 40.0],
    "sz000001": [10.0, 10.0, 10.0, 10.0],
    "sz000002": [40.0, 30.0, 20.0, 10.0],
}


def price_panel(panel, closes=None):
    """只填价格字段的面板；open/high/low 与 close 同值，便于手算。"""
    closes = closes or CLOSES
    volumes = {symbol: [1000.0] * len(values) for symbol, values in closes.items()}
    return panel(
        {"open": closes, "high": closes, "low": closes, "close": closes, "volume": volumes}
    )


def window(frame, rows):
    """取前 rows 行——用于「截断后重算」的一致性检查。"""
    return frame.iloc[:rows]


# --- 三个部件可独立配置（AC 1） ------------------------------------------------


def test_filters_alone_produce_a_boolean_frame(panel):
    """只给过滤器：候选集即过滤器之 AND。"""
    screen = Screen(filters=(lambda p: above_ma(p["close"], 2),))

    got = screen.apply(price_panel(panel)).selected

    assert got.dtypes.map(lambda dtype: dtype.kind == "b").all()
    # above_ma(close,2)：A 自第 2 行起为真；B 恰好等于均线（不算站上）；C 一路下行
    assert got["sh600000"].tolist() == [False, True, True, True]
    assert not got["sz000001"].any()
    assert not got["sz000002"].any()


def test_several_filters_are_combined_with_and(panel):
    """多个过滤器是**与**关系，不是或。"""
    always = lambda p: p["close"] > 5  # noqa: E731
    never = lambda p: p["close"] > 100  # noqa: E731

    both = Screen(filters=(always, never)).apply(price_panel(panel)).selected
    only_first = Screen(filters=(always,)).apply(price_panel(panel)).selected

    assert not both.any().any()
    assert only_first.all().all()


def test_an_empty_screen_selects_the_whole_universe(panel):
    """什么都不给是合法退化：等于「没有意见」，全收。

    这让「选股规则」可以作为回测的唯一闸门而无需特判，也说明三个部件确实各自可选。
    """
    got = Screen().apply(price_panel(panel)).selected

    assert got.all().all()


# --- 排序与取前 N --------------------------------------------------------------


def test_top_n_keeps_the_best_per_day_by_the_factor(panel):
    """取前 N 在**每一行内**比较：三只标的里每天只留动量最大的那只。"""
    screen = Screen(factor=lambda p: momentum(p["close"], 1), top_n=1)

    got = screen.apply(price_panel(panel)).selected

    # 第 1 行动量全为缺失（没有前一日）→ 全被排除；其后 A 的动量始终最大
    assert got.iloc[0].tolist() == [False, False, False]
    assert got.iloc[1:]["sh600000"].all()
    assert not got.iloc[1:][["sz000001", "sz000002"]].any().any()


def test_top_n_picks_the_best_among_those_that_passed_the_filters(panel):
    """被过滤掉的标的**不占名次**。

    它的分数最高（A 的动量 1.5 大于 B 的 0.2），但 A 不满足「收盘低于 20」。若不过滤就可能
    让 A 占掉第 1 名，从而把本该入选的 B 挤出去——那时结果会是空集，而这显然不对。
    """
    closes = {"sh600000": [10.0, 25.0, 30.0], "sz000001": [10.0, 12.0, 14.0]}
    screen = Screen(
        filters=(lambda p: p["close"] < 20,),
        factor=lambda p: momentum(p["close"], 1),
        top_n=1,
    )

    got = screen.apply(price_panel(panel, closes)).selected

    assert bool(got.loc[got.index[1], "sz000001"]) is True, "被过滤掉的高分标的占掉了名次"
    assert not got["sh600000"].any()


def test_ties_are_broken_by_symbol_ascending_so_results_repeat(panel):
    """同分按代码升序——否则同一天两次运行可能给出不同清单。"""
    flat = lambda p: p["close"] * 0.0  # noqa: E731  全为 0，故必然同分
    screen = Screen(factor=flat, top_n=1)

    got = screen.apply(price_panel(panel)).selected

    assert [s for s in got.columns if got.iloc[0][s]] == ["sh600000"]


def test_a_missing_factor_value_excludes_the_symbol(panel):
    """因子缺失即排除，不因为「不知道」而被当成合格。"""
    closes = {"sh600000": [10.0, 11.0, 12.0], "sz000001": [10.0, float("nan"), 14.0]}
    screen = Screen(factor=lambda p: momentum(p["close"], 1))

    got = screen.apply(price_panel(panel, closes)).selected

    assert bool(got.loc[got.index[1], "sz000001"]) is False
    assert bool(got.loc[got.index[1], "sh600000"]) is True


def test_top_n_grows_the_candidate_set_as_requested(panel):
    """top_n 就是「取几个」：给 2 就留下前两名。"""
    screen = Screen(factor=lambda p: momentum(p["close"], 1), top_n=2)

    got = screen.apply(price_panel(panel)).selected

    assert sum(got.iloc[1]) == 2
    assert bool(got.iloc[1]["sh600000"]) is True


# --- 组合不成立时报错（AC 1 的边界） -------------------------------------------


def test_a_top_n_without_a_factor_is_rejected(panel):
    """没有排序就无从谈「前 N」。报错而不替你按代码顺序取前 N——那是另一个语义。"""
    with pytest.raises(ValueError, match="就必须有排序因子"):
        Screen(top_n=3).apply(price_panel(panel))


def test_a_non_positive_top_n_is_rejected(panel):
    with pytest.raises(ValueError, match="至少为 1"):
        Screen(factor=lambda p: momentum(p["close"], 1), top_n=0).apply(price_panel(panel))


def test_a_filter_returning_numbers_is_rejected(panel):
    """过滤器必须回答是/否；返回连续数值是「排序因子」用错了地方。"""
    with pytest.raises(ValueError, match="不是布尔值"):
        Screen(filters=(lambda p: momentum(p["close"], 1),)).apply(price_panel(panel))


def test_a_factor_returning_booleans_is_rejected(panel):
    with pytest.raises(ValueError, match="不是浮点数"):
        Screen(factor=lambda p: above_ma(p["close"], 2)).apply(price_panel(panel))


def test_a_misaligned_filter_result_is_rejected(panel):
    """形状错位必须报错——`&` 会静默按列名对齐，得到一张看着正常的结果表。"""
    only_one = lambda p: above_ma(p["close"], 2)[["sh600000"]]  # noqa: E731

    with pytest.raises(ValueError, match="必须与面板一致"):
        Screen(filters=(only_one,)).apply(price_panel(panel))


# --- 股票池叠加（AC 4） --------------------------------------------------------


def test_the_universe_mask_is_intersected(panel):
    """股票池规则在选股时同样生效：池外的即使满足条件也不入选。"""
    prices = price_panel(panel)
    pool = pd.DataFrame(False, index=prices["close"].index, columns=prices["close"].columns)
    pool["sh600000"] = True

    screen = Screen(filters=(lambda p: p["close"] > 5,))

    without = screen.apply(prices).selected
    with_pool = screen.apply(prices, universe_mask=pool).selected

    assert without.all().all()
    assert with_pool["sh600000"].all()
    assert not with_pool[["sz000001", "sz000002"]].any().any()


def test_the_universe_mask_is_truncated_along_with_the_panel(panel):
    """给了 ``as_of`` 时掩码也跟着截断——否则形状必然对不上（面板截了、掩码没截）。

    这条是接 CLI 时暴露出来的：``apply(as_of=…, universe_mask=…)`` 原先会以「形状不一致」
    报错，而任何调用方都会撞上它，不只是 CLI。
    """
    prices = price_panel(panel)
    pool = pd.DataFrame(True, index=prices["close"].index, columns=prices["close"].columns)

    result = Screen(filters=(lambda p: p["close"] > 5,)).apply(
        prices, as_of=prices["close"].index[1], universe_mask=pool
    )

    assert len(result.selected) == 2, "结果应只到评估日为止"
    assert result.selected.all().all(), "掩码全是 True，故交集就是选股结果本身"


def test_a_misaligned_universe_mask_is_rejected(panel):
    """标的集合对不上仍要报错——截断只处理日期，不处理标的。"""
    prices = price_panel(panel)
    pool = pd.DataFrame(True, index=prices["close"].index, columns=["sh600000"])

    with pytest.raises(ValueError, match="必须与面板一致"):
        Screen().apply(prices, universe_mask=pool)


def test_a_mask_shorter_than_the_panel_is_still_rejected(panel):
    """掩码比面板短（且没有 as_of 可截）时仍要报错——那是真的对不齐，不该被静默填上。"""
    prices = price_panel(panel)
    pool = pd.DataFrame(True, index=prices["close"].index[:2], columns=prices["close"].columns)

    with pytest.raises(ValueError, match="必须与面板一致"):
        Screen().apply(prices, universe_mask=pool)


# --- 时点正确性（AC 5） --------------------------------------------------------


def test_the_result_is_causal_so_it_can_be_computed_in_one_pass(panel):
    """截断重算不变性：算到第 k 日的结果，与算完全部再取前 k 行，**逐值相同**。

    这条保证「一次算完」与「逐日各算一次」等价——引擎因此不必逐 tick 重跑整条链。
    """
    closes = {
        "sh600000": [10.0, 12.0, 11.0, 15.0],
        "sz000001": [20.0, 19.0, 22.0, 21.0],
        "sz000002": [5.0, 6.0, 5.5, 4.0],
    }
    screen = Screen(
        filters=(lambda p: p["close"] > 5, lambda p: above_ma(p["close"], 2)),
        factor=lambda p: momentum(p["close"], 1),
        top_n=1,
    )
    full = screen.apply(price_panel(panel, closes)).selected

    # 非空转：结果里得有真值，否则「两边全 False」也会通过。
    assert full.to_numpy().sum() > 0, "这条测试没有实际选中任何标的，判据会空转"

    for rows in range(1, len(full) + 1):
        truncated = price_panel(panel, {k: v[:rows] for k, v in closes.items()})
        got = screen.apply(truncated).selected

        pd.testing.assert_frame_equal(got, full.iloc[:rows], check_exact=True)


def test_the_causality_judgement_itself_catches_a_look_ahead_component(panel):
    """自检：把「用了未来」的部件喂给同一条判据，判据必须变红。

    没有这一条，上面那条参数化断言可能是**空转**的——它只证明了「因果部件是因果的」。
    这里用一个故意看全列的过滤器，证明判据真的会失败。
    """

    def peeking(working):
        return working["close"] > working["close"].mean()  # 用了整列（含未来）的均值

    closes = {"sh600000": [10.0, 12.0, 11.0, 15.0], "sz000001": [20.0, 19.0, 22.0, 21.0]}
    screen = Screen(filters=(peeking,))
    full = screen.apply(price_panel(panel, closes)).selected

    with pytest.raises(AssertionError):
        pd.testing.assert_frame_equal(
            screen.apply(price_panel(panel, {k: v[:2] for k, v in closes.items()})).selected,
            full.iloc[:2],
            check_exact=True,
        )


def test_candidates_agrees_with_apply_at_a_specified_as_of(panel):
    """AC 2 的组合用法：以指定评估日算，再取该日的候选集。

    「整张帧 + 按行取用」与「先截断到评估日再取」必须一致——两条路都该能用。
    """
    prices = price_panel(panel)
    as_of = prices["close"].index[2]
    screen = Screen(factor=lambda p: momentum(p["close"], 1), top_n=2)

    whole = screen.apply(prices)
    truncated = screen.apply(prices, as_of=as_of)

    assert truncated.candidates(as_of) == whole.candidates(as_of)
    assert len(truncated.candidates(as_of)) == 2


def test_as_of_truncates_and_the_future_does_not_leak(panel):
    """评估日之后的记录不参与——AC 5 的正面断言。"""
    screen = Screen(factor=lambda p: momentum(p["close"], 1), top_n=1)
    prices = price_panel(panel)

    as_of_second = screen.apply(prices, as_of=prices["close"].index[1]).selected
    as_of_last = screen.apply(prices).selected

    assert len(as_of_second) == 2
    pd.testing.assert_frame_equal(as_of_second, as_of_last.iloc[:2], check_exact=True)


def test_an_as_of_that_is_not_a_trading_day_is_rejected(panel):
    """报错而非向前取最近的交易日——否则「这一天的结果」其实是别的一天算的，且静默。"""
    prices = price_panel(panel)

    with pytest.raises(ValueError, match="不是交易日"):
        Screen().apply(prices, as_of="2024-01-06")  # 周六


def test_an_empty_candidate_set_is_a_legitimate_result_not_an_error(panel):
    """「那天一个都没选出」与「那天不在数据里」是两件事：前者合法。"""
    screen = Screen(filters=(lambda p: p["close"] > 1000,))

    got = screen.apply(price_panel(panel))

    assert got.selected.shape == price_panel(panel)["close"].shape
    assert not got.selected.any().any()


# --- 候选集查询（AC 2） --------------------------------------------------------


def test_candidates_are_ordered_from_best_to_worst(panel):
    """顺序即排名：动量最好的在前。"""
    screen = Screen(factor=lambda p: momentum(p["close"], 1))

    got = screen.apply(price_panel(panel))

    # 第 4 行：A 涨（+0.333），B 持平（0），C 跌（−0.5）
    assert got.candidates(panel_index(panel)[3]) == ["sh600000", "sz000001", "sz000002"]


def test_candidates_without_a_factor_fall_back_to_symbol_order(panel):
    """没有因子时按代码升序，而不是保持面板的列序——列序是实现细节，不该外泄。"""
    got = Screen().apply(price_panel(panel, {"sz000001": [1.0], "sh600000": [1.0]}))

    assert got.candidates(panel_index(panel)[0]) == ["sh600000", "sz000001"]


def test_candidates_on_a_non_trading_day_is_rejected(panel):
    got = Screen().apply(price_panel(panel))

    with pytest.raises(ValueError, match="不是交易日"):
        got.candidates(pd.Timestamp("2024-01-06"))


def test_candidates_agree_with_the_selected_frame(panel):
    """两个出口（帧与清单）必须一致，否则「买哪些」会有两个答案。"""
    screen = Screen(filters=(lambda p: above_ma(p["close"], 2),))
    result = screen.apply(price_panel(panel))

    for stamp in result.selected.index:
        from_frame = sorted(s for s in result.selected.columns if result.selected.loc[stamp, s])
        assert sorted(result.candidates(stamp)) == from_frame, stamp


def test_a_factor_alone_reports_scores_for_every_symbol(panel):
    """因子值原样交出，便于查看「差多少」而不只是「没选中」。"""
    result = Screen(factor=lambda p: momentum(p["close"], 1)).apply(price_panel(panel))

    assert result.scores.loc[result.scores.index[1], "sh600000"] == pytest.approx(1.0)
    assert result.scores.loc[result.scores.index[1], "sz000001"] == pytest.approx(0.0)


def test_a_screen_without_a_factor_has_no_scores(panel):
    result = Screen(filters=(lambda p: new_high(p["close"], 2),)).apply(price_panel(panel))

    assert result.scores.empty


def test_the_screen_sees_the_same_price_series_that_gets_traded(make_market, zero_cost_rules):
    """选股用的价格必须与撮合用的**是同一条序列**（后复权），否则规则在两个序列上被评估。

    构造一次 10 送 10：原始价在除权日腰斩（20.0 → 10.0，一个 −50% 的假跳空），而后复权
    视图把它还原成连续的 20.0。再给一个「收盘价高于 15」的过滤条件，并让买单落在**除权日
    之后**：

    - 面板取**后复权价**（正确）：20.0 > 15 → 选中 → 成交；
    - 面板取**原始价**（错误）：10.0 > 15 → 落选 → 一笔都不成交。

    于是「有没有成交」就是那个可观测的差别。这类错误不报错，只让选出来的标的与实际能
    成交的价格不是一回事（ADR-0003 否决「不复权直接用」正是为此）。
    """
    from mbt.backtest import run_portfolio_backtest
    from mbt.data import AdjustmentEvent
    from mbt.universe import UniverseRules

    frame = flat_bars(4, 20.0)
    frame.iloc[2:, :] = 10.0  # 第 3 根起因除权而腰斩
    events = (AdjustmentEvent(symbol="sh600000", ex_date=frame.index[2].date(), bonus_per_10=10.0),)
    market = MarketData(symbol="sh600000", prices=frame, events=events)
    # 另一只价格很低，故它永远不满足「高于 15」——候选只可能来自上面那只。
    other = make_market("sz000001", flat_bars(4, 5.0))

    class BuyOnTheExDate(bt.Strategy):
        def next(self):
            if len(self) == 3:  # 除权日那一根
                for data in self.datas:
                    self.buy(data=data)

    result = run_portfolio_backtest(
        [market, other],
        BuyOnTheExDate,
        cash=100_000.0,
        max_positions=2,
        rules=zero_cost_rules,
        universe_rules=UniverseRules(min_bars=0),
        screen=Screen(filters=(lambda p: p["close"] > 15,)),
        sizer=bt.sizers.FixedSize,
        sizer_options={"stake": 100},
    )

    assert (
        len(result.trades) == 1
    ), "选股面板取的是原始价——除权后的 10.0 过不了「高于 15」，而它实际成交在 20.0"
    assert result.trades.iloc[0]["price"] == pytest.approx(20.0)


def panel_index(panel):
    """面板的日期索引——供测试取某一日。"""
    return price_panel(panel)["close"].index


# --- 作为回测的入场过滤器（AC 3） ----------------------------------------------


def test_a_screen_gates_buys_without_rewriting_the_condition(make_market, zero_cost_rules):
    """同一条件既用于选股、也用于回测入场，**无需重写**（AC 3）。

    选股规则按收盘价取最高的那一只——即 ``sz000001``（20.0 > 10.0），故对 ``sh600000`` 的
    买入被撮合层拒掉。条件是那一份，不是抄一遍。
    """
    from mbt.backtest import run_portfolio_backtest
    from mbt.universe import UniverseRules

    a = make_market("sh600000", flat_bars(4, 10.0))
    b = make_market("sz000001", flat_bars(4, 20.0))
    screen = Screen(filters=(lambda p: p["close"] > 5,), factor=lambda p: p["close"], top_n=1)

    result = run_portfolio_backtest(
        [a, b],
        BuyAll,
        cash=100_000.0,
        max_positions=2,
        rules=zero_cost_rules,
        universe_rules=UniverseRules(min_bars=0),
        screen=screen,
        sizer=bt.sizers.FixedSize,
        sizer_options={"stake": 100},
    )

    assert result.trades["price"].tolist() == [20.0], "只该买到被选中的那一只"
    blocked = result.rejected[result.rejected["symbol"] == "sh600000"]
    assert len(blocked) == 1
    assert blocked.iloc[0]["reason"] == "未被选股规则选中"


def test_the_strategy_can_tell_not_selected_apart_from_out_of_the_pool(
    make_market, zero_cost_rules
):
    """两张掩码分开给：策略要能分辨「它出池了」与「今天没选它」。

    合成一道闸门虽然等价，但会把这份信息抹掉，而策略恰恰需要它来决定卖不卖——出池要清仓，
    只是没被选中则未必。
    """
    from mbt.backtest import run_portfolio_backtest
    from mbt.universe import UniverseRules

    a = make_market("sh600000", flat_bars(4, 10.0))
    b = make_market("sz000001", flat_bars(4, 20.0))
    seen = {}

    class Peek(bt.Strategy):
        def next(self):
            today = self.data0.datetime.date(0)
            for name, mask in (
                ("universe", self.broker.universe_mask),
                ("selection", self.broker.selection_mask),
            ):
                seen.setdefault(name, {})[today] = {
                    symbol: bool(mask.at[pd.Timestamp(today), symbol])
                    for symbol in ("sh600000", "sz000001")
                }

    run_portfolio_backtest(
        [a, b],
        Peek,
        rules=zero_cost_rules,
        universe_rules=UniverseRules(min_bars=0),
        screen=Screen(factor=lambda p: p["close"], top_n=1),
    )

    first = pd.Timestamp("2024-01-02").date()
    assert seen["universe"][first] == {"sh600000": True, "sz000001": True}, "两只都在池内"
    # 规则按收盘价取最高者，故被选中的是 20.0 的那一只。
    assert seen["selection"][first] == {"sh600000": False, "sz000001": True}, "只有一只被选中"


def test_a_screen_is_intersected_with_the_universe(make_market, zero_cost_rules):
    """选股规则越不出股票池，且**池子先判**。

    造法：规则只认 ``sz300750``（创业板），而准入规则只收主板——于是它被选股规则选中、
    却被池子挡住，拒单理由是「不在股票池内」。

    这条证明的是**两道闸门的先后**（池子在前）。「股票池规则在选股时同样生效」由
    ``test_the_universe_mask_is_intersected`` 直接证明——那一条测的是 ``Screen`` 自己与池子
    取交集。这里若只留撮合层，是分辨不出「选股时取了交集」与「撮合时另判了一次」的，
    因为两者的可观测结果相同。
    """
    from mbt.backtest import run_portfolio_backtest
    from mbt.universe import MAIN_BOARD, UniverseRules

    a = make_market("sh600000", flat_bars(4, 10.0))
    b = make_market("sz300750", flat_bars(4, 20.0))
    screen = Screen(filters=(lambda p: p["close"] > 15,))  # 只有创业板那只满足

    result = run_portfolio_backtest(
        [a, b],
        BuyAll,
        cash=100_000.0,
        max_positions=2,
        rules=zero_cost_rules,
        universe_rules=UniverseRules(boards=frozenset({MAIN_BOARD}), min_bars=0),
        screen=screen,
        sizer=bt.sizers.FixedSize,
        sizer_options={"stake": 100},
    )

    assert len(result.trades) == 0
    blocked = result.rejected[result.rejected["symbol"] == "sz300750"]
    assert len(blocked) == 1
    # 池子先判，故理由是「不在股票池内」——它确实也没被选中（交集的结果），
    # 但先报的那一条更能说明问题。
    assert blocked.iloc[0]["reason"] == "不在股票池内"


def test_without_a_screen_nothing_is_gated_by_selection(make_market, zero_cost_rules):
    """不给选股规则时不设那道闸门——既有行为不变。"""
    from mbt.backtest import run_portfolio_backtest
    from mbt.universe import UniverseRules

    a = make_market("sh600000", flat_bars(4, 10.0))
    b = make_market("sz000001", flat_bars(4, 20.0))

    result = run_portfolio_backtest(
        [a, b],
        BuyAll,
        cash=100_000.0,
        max_positions=2,
        rules=zero_cost_rules,
        universe_rules=UniverseRules(min_bars=0),
        sizer=bt.sizers.FixedSize,
        sizer_options={"stake": 100},
    )

    assert len(result.trades) == 2


# --- 掩码按**下单那根**判，不按成交那根（B1 那次改动的落点） --------------------


def selective_bars(open_, close, start="2024-01-02"):
    """由给定开/收盘造字段宽表：high/low 各留一点振幅，避免被当成一字板。"""
    index = pd.bdate_range(start, periods=len(close))
    return pd.DataFrame(
        {
            "open": open_,
            "high": [max(o, c) * 1.001 for o, c in zip(open_, close, strict=True)],
            "low": [min(o, c) * 0.999 for o, c in zip(open_, close, strict=True)],
            "close": close,
            "volume": [1000] * len(close),
        },
        index=index,
    )


class BuyWhenSelected(bt.Strategy):
    """只在**被选股规则选中**的那一根下买单——B1 的 `_enter_if_selected` 同一形状。"""

    def next(self):
        today = self.data.datetime.date(0)
        mask = self.broker.selection_mask
        if mask is not None and bool(mask.at[pd.Timestamp(today), self.data._name]):
            self.buy()


def test_a_buy_fills_at_the_next_open_even_if_the_selection_is_gone_by_then(
    make_market, zero_cost_rules
):
    """**本期改动的落点**：选股只在 T 根成立，T+1 已不成立，订单**照样成交**。

    改之前：撮合在**成交那一根**（T+1）复查选股掩码，于是这笔单一律被拒，理由写作
    「未被选股规则选中」。实测（326 标的、2,701 根）B1 的入选格里有 **86.3% 只连续成立一天**，
    故那道复查实际上把入场条件悄悄改成了「入选 **且** 次日仍入选」。

    构造：收盘 [10.6, 10.0, 10.0]，选股规则是「收盘 > 10.5」，故掩码 = [True, False, False]。
    策略在第 0 根下单；第 1 根掩码已为 False。断言**成交了**，且成交价 = 第 1 根的**开盘价**
    （11.0，刻意与收盘 10.0 不同，以证明用的是下一根开盘而不是当根收盘）。
    """
    from mbt.backtest import run_portfolio_backtest
    from mbt.universe import UniverseRules

    prices = selective_bars(open_=[10.6, 11.0, 11.0], close=[10.6, 10.0, 10.0])
    market = make_market("sh600000", prices)
    screen = Screen(filters=(lambda p: p["close"] > 10.5,))

    result = run_portfolio_backtest(
        [market],
        BuyWhenSelected,
        cash=1_000_000.0,
        max_positions=1,
        rules=zero_cost_rules,
        universe_rules=UniverseRules(min_trading_days=0),
        screen=screen,
        sizer=bt.sizers.FixedSize,
        sizer_options={"stake": 100},
    )

    buys = result.trades[result.trades["size"] > 0]
    assert len(buys) == 1, "掩码在 T 根成立，订单应当成交"
    assert buys.iloc[0]["price"] == pytest.approx(11.0), "成交价必须是下一根的开盘价"
    assert len(result.rejected) == 0, "不该有任何拒单"


def test_a_buy_is_still_refused_when_the_selection_never_held_at_creation(
    make_market, zero_cost_rules
):
    """对照：下单那根**就没被选中**时照样拒单——这道守卫没有被削弱。

    构造：收盘 [10.0, 10.6, 10.6]，选股规则是「收盘 > 10.5」。策略在第 0 根下单，而第 0 根
    收盘 10.0 **未**被选中；第 1 根才被选中。旧口径（按成交那根判）会**成交**，新口径
    （按下单那根判）会**拒单**——故这条同时证明判据时点真的换了，不是「两处都放过」。
    """
    from mbt.backtest import run_portfolio_backtest
    from mbt.universe import UniverseRules

    prices = selective_bars(open_=[10.0, 10.6, 10.6], close=[10.0, 10.6, 10.6])
    market = make_market("sh600000", prices)
    screen = Screen(filters=(lambda p: p["close"] > 10.5,))

    class BuyOnFirstBar(bt.Strategy):
        def next(self):
            if len(self) == 1:
                self.buy()

    result = run_portfolio_backtest(
        [market],
        BuyOnFirstBar,
        cash=1_000_000.0,
        max_positions=1,
        rules=zero_cost_rules,
        universe_rules=UniverseRules(min_trading_days=0),
        screen=screen,
        sizer=bt.sizers.FixedSize,
        sizer_options={"stake": 100},
    )

    assert len(result.trades) == 0, "下单那根没被选中，不该成交"
    assert len(result.rejected) == 1
    assert result.rejected.iloc[0]["reason"] == "未被选股规则选中"


def test_the_pool_mask_is_also_judged_on_the_creation_bar(make_market, zero_cost_rules):
    """股票池那条走同一口径：池子**在下单那根**成立即可，成交那根不复查。

    池子通常按「上市满 N 个交易日」算，本来就会连续成立多日；但与选股掩码走两个口径是
    说不通的——两者都是「这根下不下单」的判据，故一并按创建日判。
    """
    from mbt.backtest import run_portfolio_backtest
    from mbt.universe import UniverseRules

    prices = selective_bars(open_=[10.0, 11.0, 11.0], close=[10.0, 10.0, 10.0])
    market = make_market("sh600000", prices)

    # 池子：只在前两根内成立（min_bars=1 时第 0 根就够，之后仍成立），故改用 extra_mask
    # 把第 1 根起踢出池子——这正好是「下单那根在池、成交那根不在」的情形。
    inside = pd.DataFrame(
        {"sh600000": [True, False, False]},
        index=prices.index,
    )
    rules = UniverseRules(min_trading_days=0, extra_mask=inside)

    class BuyOnFirstBar(bt.Strategy):
        def next(self):
            if len(self) == 1:
                self.buy()

    result = run_portfolio_backtest(
        [market],
        BuyOnFirstBar,
        cash=1_000_000.0,
        max_positions=1,
        rules=zero_cost_rules,
        universe_rules=rules,
        sizer=bt.sizers.FixedSize,
        sizer_options={"stake": 100},
    )

    assert len(result.trades) == 1, "下单那根在池内，应当成交"
    assert result.trades.iloc[0]["price"] == pytest.approx(11.0)
