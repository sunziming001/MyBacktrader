"""**选股分数**要真的到得了策略手里（票据 #83）。

背景：``Screen`` 一次算出**两**样东西——``selected``（是/否）与 ``scores``（**选股分数**，
``CONTEXT.md``）。而引擎往策略那边传的历来只有 ``selected``，分数在传进去的路上就丢了。于是
「按分数取前 N 名」这句话在名额紧张的那几天不成立：引擎按**标的代码顺序**挑，而不是按名次挑。

这一处的能力与砖型无关，故这里用一条**很小的合成规则**验，不牵扯任何真实的选股逻辑。

四条判据，各自的证据不同：

1. **通了**——策略读到的分数就是规则算出的那一份（不是它自己复算的另一份）；
2. **按名次买**——5 个候选、只空 2 个名额时，买中的是分数最高的那两只，而**不是**代码
   顺序的前两只。这一条是**核心**：把分数换成任意一组与代码顺序错开的数，买中的那两只
   必须跟着分数走；
3. **不外溢**——没给规则、或规则不谈名次时这个字段不存在；给了也不改变不读它的策略的任何
   一笔成交。这一条用**库里真实的规则**（动量）走一遍，不只用合成规则；
4. **缺失即不合格**——分数缺失的标的不因「名次不知道」而被当成合格。
"""

from __future__ import annotations

import backtrader as bt
import numpy as np
import pandas as pd
import pytest

from examples.strategies import signal_value
from mbt.backtest import run_portfolio_backtest
from mbt.screen import SCREEN_SCORE_FIELD, Screen, ScreenResult, momentum_screen
from mbt.universe import UniverseRules

#: 五个候选，按**代码顺序**排。故意让分数与这个顺序错开，好把「按名次」与「按代码」分开。
SYMBOLS = ("sh600000", "sh688001", "sz000001", "sz000002", "sz300750")

#: 合成规则给出的分数：与代码顺序**不一致**，故「买中哪两只」能唯一地指出用的是哪种排序。
#: 三种假说在这组数上给出**三个不同**的答案，故这条测试不是空转：
#:
#: ==============  ============================  ================
#: 假说            答案                            与预期相同？
#: ==============  ============================  ================
#: 按分数          ``sz000001``(5)、``sz300750``(4)  **是**
#: 按代码顺序      ``sh600000``、``sh688001``     否
#: 按价格          ``sh688001``(50)、``sz300750``(40) 否
#: ==============  ============================  ================
SCRAMBLED_SCORES = {
    "sh600000": 1.0,
    "sh688001": 2.0,
    "sz000001": 5.0,
    "sz000002": 3.0,
    "sz300750": 4.0,
}

#: 按分数该买中的两只（降序前二）。
EXPECTED_BY_SCORE = {"sz000001", "sz300750"}

#: 按**代码顺序**会被买中的两只——它必须**不是**答案，否则这条测试量不出名次。
WOULD_BE_BY_CODE_ORDER = {"sh600000", "sh688001"}

#: 价格与分数**无关**，故「按价格挑」也会是错的答案（见上面那张表）。
CLOSES = {"sh600000": 10.0, "sh688001": 50.0, "sz000001": 20.0, "sz000002": 30.0, "sz300750": 40.0}


def _run(markets, strategy, *, rules, screen=None, max_positions=2, signals=None):
    """把九参数的那一大串收成一处——五个用例都跑同一个配置，只有规则与策略不同。"""
    return run_portfolio_backtest(
        markets,
        strategy,
        cash=1_000_000.0,
        max_positions=max_positions,
        rules=rules,
        universe_rules=UniverseRules(min_bars=0),
        screen=screen,
        signals=signals,
        sizer=bt.sizers.FixedSize,
        sizer_options={"stake": 100},
    )


def flat_markets(make_prices, make_market, bars=4):
    """五个候选的合成行情：各自一条平走的价格。"""
    return [make_market(name, make_prices([CLOSES[name]] * bars)) for name in SYMBOLS]


def scrambled_factor(panel):
    """合成因子：每个标的给一个**固定**分数，与它的价格和代码都无关。"""
    closes = panel["close"]
    return pd.DataFrame(
        {name: np.full(len(closes), value) for name, value in SCRAMBLED_SCORES.items()},
        index=closes.index,
    )


def factor_with_a_hole(panel):
    """同上，但把 ``sh600000`` 的分数留成**缺失**——用来验「缺失即不合格」。"""
    frame = scrambled_factor(panel)
    frame["sh600000"] = np.nan
    return frame


def run_and_capture(markets, rules, screen):
    """跑一次并把那个探针策略交回来——backtrader 不把策略实例交给调用方，故这一层要自己留。"""
    captured = {}

    class Spy(SignalSpy):
        def next(self):
            super().next()
            captured.setdefault("spy", self)

    _run(markets, Spy, rules=rules, screen=screen)
    return captured["spy"]


def by_score_screen() -> Screen:
    """一条只有排序因子、不过滤的合成规则。"""
    return Screen(factor=scrambled_factor)


def no_factor_screen() -> Screen:
    """与 :func:`by_score_screen` 选出**同一批**标的，但不排序——故它不谈名次。"""
    return Screen()


class StrippedScores:
    """把一条真规则包起来，**选出同一批标的但把分数去掉**。

    这是「注入分数本身有没有副作用」唯一能拿到对照的办法：代码里注入是无条件发生的（给了规则
    就有），故对照不能靠「关掉注入」，只能靠**造一条没有名次的规则**——它让 ``scores`` 成为
    空帧，于是引擎不注入，而 ``selected`` 与真规则逐格相同。两条路只有「有没有分数」这一处
    差别，成交与净值必须逐笔相同。
    """

    def __init__(self, rule):
        self._rule = rule

    def apply(self, panel, *, as_of=None, universe_mask=None, progress=None):
        screened = self._rule.apply(
            panel, as_of=as_of, universe_mask=universe_mask, progress=progress
        )
        # 空帧（0 行）即「这条规则不谈名次」——引擎据此不注入，与「压根没给规则」同一条路径。
        return ScreenResult(selected=screened.selected, scores=screened.scores.iloc[:0])


class BuyByScore(bt.Strategy):
    """按策略读到的选股分数**从高到低**买入，每根 K 线都试一遍。

    这是「按名次填空着的名额」的最小实现：名额由引擎的 ``max_positions`` 管，本类只负责把
    买单按名次**依次**提交出去，故名额不够时先到的那几只（即分数最高的那几只）占住名额。
    """

    def __init__(self):
        self.by_name = {data._name: data for data in self.datas}
        self.seen_rows = []

    def next(self):
        today = pd.Timestamp(self.datetime.date(0))
        scores = self.broker.signals.get(SCREEN_SCORE_FIELD) if self.broker.signals else None
        if scores is None or today not in scores.index:
            return
        row = scores.loc[today]
        if not self.seen_rows:
            self.seen_rows.append(row)
        ranked = sorted(
            (name for name in row.index if row[name] == row[name]),
            key=lambda name: (-row[name], name),
        )
        for name in ranked:
            self.buy(data=self.by_name[name])


class BuyEverything(bt.Strategy):
    """**不读任何信号**，见到可交易的标的就买——用来量注入本身有没有副作用。"""

    def next(self):
        today = pd.Timestamp(self.datetime.date(0))
        for data in self.datas:
            if self.getposition(data).size:
                continue
            if not self.broker.tradability_mask.at[today, data._name]:
                continue
            self.buy(data=data, size=100)


class SignalSpy(bt.Strategy):
    """第一根记下 ``broker.signals`` 与它读到的那个分数，供断言检查。"""

    def __init__(self):
        self.first_row = None
        self.value = None

    def next(self):
        if self.first_row is not None:
            return
        scores = self.broker.signals.get(SCREEN_SCORE_FIELD) if self.broker.signals else None
        today = pd.Timestamp(self.datetime.date(0))
        self.value = signal_value(self.broker.signals, SCREEN_SCORE_FIELD, today, "sh600000")
        if scores is not None and today in scores.index:
            self.first_row = scores.loc[today]


# --- 1. 通了：策略拿到的就是规则算出的那一份 --------------------------------


def test_the_strategy_reads_the_exact_scores_the_rule_computed(
    make_prices, make_market, zero_cost_rules
):
    """策略看到的选股分数与规则算出的**同值**——不是它自己复算出来的另一份。

    断言取的是**分数本身**（``SCRAMBLED_SCORES``），而不是「买中了谁」：后者只说明排序对了，
    说明不了那个数是从规则那边过来的。
    """
    markets = flat_markets(make_prices, make_market)
    spy = run_and_capture(markets, zero_cost_rules, by_score_screen())

    assert spy.first_row is not None, "策略一次都没读到分数——那字段没传下去"
    for name, expected in SCRAMBLED_SCORES.items():
        assert spy.first_row[name] == pytest.approx(expected), f"{name} 的分数不是规则给的那个"


# --- 2. 按名次买（核心） -----------------------------------------------------


def test_the_two_slots_go_to_the_highest_scores_not_to_code_order(
    make_prices, make_market, zero_cost_rules
):
    """5 个候选、只空 2 个名额时，买中的是**分数最高的两只**。

    这一条是本票据的核心：名次在引擎那边曾经丢掉，而丢掉之后**不会报错**——它只是悄悄地
    按代码顺序挑，且只在名额不够的那几天与你的预期分岔。
    """
    markets = flat_markets(make_prices, make_market)

    result = _run(markets, BuyByScore, rules=zero_cost_rules, screen=by_score_screen())
    bought = set(result.trades["symbol"])

    assert bought == EXPECTED_BY_SCORE, f"买中的是 {sorted(bought)}，与分数前二不符"
    assert bought != WOULD_BE_BY_CODE_ORDER, "买中的恰好是代码顺序前二——那这条测试量不出名次"


def test_an_explicit_top_n_and_the_slots_agree_on_who_gets_bought(
    make_prices, make_market, zero_cost_rules
):
    """规则自己截前 2 名时，买中的仍是那两只——名次与名额两处口径一致。"""
    markets = flat_markets(make_prices, make_market)

    result = _run(
        markets, BuyByScore, rules=zero_cost_rules, screen=Screen(factor=scrambled_factor, top_n=2)
    )

    assert set(result.trades["symbol"]) == EXPECTED_BY_SCORE


# --- 3. 不外溢：缺席、以及不改变别人 ----------------------------------------


def test_the_field_is_absent_when_no_rule_was_given(make_prices, make_market, zero_cost_rules):
    """没给选股规则时这个字段**不存在**——不是空帧、不是全 NaN。"""
    markets = flat_markets(make_prices, make_market)
    spy = run_and_capture(markets, zero_cost_rules, None)

    assert spy.value != spy.value, "缺席的字段必须读成 NaN（「不知道」天然是不动作）"


def test_a_rule_that_says_nothing_about_ranking_injects_nothing(
    make_prices, make_market, zero_cost_rules
):
    """规则**不过滤也不排序**时，它选出全部标的，但不该硬塞一个空帧当分数。

    空帧与「字段不存在」在策略那边读起来一样（都是缺失），但把空帧塞进去会让
    ``SCREEN_SCORE_FIELD in signals`` 这个判断说谎——而那是策略分辨「这条规则谈不谈名次」
    的依据。
    """
    markets = flat_markets(make_prices, make_market)
    spy = run_and_capture(markets, zero_cost_rules, no_factor_screen())

    assert spy.value != spy.value


def test_the_caller_supplied_signals_mapping_is_not_mutated(
    make_prices, make_market, zero_cost_rules
):
    """引擎往信号里并分数时**复制一份**，不改调用方给的那个映射。

    就地改会让调用方的对象在这次回测之后多出一个字段——他没要求过，而且下一次跑别的规则时
    它还留着，于是「这条规则谈不谈名次」在两次运行之间被串了起来。
    """
    markets = flat_markets(make_prices, make_market)
    given: dict = {}

    _run(markets, BuyEverything, rules=zero_cost_rules, screen=by_score_screen(), signals=given)

    assert given == {}, "调用方给的信号映射被就地改了"


def test_injecting_the_scores_changes_nothing_for_a_strategy_that_ignores_them(
    make_prices, make_market, zero_cost_rules
):
    """分数是**纯增量**：同一条规则、同样的选出集合，有没有分数跑出**同样的成交与净值**。

    用**库里真实的规则**（动量）走一遍，不只用合成规则——注入那一步与规则无关，故真实规则
    的证据才说明得了「其它规则一个字节都不受影响」。

    （这里不直接跑 B1：``b1_screen`` 要财务数据算出来的 PE 字段，而套件里没有行情——那是
    ``realmdata`` 那一组的事。注入点是规则的**下游**，故这条与 B1 走的是同一段代码。）
    """
    markets = flat_markets(make_prices, make_market)
    rule = momentum_screen(window=2, top_n=2)

    def run_with(screen):
        return _run(markets, BuyEverything, rules=zero_cost_rules, screen=screen)

    ranked = run_with(rule)
    stripped = run_with(StrippedScores(rule))

    assert not ranked.trades.empty, "两条路都没成交，那这条对照量不出东西"
    assert ranked.trades.equals(stripped.trades), "注入分数改变了不读它的策略的成交"
    assert ranked.equity_curve.equals(stripped.equity_curve), "净值曲线也不该变"


# --- 4. 缺失即不合格 ---------------------------------------------------------


def test_a_candidate_without_a_score_is_not_treated_as_qualified(
    make_prices, make_market, zero_cost_rules
):
    """分数缺失的标的不因「名次不知道」而被当成合格——它压根不该进候选。

    这一条是 ``Screen.apply`` 早就有的纪律（``selected &= scores.notna()``），但分数现在
    多走了一条路（并进信号给策略），故要在**这条新路上**再钉一次：传下去的帧里那一格仍是
    缺失，而不是被填成 0（填成 0 等于说「它排最后」，那是另一个意思）。
    """
    markets = flat_markets(make_prices, make_market)
    spy = run_and_capture(markets, zero_cost_rules, Screen(factor=factor_with_a_hole))

    assert spy.first_row is not None
    assert spy.first_row["sh600000"] != spy.first_row["sh600000"], "缺失被填成了数"

    result = _run(
        markets, BuyByScore, rules=zero_cost_rules, screen=Screen(factor=factor_with_a_hole)
    )
    assert "sh600000" not in set(result.trades["symbol"])
    assert set(result.trades["symbol"]) == EXPECTED_BY_SCORE, "空缺不该把名次往后挪一位"
