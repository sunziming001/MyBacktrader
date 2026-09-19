"""砖型策略：何时买、买多少、先买谁、何时卖（票据 #87）。

**买入条件不在这里测**——它是 :func:`mbt.screen.brick_screen` 那几条过滤器的事，已由
``tests/test_brick_screen.py`` 钉住。这里测的是策略那四件事，它们全都依赖**持仓与资金**，
而选股规则按定义不管持仓。

两组证据，各有来路：

- **端到端的走一遍**（真实规则 + 真实信号 + 一段落点已知的合成行情）：证明两条路接上了，
  且成交落在该落的那根上。
- **信号与选股结果手工给定**（其余用例）：这一层的判据要精确到某一天，而砖型那个公式的
  落点是递推算出来的、不是想让它在哪天就在哪天。故这里把「今天哪些是候选、各排第几、
  哪些是绿砖」直接写出来——测的仍是策略（它怎么用这三样东西），而三样东西本身另有测试。
"""

from __future__ import annotations

import backtrader as bt
import numpy as np
import pandas as pd
import pytest

from mbt.backtest import FixedAmountSizer, run_portfolio_backtest
from mbt.data import Panel
from mbt.screen import ScreenResult, brick_screen, brick_signals
from mbt.signals import above_yellow, white_above_yellow
from mbt.strategy import Brick
from mbt.universe import UniverseRules

#: 每笔的目标金额（需求方给定，票据 #81：每笔 20 万）。
AMOUNT = 200_000.0

#: 十二个交易日，测试自己排日子。
DATES = pd.bdate_range("2024-01-02", periods=12)

MAIN = "sh600000"
OTHER = "sz000001"


def flat_frame(opens=None, closes=None, *, volume=1000):
    """一段 o/h/l/c 都可以指定的 K 线：high/low 拉开一点，免得某根被当成一字板。"""
    count = len(closes)
    if opens is None:
        opens = list(closes)
    high = [max(o, c) * 1.02 for o, c in zip(opens, closes, strict=True)]
    low = [min(o, c) * 0.98 for o, c in zip(opens, closes, strict=True)]
    return pd.DataFrame(
        {
            "open": opens,
            "high": high,
            "low": low,
            "close": closes,
            "volume": [volume] * count,
        },
        index=DATES[:count],
    )


def market(make_market, symbol, closes, *, opens=None, drop=()):
    """一个标的的已质检行情；``drop`` 给出要**整根去掉**的日期下标（停牌）。"""
    frame = flat_frame(opens, closes)
    if drop:
        frame = frame.drop(index=[DATES[i] for i in drop])
    return make_market(symbol, frame)


class CraftedScreen:
    """把选股结果**直接给定**，绕过规则本身。

    这一层要测的是策略怎么用「选了谁、各排第几」，故那两样东西手工写出来最清楚——用真规则
    的话，落点会随砖型那个递推公式漂，而本条并不想测那个公式。

    ``align=True``（默认）把给定结果对到引擎手上的面板上。有一条测试专门要 ``align=False``：
    它要的就是**对不上**（少几天），好验那道「选股表必须盖住每个交易日」的守卫。
    """

    def __init__(self, selected: pd.DataFrame, scores: pd.DataFrame | None = None, *, align=True):
        self._selected = selected
        self._scores = scores
        self._align = align

    def apply(self, panel, *, as_of=None, universe_mask=None, progress=None):
        _ = (panel, as_of, universe_mask, progress)
        selected = self._selected
        scores = self._scores if self._scores is not None else selected.iloc[:0]
        if self._align:
            # 形状要与引擎手上的面板逐格一致，否则 `Screen` 那套形状校验在引擎里不设防。
            # 补齐出来的行（某标的停牌、别的标的在交易）**不是候选**，故填 0 而不是让它
            # 变成 NaN——`bool(NaN)` 是 True，那会把停牌日当成入选。
            reference = next(iter(panel.fields.values()))
            selected = self._selected.reindex_like(reference).astype(float).fillna(0.0) > 0
            scores = scores.reindex_like(reference)
        return ScreenResult(selected=selected, scores=scores)


def selection_frame(picked: dict[str, list[int]], symbols=(MAIN,)) -> pd.DataFrame:
    """``{标的: [第几根]}`` → 选股掩码（其余格为假）。"""
    frame = pd.DataFrame(False, index=DATES, columns=list(symbols))
    for symbol, bars in picked.items():
        for bar in bars:
            frame.loc[DATES[bar], symbol] = True
    return frame


def score_frame(values: dict[str, list[float]]) -> pd.DataFrame:
    """``{标的: [逐根分数]}`` → 分数表。"""
    return pd.DataFrame(values, index=DATES)


def hold_everything_screen(symbols=(MAIN,)) -> CraftedScreen:
    """每天都把标的全选上——用来单独看**卖**那一条，不必先造一个买入信号。"""
    frame = pd.DataFrame(True, index=DATES, columns=list(symbols))
    return CraftedScreen(frame)


def run(
    markets,
    strategy,
    *,
    rules,
    green: dict[str, list[int]] | None = None,
    signals=None,
    screen=None,
    cash=1_000_000.0,
    max_positions=5,
    **params,
):
    """跑一次组合回测；砖型那一套参数（固定金额 sizer、最大 5 只）默认就在这里。

    ``green`` 给出「哪个标的的第几根是绿砖」；不给就是「哪根都不是绿砖」。信号与这批行情
    逐格对齐是**这里**保证的——多一列少一列 ``Panel`` 都会当场报错，而那个报错与要测的东西
    无关。要给现成的信号（端到端那条）就传 ``signals=``。

    ``max_positions`` 同时给**撮合层**（它拦掉超出的买入）与 sizer——两处口径必须一致，
    否则「名额满了」这件事会有两种互相矛盾的读法。
    """
    if signals is None:
        symbols = [market.symbol for market in markets]
        frame = pd.DataFrame(0.0, index=DATES, columns=symbols)
        for symbol, where in (green or {}).items():
            for bar in where:
                frame.loc[DATES[bar], symbol] = 1.0
        signals = {"green_brick": frame}

    return run_portfolio_backtest(
        markets,
        strategy,
        cash=cash,
        max_positions=max_positions,
        rules=rules,
        universe_rules=UniverseRules(min_trading_days=0),
        screen=screen,
        signals=signals,
        sizer=FixedAmountSizer,
        sizer_options={"amount": AMOUNT, "max_positions": max_positions},
        **params,
    )


# --- 端到端：真实规则 + 真实信号走一遍 -----------------------------------------


#: 端到端那条用的行情里，**成交那几根**的下标（0 起）。
#:
#: 它们是「下单日的次一根」——入选日或绿砖日的**下一根**。这道门在下面那条用例里收到窗口
#: ``(2,)``（理由见那里的注释），于是这段行情给出**两个完整回合**：
#:
#: ==============  ======  ==========================================
#: 日期            下标    这一天是什么
#: ==============  ======  ==========================================
#: 2024-01-25      17      首次入选（01-24）的次根，**买入**成交于此
#: 2024-02-15      32      首次绿砖（02-14）的次根，**卖出**成交于此
#: 2024-02-22      37      再次入选（02-21）的次根，**买入**成交于此
#: 2024-02-29      42      再次绿砖（02-28）的次根，**卖出**成交于此
#: ==============  ======  ==========================================
FILL_BARS = (17, 32, 37, 42)

#: 成交那几根的开盘价相对收盘价的倍数。
#:
#: 刻意与收盘价不同——否则「成交价取的是**开盘**」与「取的是收盘」在这份数据上分不出来。
#: 砖型图只用 high/low/close，故改 open **不影响**上面那些落点。
FILL_OPEN_RATIO = 1.01


def brick_prices():
    """一段落点已知的合成行情：走出一段段涨跌，故砖型图真的红绿相间。

    实测的落点（下面那条端到端测试就按这几个日子断言）：入选 **01-24** 与 **02-21**，
    绿砖落在 01-08~01-23、02-14~02-20、02-28~03-05 三段里。
    """
    closes = np.concatenate(
        [
            np.linspace(20.0, 10.0, 16),
            np.linspace(10.2, 11.4, 6),
            np.array([11.35, 11.30]),
            np.linspace(11.9, 15.5, 6),
            np.linspace(15.4, 12.6, 5),
            np.linspace(12.7, 18.0, 6),
            np.linspace(17.4, 13.0, 5),
        ]
    )
    index = pd.bdate_range("2024-01-02", periods=closes.size)
    opens = closes.copy()
    opens[list(FILL_BARS)] = closes[list(FILL_BARS)] * FILL_OPEN_RATIO
    return pd.DataFrame(
        {
            "open": opens,
            "high": closes * 1.02,
            "low": closes * 0.98,
            "close": closes,
            "volume": [1000] * closes.size,
        },
        index=index,
    )


def gate_panel(market) -> Panel:
    """把一只标的的行情拼成 黄线门要的面板（它只用 close）。"""
    return Panel(
        {
            field: market.prices[[field]].rename(columns={field: market.symbol})
            for field in ("close", "high")
        }
    )


def test_the_whole_pipeline_buys_then_sells_on_the_right_bars(make_market, zero_cost_rules):
    """端到端：下单在**当天收盘**、成交在**次一根开盘**，买卖各按它该有的那一根落。

    这条走的是生产那两条路——``brick_screen`` 出名单、``brick_signals`` 出绿砖读数——故它
    同时证明「两条路接上了」与「成交落在该落的那根、且取的是开盘价」。

    四个落点（见 :data:`FILL_BARS`）成对：**入选 → 次根买入**、**绿砖 → 次根卖出**，两轮。
    """
    prices = brick_prices()
    frame = make_market(MAIN, prices)

    result = run(
        [frame],
        Brick,
        rules=zero_cost_rules,
        signals=brick_signals([frame]),
        # **黄线族的门收到 ``(2,)``、涨幅门关掉**：这条用例验的是**策略**（何时买、何时卖、
        # 按哪个价成交），而这段行情的落点是按砖型那三条判据选的——出厂默认下它会少选一天
        # （第二个入选日涨幅 +8.35%，被「涨幅 < 5.8%」挡掉；而黄线要 114 根，这段只有 46 根）。
        # 故这里把三道门都放宽、并当场断言它们确实放行，免得测试变成在测门。
        screen=brick_screen(yellow_windows=(2,), max_gain=None),
    )

    # 三道门都要断言：它们互不蕴含（见 `white_above_yellow` 的 docstring），故「门放行了」
    # 这件事得逐道确认，否则测试可能在测另一道门。
    closes = gate_panel(frame)["close"]
    for label, gate in (
        ("白线在黄线上", white_above_yellow(closes, 10, (2,))[MAIN]),
        ("收在黄线之上", above_yellow(closes, (2,))[MAIN]),
    ):
        for bar in FILL_BARS[::2]:  # 那两个**下单日**（成交在它们的次根）
            assert bool(gate.iloc[bar - 1]), f"第 {bar - 1} 根没通过「{label}」那道门"

    fills = result.trades.sort_values("date").reset_index(drop=True)
    assert len(fills) == 4, f"两轮买卖各一笔，实际成交 {len(fills)} 笔"

    expected_dates = [prices.index[bar] for bar in FILL_BARS]
    assert fills["date"].tolist() == expected_dates

    # 成交价取的是那一根的**开盘**，而不是收盘——上面刻意让两者不同。
    expected_prices = [float(prices["open"].iloc[bar]) for bar in FILL_BARS]
    assert fills["price"].tolist() == pytest.approx(expected_prices)

    assert (fills["size"] > 0).tolist() == [True, False, True, False], "买、卖、买、卖"
    assert fills["size"].iloc[1] == -fills["size"].iloc[0], "一次清仓，不是分批"
    assert fills["size"].iloc[3] == -fills["size"].iloc[2]


# --- 成交时点与金额 -----------------------------------------------------------


def test_the_buy_fills_at_the_next_bar_open(make_market, zero_cost_rules):
    """名单在**本根收盘**算出来，订单在**次一根开盘**成交——按收盘价成交是拿不到的价。"""
    closes = [10.0] * 12
    opens = [10.0] * 4 + [11.0] + [10.0] * 7  # 第 5 根的**开盘**与别根不同
    frame = market(make_market, MAIN, closes, opens=opens)
    screen = CraftedScreen(selection_frame({MAIN: [3]}))

    result = run([frame], Brick, rules=zero_cost_rules, screen=screen)

    fills = result.trades
    assert len(fills) == 1
    assert fills.iloc[0]["date"] == DATES[4], "第 4 根（0 起）入选，成交落在第 5 根"
    assert fills.iloc[0]["price"] == pytest.approx(11.0), "成交价取的是那一根的**开盘**"


def test_each_position_gets_the_fixed_amount(make_market, zero_cost_rules):
    """每笔投入**固定 20 万**，而不是「组合总值的一份」。

    sizer 按**最坏成交价**定量（这里是 10 × 1.1 = 11，主板限幅 10%），故股数是
    ``⌊200000 ÷ 11⌋ = 18181``。这条同时钉住「按最坏价留余地」那道既有的账没在固定金额这条
    路上漏掉。
    """
    frame = market(make_market, MAIN, [10.0] * 12)
    screen = CraftedScreen(selection_frame({MAIN: [3]}))

    result = run([frame], Brick, rules=zero_cost_rules, screen=screen)

    assert len(result.trades) == 1
    assert result.trades.iloc[0]["size"] == int(AMOUNT / 11.0)


def test_a_cash_shortfall_buys_less_instead_of_skipping(make_market, zero_cost_rules):
    """资金不够时**少买**，不跳过——空着的名额比买不足更浪费。"""
    frame = market(make_market, MAIN, [10.0] * 12)
    screen = CraftedScreen(selection_frame({MAIN: [3]}))

    result = run([frame], Brick, rules=zero_cost_rules, screen=screen, cash=100_000.0)

    assert len(result.trades) == 1, "钱不够就不买 —— 那这个名额白空着"
    assert result.trades.iloc[0]["size"] == int(100_000.0 / 11.0)


def test_a_held_symbol_is_not_bought_again(make_market, zero_cost_rules):
    """已持有的**不加仓**——与 B1、均线交叉那两条策略同一处置。"""
    frame = market(make_market, MAIN, [10.0] * 12)
    # 每天都入选，而第一次买入之后就一直是持仓
    screen = CraftedScreen(pd.DataFrame(True, index=DATES, columns=[MAIN]))

    result = run([frame], Brick, rules=zero_cost_rules, screen=screen)

    assert len(result.trades) == 1, "同一天/隔天的重复入选不该变成加仓"
    assert result.trades.iloc[0]["size"] > 0


def test_the_higher_score_gets_the_last_slot(make_market, zero_cost_rules):
    """名额只有一个而两只都入选时，买的是**分数高的那只**。

    分数由引擎从规则那边**原样**交过来（票据 #83），策略据此排序。这条与
    ``tests/test_screen_score.py`` 那条成对：那条测「分数交到了策略手里」，这条测「策略真的
    按它排」。
    """
    first = market(make_market, MAIN, [10.0] * 12)
    second = market(make_market, OTHER, [20.0] * 12)
    screen = CraftedScreen(
        pd.DataFrame(True, index=DATES, columns=[MAIN, OTHER]),
        score_frame({MAIN: [0.1] * 12, OTHER: [0.9] * 12}),
    )

    result = run([first, second], Brick, rules=zero_cost_rules, screen=screen, max_positions=1)

    assert result.trades["symbol"].tolist() == [OTHER], "分数高的那只该占住名额"


# --- 卖出 ---------------------------------------------------------------------


def test_a_green_brick_sells_at_the_next_bar_open(make_market, zero_cost_rules):
    """持有期间当天是绿砖 → **次一根开盘**清仓。"""
    closes = [10.0] * 12
    opens = [10.0] * 4 + [11.0] * 8  # 第 5 根起开盘不同，好认出成交价取的是哪一根
    frame = market(make_market, MAIN, closes, opens=opens)
    screen = CraftedScreen(selection_frame({MAIN: [0]}))

    result = run([frame], Brick, rules=zero_cost_rules, screen=screen, green={MAIN: [5]})

    fills = result.trades.sort_values("date").reset_index(drop=True)
    assert len(fills) == 2
    assert fills.iloc[0]["date"] == DATES[1], "第 0 根入选，次根开盘买入"
    assert fills.iloc[1]["date"] == DATES[6], "第 5 根读到绿砖，次根开盘卖出"
    assert fills.iloc[1]["price"] == pytest.approx(11.0)


def test_a_buy_day_that_turns_green_sells_the_following_open(make_market, zero_cost_rules):
    """**买入当天就是绿砖**那种情形，自然落到「次日开盘卖出」——不为它单写一条分支。

    时序上必然如此：买入日的砖要等它收盘才知道，而当日买入的股份当日又不可卖（T+1），
    故最早能卖的就是买入日的次一根。
    """
    frame = market(make_market, MAIN, [10.0] * 12)
    screen = CraftedScreen(selection_frame({MAIN: [0]}))

    # 第 1 根就是**成交那一根**（第 0 根下单、第 1 根成交），而它当天是绿砖。
    result = run([frame], Brick, rules=zero_cost_rules, screen=screen, green={MAIN: [1]})

    fills = result.trades.sort_values("date").reset_index(drop=True)
    assert len(fills) == 2
    assert fills.iloc[0]["date"] == DATES[1], "买入成交在第 1 根"
    assert fills.iloc[1]["date"] == DATES[2], "买入日的次一根开盘卖出"
    assert fills.iloc[1]["size"] == -fills.iloc[0]["size"]


class CountingBrick(Brick):
    """记下**挂过几张卖单**（按订单号去重）——用来量「有没有重复下单」。

    计数放在 ``notify_order`` 里而不是别处：那是 backtrader 的公开回调，订单每一次状态变化
    都会经过它，故「这里见到几张卖单」就是「策略挂了几张」的忠实读数。
    """

    def __init__(self):
        super().__init__()
        self.sell_refs: list[int] = []

    def notify_order(self, order):
        super().notify_order(order)
        if order.issell() and order.ref not in self.sell_refs:
            self.sell_refs.append(order.ref)


def run_capturing(markets, strategy_cls, **kwargs):
    """跑一次并交回**那个策略实例**——backtrader 不把它给调用方，故这一层自己留一份。

    用子类而不是改 ``Brick``：测的是它的行为，不该为了可测性在生产类上开一个出口。
    """
    captured: list = []

    class Spy(strategy_cls):
        def __init__(self):
            super().__init__()
            captured.append(self)

    result = run(markets, Spy, **kwargs)
    assert captured, "策略一次都没被实例化"
    return result, captured[0]


def test_a_pending_sell_is_not_replaced_every_bar(make_market, zero_cost_rules):
    """绿砖连出好几根时，**只挂一张**卖单——不重复下单。

    造法是让卖单被停牌挡住：第 3 根读到绿砖、卖单挂在第 3 根，而第 4~6 根该标的没有 K 线，
    故它一直没成交。若每根都重挂，那里会多出好几张单。
    """
    frame = market(make_market, MAIN, [10.0] * 12, drop=(4, 5, 6))
    companion = market(make_market, OTHER, [20.0] * 12)
    screen = CraftedScreen(selection_frame({MAIN: [0]}))

    result, strategy = run_capturing(
        [frame, companion],
        CountingBrick,
        rules=zero_cost_rules,
        screen=screen,
        green={MAIN: [3, 4, 5, 6]},
        order_expiry_ticks=10,
    )

    fills = result.trades.sort_values("date").reset_index(drop=True)
    assert len(fills) == 2, "买入 + 卖出各一笔"
    assert fills.iloc[1]["size"] < 0
    assert len(strategy.sell_refs) == 1, f"卖单挂了 {len(strategy.sell_refs)} 张，本该只有一张"


def test_an_expired_sell_order_is_replaced_on_the_next_green_bar(make_market, zero_cost_rules):
    """卖单因停牌太久**作废**之后，下一次绿砖要能**重新挂**——否则持仓被永久卡住。

    没有这一步时，那只标的既不会再挂单、也永远不会卖：``_exiting`` 里的标记只增不减。
    """
    frame = market(make_market, MAIN, [10.0] * 12, drop=(4, 5, 6))
    companion = market(make_market, OTHER, [20.0] * 12)
    screen = CraftedScreen(selection_frame({MAIN: [0]}))

    result, strategy = run_capturing(
        [frame, companion],
        CountingBrick,
        rules=zero_cost_rules,
        screen=screen,
        green={MAIN: [3, 8]},
        order_expiry_ticks=1,
    )

    fills = result.trades.sort_values("date").reset_index(drop=True)
    assert len(fills) == 2, "买入 + 卖出：作废之后必须重挂，否则这笔持仓永远卖不掉"
    assert fills.iloc[1]["size"] < 0
    assert fills.iloc[1]["date"] >= DATES[9], "第二次绿砖（第 8 根）之后才成交"
    assert len(strategy.sell_refs) == 2, "第一张作废了，第二张才卖掉——故该有两张"


def test_there_is_no_stop_loss(make_market, zero_cost_rules):
    """**没有止损**：价格一路崩而始终不是绿砖时，持仓被一直拿着。

    这是需求方的设定（那张票的验收写着「无止损、无时间封顶」），代价是回撤不设上界。这条
    把它写成断言，免得将来有人顺手补一个止损——那会改掉整份回测的结果。
    """
    closes = [10.0, 9.0, 8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0, 1.0, 1.0]
    frame = market(make_market, MAIN, closes)
    screen = CraftedScreen(selection_frame({MAIN: [0]}))

    result = run([frame], Brick, rules=zero_cost_rules, screen=screen)

    assert len(result.trades) == 1, "只有买入那一笔——跌掉九成也不卖"
    assert result.trades.iloc[0]["size"] > 0
    assert result.final_value < 900_000.0, "净值确实跟着跌下去了（不跌才说明这条测试没验到）"


# --- 与引擎的约定 -------------------------------------------------------------


def test_a_candidate_list_must_cover_every_trading_day(make_market, zero_cost_rules):
    """选股表缺交易日 → **报错**，而不是静默地那一根都不建仓。

    与 B1 同一处判据（那条路也有一份）：``.get`` 查不到当天与「当天确实没有候选」长得一模
    一样，而前者会让整段区间一根都不建仓。故这里刻意**不对齐**（``align=False``）——对齐了
    就会把缺的日子补上，那道守卫也就无从触发。
    """
    frame = market(make_market, MAIN, [10.0] * 12)
    partial = pd.DataFrame(True, index=DATES[:-3], columns=[MAIN])
    screen = CraftedScreen(partial, align=False)

    with pytest.raises(ValueError, match="选股表缺少"):
        run([frame], Brick, rules=zero_cost_rules, screen=screen)


def test_the_strategy_is_named_as_a_library_object():
    """策略住在**库里**，``mbt backtest --strategy mbt.strategy:Brick`` 指得到它。

    它放进库而不是示例目录，是因为测试要 import 它、而 ``--strategy`` 也要指名它。
    """
    import importlib

    module = importlib.import_module("mbt.strategy")
    assert getattr(module, "Brick", None) is Brick
    assert issubclass(Brick, bt.Strategy)


def test_the_engine_clock_and_signal_reader_are_shared_with_the_examples():
    """``EngineClock`` 与 ``signal_value`` 在**库里**，示例改为从库里导入。

    抄一份的结果是每个使用者都要重新推导一次「今天到底是哪一根」——而那道题第一个版本就
    写错了（见类的 docstring）。这条钉住「只有一处定义」。
    """
    import importlib

    module = importlib.import_module("examples.strategies")
    from mbt.strategy import EngineClock, signal_value

    assert module.EngineClock is EngineClock
    assert module.signal_value is signal_value
