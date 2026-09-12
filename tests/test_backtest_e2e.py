"""验证单标的端到端最窄切片：价格表 → 回测 → 净值曲线与成交明细。

测试缝 S1：只走库公开入口。**刻意不使用假引擎**——本测试要验证的正是
「真实 backtrader + 我们的接线」这个组合；换成假引擎，测的就只剩我们自己的记账，
而漏掉最可能出现意外的地方（ADR-0007）。

期望值来自 backtrader 的成交语义（在 ``next()`` 中下的单于**次根 K 线开盘**成交）
加手工算术，而非由被测代码产出。
"""

import backtrader as bt
import pandas as pd
import pytest

from mbt.backtest import run_backtest
from mbt.data import TdxDataSource


class BuyOnce(bt.Strategy):
    """最小策略：没有持仓就买入，此后不动。"""

    def next(self):
        if not self.position:
            self.buy()


def test_equity_curve_is_hand_computable(make_prices, zero_cost_rules):
    """每根 K 线开高低收相同，故净值可手算。

    资金 1000，默认下单量为 1 股：次根 K 线以 21 成交，占用 21，余 979。
    净值依次为 1000（未持仓）、979+21=1000、979+22=1001、979+23=1002。

    价格从 20 起、每步约 +5%，是刻意的：``make_prices`` 令每根 K 线 ``o==h==l==c``，
    若相邻收盘恰好相差一个限幅（如 10→11 恰为 +10%），那根 K 线就是**涨停一字板**，
    买单会被撮合约束正确地挡下。此处要测的是净值记账，故避开该情形。

    用零费用规则表，使本测试只锁撮合机制；制度费用的数值由 ``test_costs.py`` 覆盖。
    """
    prices = make_prices([20.0, 21.0, 22.0, 23.0])

    result = run_backtest(
        prices, symbol="sh600000", strategy=BuyOnce, cash=1000.0, rules=zero_cost_rules
    )

    assert list(result.equity_curve.index) == list(prices.index)
    assert result.equity_curve.tolist() == [1000.0, 1000.0, 1001.0, 1002.0]
    assert result.final_value == 1002.0


def test_trades_record_every_fill(make_prices, zero_cost_rules):
    prices = make_prices([20.0, 21.0, 22.0, 23.0])

    result = run_backtest(
        prices, symbol="sh600000", strategy=BuyOnce, cash=1000.0, rules=zero_cost_rules
    )

    trades = result.trades
    assert len(trades) == 1
    fill = trades.iloc[0]
    assert fill["date"] == pd.Timestamp("2024-01-03")
    assert fill["size"] == 1
    assert fill["price"] == 21.0
    assert fill["value"] == 21.0


def test_trade_columns_are_exactly_the_documented_contract(make_prices, zero_cost_rules):
    """成交明细的列就是公开契约 ``TRADE_COLUMNS``——不多、不少、顺序一致。"""
    from mbt.backtest.engine import TRADE_COLUMNS

    result = run_backtest(
        make_prices([20.0, 21.0]),
        symbol="sh600000",
        strategy=BuyOnce,
        cash=1000.0,
        rules=zero_cost_rules,
    )

    assert list(result.trades.columns) == list(TRADE_COLUMNS)
    assert "symbol" in TRADE_COLUMNS, "平仓配对按标的分别做，这一列是必需的"


def test_every_fill_carries_the_symbol_it_belongs_to(make_prices, zero_cost_rules):
    """每一笔成交都要能**归属到具体标的**。

    ``rejected`` 一直有这一列，``trades`` 却漏了——于是多标的回测里看成交明细只能靠猜
    （用价格反推）。更要紧的是它不只是给人看的：平仓配对必须按标的分别做，少了这一列，
    一个标的的买入会被当成另一个标的卖出的对手方（见
    :func:`mbt.metrics.closed_trade_pnls`）。
    """
    result = run_backtest(
        make_prices([20.0, 21.0]),
        symbol="sh600000",
        strategy=BuyOnce,
        cash=1000.0,
        rules=zero_cost_rules,
    )

    assert list(result.trades["symbol"]) == ["sh600000"]


def test_value_is_the_traded_amount_not_the_position_cost_basis(make_prices, zero_cost_rules):
    """``value`` 是**成交金额** ``size × price``，不是 backtrader 的 ``executed.value``。

    实测定死这条：买 100@10、卖 100@12 时，``executed.value`` **两行都是 1000**——平仓时它
    仍是当初的买入成本，不是卖出所得（应是 1200）。用它会静默毁掉两处：平仓配对（卖出所得
    被当成买入成本，于是盈亏退化成「只等于手续费」、**胜率恒为 0**）与换手率。

    这个坑一度真实存在，且被测试夹具掩盖着：夹具按 ``size × price`` 造数（与实现该有的口径
    一致），而引擎产出的是 ``executed.value``——**测试与实现各说各话**，所以没人发现。
    """

    class BuyThenSell(bt.Strategy):
        def next(self):
            if len(self) == 1:
                self.buy(size=100)
            elif len(self) == 3:
                self.close()

    # 第 2 根 open 10（买入成交），第 4 根 open 12（卖出成交）。
    prices = pd.DataFrame(
        {
            "open": [10.0, 10.0, 12.0, 12.0],
            "high": [10.0, 12.0, 12.0, 12.0],
            "low": [10.0, 10.0, 12.0, 12.0],
            "close": [10.0, 12.0, 12.0, 12.0],
            "volume": [1000, 1000, 1000, 1000],
        },
        index=pd.bdate_range("2024-01-02", periods=4),
    )

    result = run_backtest(
        prices, symbol="sh600000", strategy=BuyThenSell, cash=100_000.0, rules=zero_cost_rules
    )

    fills = result.trades.reset_index(drop=True)
    assert len(fills) == 2
    buy, sell = fills.iloc[0], fills.iloc[1]
    assert buy["price"] == 10.0 and buy["value"] == 1000.0
    assert sell["price"] == 12.0
    assert sell["value"] == -1200.0, "卖出是负的成交金额；若为 -1000 则记成了成本基础"


def test_a_profitable_round_trip_is_counted_as_a_win(make_prices, zero_cost_rules):
    """一次**赚钱**的往返必须被算成盈利——钉住上面那条口径的下游后果。

    在 ``value`` 记成成本基础时，这一笔的盈亏会被算成 0（甚至只剩手续费），于是胜率与盈亏比
    对**每一次回测**都是废数。这条测试从指标那一端倒推，确保口径真的接上了。
    """
    from mbt.metrics import closed_trade_pnls, compute_metrics

    class BuyThenSell(bt.Strategy):
        def next(self):
            if len(self) == 1:
                self.buy(size=100)
            elif len(self) == 3:
                self.close()

    prices = pd.DataFrame(
        {
            "open": [10.0, 10.0, 12.0, 12.0],
            "high": [10.0, 12.0, 12.0, 12.0],
            "low": [10.0, 10.0, 12.0, 12.0],
            "close": [10.0, 12.0, 12.0, 12.0],
            "volume": [1000, 1000, 1000, 1000],
        },
        index=pd.bdate_range("2024-01-02", periods=4),
    )

    result = run_backtest(
        prices, symbol="sh600000", strategy=BuyThenSell, cash=100_000.0, rules=zero_cost_rules
    )

    assert closed_trade_pnls(result.trades) == pytest.approx([200.0]), "买 100@10 卖 100@12 赚 200"
    metrics = compute_metrics(result.equity_curve, result.trades)
    assert metrics.win_rate == 1.0


def test_end_to_end_from_local_tdx_fixture(fixture_root, zero_cost_rules):
    """曳光弹：从本地通达信文件解析，一路跑到净值曲线与成交明细。"""
    prices = TdxDataSource(fixture_root).daily("sh600000")

    result = run_backtest(
        prices, symbol="sh600000", strategy=BuyOnce, cash=100_000.0, rules=zero_cost_rules
    )

    assert len(result.equity_curve) == len(prices)
    assert result.equity_curve.index[0] == prices.index[0]
    assert result.equity_curve.index[-1] == prices.index[-1]
    assert len(result.trades) == 1
    assert result.trades.iloc[0]["date"] == prices.index[1]
