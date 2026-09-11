"""验证单标的端到端最窄切片：价格表 → 回测 → 净值曲线与成交明细。

测试缝 S1：只走库公开入口。**刻意不使用假引擎**——本测试要验证的正是
「真实 backtrader + 我们的接线」这个组合；换成假引擎，测的就只剩我们自己的记账，
而漏掉最可能出现意外的地方（ADR-0007）。

期望值来自 backtrader 的成交语义（在 ``next()`` 中下的单于**次根 K 线开盘**成交）
加手工算术，而非由被测代码产出。
"""

import backtrader as bt
import pandas as pd

from mbt.backtest import run_backtest
from mbt.data import TdxDataSource


class BuyOnce(bt.Strategy):
    """最小策略：没有持仓就买入，此后不动。"""

    def next(self):
        if not self.position:
            self.buy()


def test_equity_curve_is_hand_computable(make_prices):
    """每根 K 线开高低收相同，故净值可手算。

    资金 1000，默认下单量为 1 股：次根 K 线以 11 成交，占用 11，余 989。
    净值依次为 1000（未持仓）、989+11=1000、989+12=1001、989+13=1002。
    """
    prices = make_prices([10.0, 11.0, 12.0, 13.0])

    result = run_backtest(prices, strategy=BuyOnce, cash=1000.0)

    assert list(result.equity_curve.index) == list(prices.index)
    assert result.equity_curve.tolist() == [1000.0, 1000.0, 1001.0, 1002.0]
    assert result.final_value == 1002.0


def test_trades_record_every_fill(make_prices):
    prices = make_prices([10.0, 11.0, 12.0, 13.0])

    result = run_backtest(prices, strategy=BuyOnce, cash=1000.0)

    trades = result.trades
    assert len(trades) == 1
    fill = trades.iloc[0]
    assert fill["date"] == pd.Timestamp("2024-01-03")
    assert fill["size"] == 1
    assert fill["price"] == 11.0
    assert fill["value"] == 11.0


def test_trade_columns_are_exactly_the_documented_contract(make_prices):
    """成交明细的列就是公开契约 ``TRADE_COLUMNS``——不多、不少、顺序一致。"""
    from mbt.backtest.engine import TRADE_COLUMNS

    result = run_backtest(make_prices([10.0, 11.0]), strategy=BuyOnce, cash=1000.0)

    assert list(result.trades.columns) == list(TRADE_COLUMNS)


def test_end_to_end_from_local_tdx_fixture(fixture_root):
    """曳光弹：从本地通达信文件解析，一路跑到净值曲线与成交明细。"""
    prices = TdxDataSource(fixture_root).daily("sh600000")

    result = run_backtest(prices, strategy=BuyOnce, cash=100_000.0)

    assert len(result.equity_curve) == len(prices)
    assert result.equity_curve.index[0] == prices.index[0]
    assert result.equity_curve.index[-1] == prices.index[-1]
    assert len(result.trades) == 1
    assert result.trades.iloc[0]["date"] == prices.index[1]
