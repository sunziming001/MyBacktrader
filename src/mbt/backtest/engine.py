"""回测：把价格表交给引擎，取回净值曲线与成交明细。

引擎被隔离在本模块之后（ADR-0007）：数据层与信号层不依赖 backtrader。
"""

from __future__ import annotations

from dataclasses import dataclass

import backtrader as bt
import pandas as pd

#: 成交明细的列。
TRADE_COLUMNS = ("date", "size", "price", "value", "commission")


@dataclass(frozen=True)
class BacktestResult:
    """一次回测的产物。

    属性:
        equity_curve: 净值曲线，交易日为索引、组合总资产为值。
        trades: 成交明细，每笔成交一行（买入为正、卖出为负）。
        final_value: 期末总资产。
    """

    equity_curve: pd.Series
    trades: pd.DataFrame
    final_value: float


class _EquityRecorder(bt.Analyzer):
    """逐根 K 线记录组合总资产。"""

    def start(self):
        self._dates = []
        self._values = []

    def next(self):
        self._dates.append(self.datas[0].datetime.date(0))
        self._values.append(self.strategy.broker.getvalue())

    def get_analysis(self):
        return pd.Series(self._values, index=pd.DatetimeIndex(self._dates), name="value")


class _FillRecorder(bt.Analyzer):
    """记录每一笔实际成交。"""

    def start(self):
        self._fills = []

    def notify_order(self, order):
        if order.status != order.Completed:
            return
        self._fills.append(
            {
                "date": bt.num2date(order.executed.dt),
                "size": order.executed.size,
                "price": order.executed.price,
                "value": order.executed.value,
                "commission": order.executed.comm,
            }
        )

    def get_analysis(self):
        return pd.DataFrame(self._fills, columns=list(TRADE_COLUMNS))


def run_backtest(prices, strategy, cash=100_000.0, commission=0.0, **strategy_params):
    """对单一标的的价格表跑一次回测。

    参数:
        prices: 交易日为索引、含 ``open`` / ``high`` / ``low`` / ``close`` / ``volume``
            的价格表。传入**原始价**会得到未复权的结果——复权视图由复权层提供。
        strategy: ``backtrader.Strategy`` 的子类。
        cash: 期初资金。
        commission: 手续费率，一期为单一费率。

    .. warning::

        本函数是本项目的一期曳光弹，**尚不可用于策略判断**：它既不做复权，
        也不实现 A 股交易制度约束（T+1、涨跌停不可成交、停牌、印花税）。
        见票据 02 与 03。
    """
    cerebro = bt.Cerebro()
    cerebro.adddata(bt.feeds.PandasData(dataname=prices))
    cerebro.addstrategy(strategy, **strategy_params)
    cerebro.broker.setcash(cash)
    cerebro.broker.setcommission(commission=commission)
    cerebro.addanalyzer(_EquityRecorder, _name="equity")
    cerebro.addanalyzer(_FillRecorder, _name="fills")

    runs = cerebro.run()
    run = runs[0]

    return BacktestResult(
        equity_curve=run.analyzers.equity.get_analysis(),
        trades=run.analyzers.fills.get_analysis(),
        final_value=cerebro.broker.getvalue(),
    )
