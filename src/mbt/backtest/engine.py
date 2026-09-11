"""回测：把价格表交给引擎，取回净值曲线与成交明细。

引擎被隔离在本模块之后（ADR-0007）：数据层与信号层不依赖 backtrader。
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import backtrader as bt
import pandas as pd

from mbt.rules import RuleTable

from .costs import AStockBroker, AStockCommissionInfo

#: 出厂规则表路径。每条数值都带出处，见该文件内的注释。
DEFAULT_RULES_PATH = Path(__file__).resolve().parent.parent / "rules" / "a_share.toml"

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
        executed = order.executed
        # 由 TRADE_COLUMNS 派生键，列改名时不会静默产生 NaN 列；
        # strict=True 让字段数与列数不符时立即报错，而非静默截断
        self._fills.append(
            dict(
                zip(
                    TRADE_COLUMNS,
                    (
                        bt.num2date(executed.dt),
                        executed.size,
                        executed.price,
                        executed.value,
                        executed.comm,
                    ),
                    strict=True,
                )
            )
        )

    def get_analysis(self):
        return pd.DataFrame(self._fills, columns=list(TRADE_COLUMNS))


def run_backtest(
    prices,
    symbol,
    strategy,
    cash=100_000.0,
    rules=None,
    commission=0.0,
    commission_min=0.0,
    slippage=0.0,
    **strategy_params,
):
    """对单一标的的价格表跑一次回测。

    参数:
        prices: 交易日为索引、含 ``open`` / ``high`` / ``low`` / ``close`` / ``volume``
            的价格表。传入**原始价**会得到未复权的结果——复权视图由复权层提供。
        symbol: 标的符号，形如 ``sh600000``。交易制度按它取板块与 ST 状态，
            因此**必填**——缺了它就无法确定过户费与涨跌幅限制，费用会算错。
        strategy: ``backtrader.Strategy`` 的子类。
        cash: 期初资金。
        rules: 规则表，可以是 ``RuleTable`` 或 TOML 文件路径。默认取出厂规则表，
            但**出厂规则表尚未落地**（票据 04 切片 8，等制度数值查证），故目前
            请显式传入路径。
        commission: 手续费率（券商约定，如 0.0003）。
        commission_min: 单笔最低手续费。
        slippage: 滑点比例，成交价按此劣化。

    .. warning::

        本函数**尚不可用于策略判断**：它不做复权（待票据 03）。A 股制度约束已完成
        费用、T+1、涨跌停不可成交、无成交量不可成交（票据 04 切片 1–6），但仍缺：

        - 行情**异常检测**（切片 7）：单日涨跌幅超出制度上限目前不会报错，
          而 ADR-0005 要求异常报错；
        - 出厂规则表（切片 8），故现在必须显式传入 ``rules``。

        未复权意味着除权跳空会原样进入回测，收益结论仍偏乐观。
    """
    table = rules if isinstance(rules, RuleTable) else RuleTable.load(rules or DEFAULT_RULES_PATH)

    cerebro = bt.Cerebro()

    data = bt.feeds.PandasData(dataname=prices)
    data._mbt_symbol = symbol
    data._name = symbol
    cerebro.adddata(data)
    cerebro.addstrategy(strategy, **strategy_params)

    broker = AStockBroker(rules=table)
    broker.setcash(cash)
    broker.addcommissioninfo(
        AStockCommissionInfo(rules=table, commission=commission, commission_min=commission_min),
        name=symbol,
    )
    if slippage:
        broker.set_slippage_perc(slippage)
    cerebro.setbroker(broker)

    cerebro.addanalyzer(_EquityRecorder, _name="equity")
    cerebro.addanalyzer(_FillRecorder, _name="fills")

    runs = cerebro.run()
    run = runs[0]

    return BacktestResult(
        equity_curve=run.analyzers.equity.get_analysis(),
        trades=run.analyzers.fills.get_analysis(),
        final_value=cerebro.broker.getvalue(),
    )
