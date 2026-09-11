# MyBacktrader

基于 [Backtrader](https://www.backtrader.com/) 与通达信本地数据解析的 A 股策略回测与选股工具。

术语以 [`CONTEXT.md`](CONTEXT.md) 为准；架构决策见 [`docs/adr/`](docs/adr/)；开发约定见
[`AGENTS.md`](AGENTS.md)。

## ⚠️ 当前状态：一期曳光弹，产出**不可用于策略判断**

本项目正在按票据逐张推进。当前已完成的是**端到端最窄切片**（票据
[#2](https://github.com/sunziming001/MyBacktrader/issues/2)），它只打通管道、暴露集成问题，
**还不是一个可用的回测工具**。具体地，它目前：

- **不做复权**（待票据 [#3](https://github.com/sunziming001/MyBacktrader/issues/3)）。解析器返回原始价，
  除权除息造成的价格跳空会原样进入回测，因而会得到错误的收益。
- **不实现 A 股交易制度约束**（待票据 [#4](https://github.com/sunziming001/MyBacktrader/issues/4)）。
  没有 T+1、没有涨跌停不可成交、没有停牌、没有印花税与过户费。
  这意味着策略可以做到现实中做不到的事，回测结果会**系统性偏乐观**。
- **没有选股能力**（待票据 [#7](https://github.com/sunziming001/MyBacktrader/issues/7)），
  也不会评估基准、回撤等指标。

用当前代码得出的任何收益数字都只能视为**管道连通性的证据**，不能作为策略好坏的依据。

> **与 ADR-0002 的冲突（显式声明）**：[ADR-0002](docs/adr/0002-a-share-trading-constraints.md)
> 决策「回测从第一天就实现 A 股交易制度约束」。当前代码**违背**该决策，这是刻意接受的：
> ADR-0002 约束的是**可用于策略判断的回测**，而本票产物按定义不是回测结论，只是管道证据。
> 制度约束由票据 [#4](https://github.com/sunziming001/MyBacktrader/issues/4) 补齐；
> 在它落地之前，`run_backtest` 的输出不得用于任何策略优劣的判断。

完整的范围与验收标准见规格 [issue #1](https://github.com/sunziming001/MyBacktrader/issues/1)。

## 安装

需要 Python 3.10。

```powershell
python -m venv .venv
.venv\Scripts\python.exe -m pip install -e ".[dev]"
```

`backtrader` 被精确 pin 到 `1.9.78.123`（上游自 2023 年起停更），不要放宽为范围约束（ADR-0007）。

## 用法

数据源根路径即通达信安装目录下的 `vipdoc`：

```python
import backtrader as bt

from mbt.backtest import run_backtest
from mbt.data import TdxDataSource

prices = TdxDataSource(r"D:\Tools\tdx\vipdoc").daily("sh600000")


class BuyAndHold(bt.Strategy):
    def next(self):
        if not self.position:
            self.buy()


result = run_backtest(prices, strategy=BuyAndHold, cash=100_000.0)

result.equity_curve  # 净值曲线：交易日为索引，组合总资产为值
result.trades        # 成交明细：每笔成交一行
result.final_value   # 期末总资产
```

## 测试

```powershell
.venv\Scripts\python.exe -m pytest          # 全量
.venv\Scripts\python.exe -m pytest -m "not realmdata"   # 不依赖本机数据
```

测试自足：所有 fixture 都提交在 `tests/fixtures/` 下，不依赖本机通达信安装路径。
标记为 `realmdata` 的测试使用本机真实数据验证格式假设，数据缺席时自动跳过
（可用 `MBT_TDX_ROOT` 指定路径）。
