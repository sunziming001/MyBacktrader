# MyBacktrader

基于 [Backtrader](https://www.backtrader.com/) 与通达信本地数据解析的 A 股策略回测与选股工具。

术语以 [`CONTEXT.md`](CONTEXT.md) 为准；架构决策见 [`docs/adr/`](docs/adr/)；开发约定见
[`AGENTS.md`](AGENTS.md)。

## ⚠️ 当前状态：仍在施工，产出**不可用于策略判断**

本项目正在按票据逐张推进。当前已完成**端到端最窄切片**（票据
[#2](https://github.com/sunziming001/MyBacktrader/issues/2)）与交易制度约束
（票据 [#4](https://github.com/sunziming001/MyBacktrader/issues/4) 收尾中），
**还不是一个可用的回测工具**。具体地，它目前：

- **不做复权**（待票据 [#3](https://github.com/sunziming001/MyBacktrader/issues/3)）。解析器返回原始价，
  除权除息造成的价格跳空会原样进入回测，因而会得到错误的收益。
- **A 股交易制度约束已基本实现**（票据 [#4](https://github.com/sunziming001/MyBacktrader/issues/4) 收尾中）：
  - **费用侧已完成**：手续费、**印花税**（仅卖出方）、**过户费**、滑点，且费率按**成交日**查制度规则表
    （2023-08-28 印花税减半、2022-04-29 过户费沪深统一等变更因此生效）。
  - **撮合侧已完成**：**T+1**（当日买入不可当日卖出）、**涨跌停不可成交**、
    **停牌/无成交量不可成交**，且不可成交时挂单**保留至下一可交易日**；
    涨跌停判定按标的当日**板块与 ST 状态**取限幅，并区分方向——涨停一字买不进但卖得出，
    跌停一字卖不出但买得到。
  - **尚缺**：行情异常检测——它已**移出 #4、并入票据 [#3](https://github.com/sunziming001/MyBacktrader/issues/3)**，
    因为区分「公司行为（除权送转）」与「坏数据」必须知道复权信息（见下），
    以及**出厂规则表**——所以现在必须显式传 `rules=`，见下方示例。
- **没有选股能力**（待票据 [#7](https://github.com/sunziming001/MyBacktrader/issues/7)），
  也不会评估基准、回撤等指标。

用当前代码得出的任何收益数字都只能视为**管道连通性的证据**，不能作为策略好坏的依据。

> **与 ADR-0002 的冲突已收窄到一处，但那一处不在本票范围内**：[ADR-0002](docs/adr/0002-a-share-trading-constraints.md)
> 决策「回测从第一天就实现 A 股交易制度约束」。费用与撮合现已落地；剩余的行情**异常检测**
> （ADR-0005）经实测判定**无法先于复权完成**——同批数据里 460 次超 10% 的日间变动中 444 次
> 合法，照字面「涨跌幅超限即报错」会把正常交易日全数误报，故已移入票据
> [#3](https://github.com/sunziming001/MyBacktrader/issues/3)。在异常检测落地之前，
> `run_backtest` 对坏数据会**沉默接受**，故其输出仍不得用于策略优劣的判断。

> **ST 状态的已知局限**：涨跌幅限制需要知道标的当日是否 ST（主板 ST 为 5%），
> 但本地价格数据不含股票名称，无法回溯历史上的 ST 状态。规则表因此以
> `[[st_period]]` 显式登记 ST 期间，**未登记即视为非 ST**。这是宁漏不错的取舍：
> 漏登记会让限幅偏大（放过本不该成交的交易），而不是凭空造出停牌。
> 详见 [`docs/research/tdx-halt-and-limit-representation.md`](docs/research/tdx-halt-and-limit-representation.md)。

完整的范围与验收标准见规格 [issue #1](https://github.com/sunziming001/MyBacktrader/issues/1)。

## 安装

需要 Python 3.10。

```powershell
python -m venv .venv
.venv\Scripts\python.exe -m pip install -e ".[dev]"
```

`backtrader` 被精确 pin 到 `1.9.78.123`（上游自 2023 年起停更），不要放宽为范围约束（ADR-0007）。

## 用法

数据源根路径即通达信安装目录下的 `vipdoc`。`symbol` 是必填的——交易制度按它取板块与
ST 状态，缺了它过户费与涨跌幅限制就无从确定。

```python
import backtrader as bt

from mbt.backtest import run_backtest
from mbt.data import TdxDataSource

prices = TdxDataSource(r"D:\Tools\tdx\vipdoc").daily("sh600000")


class BuyAndHold(bt.Strategy):
    def next(self):
        if not self.position:
            self.buy()


# 出厂规则表尚未落地（票据 #4 切片 8），故当前必须显式传入规则表路径。
# 该文件写好后此参数可省略。
result = run_backtest(
    prices,
    symbol="sh600000",
    strategy=BuyAndHold,
    cash=100_000.0,
    rules=r"D:\rules\a_share.toml",
    commission=0.0003,       # 手续费率（券商约定）
    commission_min=5.0,      # 单笔最低手续费
    slippage=0.002,          # 滑点
)

result.equity_curve  # 净值曲线：交易日为索引，组合总资产为值
result.trades        # 成交明细：每笔成交一行
result.final_value   # 期末总资产
```

### 交易制度规则表

制度参数以「生效日期 × 板块」的规则表建模，查表按**成交日**取当日生效的那一条
（ADR-0002）。规则表是**可配置数据文件**，新制度到来时不必改代码：

- 出厂表：`src/mbt/rules/a_share.toml`——**尚未落地**，其数值须逐条查证出处
  （票据 [#4](https://github.com/sunziming001/MyBacktrader/issues/4) 切片 8）。
  目前传入不存在的路径会**明确报错**，而不是退回某个默认限幅（ADR-0005）。
- 测试缝：`run_backtest(..., rules=<路径>)` 可注入任意规则表，测试因此指向
  `tests/fixtures/rules/` 下的夹具，不依赖出厂数值。撮合约束的回归样本见
  `tests/test_trading_constraints.py`，用的是仓库内的真实行情 fixture。

## 测试

```powershell
.venv\Scripts\python.exe -m pytest          # 全量
.venv\Scripts\python.exe -m pytest -m "not realmdata"   # 不依赖本机数据
```

测试自足：所有 fixture 都提交在 `tests/fixtures/` 下，不依赖本机通达信安装路径。
标记为 `realmdata` 的测试使用本机真实数据验证格式假设，数据缺席时自动跳过
（可用 `MBT_TDX_ROOT` 指定路径）。
