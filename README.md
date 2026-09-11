# MyBacktrader

基于 [Backtrader](https://www.backtrader.com/) 与通达信本地数据解析的 A 股策略回测与选股工具。

术语以 [`CONTEXT.md`](CONTEXT.md) 为准；架构决策见 [`docs/adr/`](docs/adr/)；开发约定见
[`AGENTS.md`](AGENTS.md)。

## ⚠️ 当前状态：仍在施工，产出**不可用于策略判断**

本项目正在按票据逐张推进。当前已完成**端到端最窄切片**（票据
[#2](https://github.com/sunziming001/MyBacktrader/issues/2)）、交易制度约束
（票据 [#4](https://github.com/sunziming001/MyBacktrader/issues/4)）与**权息解密与复权视图**
（票据 [#3](https://github.com/sunziming001/MyBacktrader/issues/3)），
**还不是一个可用的回测工具**。具体地，它目前：

- **复权视图已实现**（票据 [#3](https://github.com/sunziming001/MyBacktrader/issues/3)）：自研解密本地权息
  文件 `gbbq`（不引入运行期第三方解析库，见 ADR-0008），按标准除权除息公式算出复权因子，
  提供**后复权**（回测用）与**前复权**（展示用）两种视图，事件按**评估日**过滤以防前视偏差。
  但**回测不会自动复权**——`run_backtest` 用的就是传给它的价格，要复权须显式套用后复权视图
  （见「用法」）。该票据另行承接的**行情异常检测尚未实现**（见下）。
- **A 股交易制度约束已实现**（票据 [#4](https://github.com/sunziming001/MyBacktrader/issues/4)）：
  - **费用侧已完成**：手续费、**印花税**（仅卖出方）、**过户费**、**经手费**、**证管费**、
    滑点，且费率按**成交日**查制度规则表（2023-08-28 印花税减半与经手费下调、
    2022-04-29 过户费降费等变更因此生效）。经手费与证管费是否叠加由
    `commission_mode` 决定，见下。
  - **撮合侧已完成**：**T+1**（当日买入不可当日卖出）、**涨跌停不可成交**、
    **停牌/无成交量不可成交**，且不可成交时挂单**保留至下一可交易日**；
    涨跌停判定按标的当日**板块与 ST 状态**取限幅，并区分方向——涨停一字买不进但卖得出，
    跌停一字卖不出但买得到。
  - **出厂规则表已落地**：`src/mbt/rules/a_share.toml`，覆盖 2015–2026 的涨跌幅、
    ST 限幅、印花税、过户费、经手费、证管费，每条数值都带出处。**但 ST 限幅目前实际选不中**（见下）。
- **没有选股能力**（待票据 [#7](https://github.com/sunziming001/MyBacktrader/issues/7)），
  也不会评估基准、回撤等指标。

用当前代码得出的任何收益数字都只能视为**管道连通性的证据**，不能作为策略好坏的依据。

> **与 ADR-0002 的冲突收窄到一处，且这一处现已具备实施前提**：[ADR-0002](docs/adr/0002-a-share-trading-constraints.md)
> 决策「回测从第一天就实现 A 股交易制度约束」。费用、撮合与复权均已落地；剩余的行情**异常检测**
> （ADR-0005）此前无法先于复权完成——同批数据里 460 次超 10% 的日间变动中 444 次合法，
> 照字面「涨跌幅超限即报错」会把正常交易日全数误报。**复权落地后已能区分「除权除息」与
> 「坏数据」，故它现在可实施了**，但仍**尚未实现**：在它落地之前，`run_backtest` 对坏数据
> 会**沉默接受**，其输出仍不得用于策略优劣的判断。

> **ST 状态的已知局限（本工具当前最大的制度盲区）**：涨跌幅限制需要知道标的当日是否 ST
> （沪深主板 ST 为 5%，自 2026-07-06 起上调为 10%；创业板 ST 自 2020-08-24 起为 20%），
> 但本地价格数据不含股票名称，无法回溯历史上的 ST 状态。规则表因此以
> `[[st_period]]` 显式登记 ST 期间，**未登记即视为非 ST**。
>
> 出厂表**不登记任何 ST 期间**，故那些 ST 限幅行目前**永远不会被选中**——一律按非 ST 取值。
> 这是宁漏不错的取舍：漏登记会让限幅偏大（放过本不该成交的交易），而不是凭空造出停牌。
> 等到带生效日期的 ST 名称历史到位，出厂表里的 ST 限幅即自动生效。
> 详见 [`docs/research/tdx-halt-and-limit-representation.md`](docs/research/tdx-halt-and-limit-representation.md)。

> **成本口径必须自己声明（否则会算错）**：交易成本由**手续费 + 印花税 + 过户费 +
> 经手费 + 证管费**构成，其中**经手费与证管费总是发生**（2023-08-28 后合计单边
> 0.0054%，双向约 0.011%），但券商的两种报价口径下处理方式**相反**：
>
> - 报「**全佣**」时，经纪费率里已含这两项 → 用 `commission_mode="all_in"`，不叠加；
> - 报「**净佣**」时未含 → 用 `commission_mode="net"`，按成交日叠加。
>
> 因此 `commission > 0` 时**必须**指定 `commission_mode`，否则 `run_backtest` 直接报错——
> 不替你猜，因为猜错的两个方向分别会导致成本静默高估或低估。
> 逐条费率与出处见 [`docs/research/a-share-trading-rules.md`](docs/research/a-share-trading-rules.md) §1.7、§1.8。

> **仍低估成本的少数项**：**证券结算风险基金**（十万分之三，双向，由券商缴纳，通常已计入
> 其成本）与**大宗交易费率下浮**（沪深按标准费率下浮 30%）未建模——前者不影响投资者费用单，
> 后者本层不区分交易方式。这两项的量级远小于经手费与证管费，但方向同为偏乐观。

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
# 原始价：此处仅演示管道连通；真实回测请先套用后复权视图，见下文「复权」。


class BuyAndHold(bt.Strategy):
    def next(self):
        if not self.position:
            self.buy()


# 出厂规则表已落地（src/mbt/rules/a_share.toml），故 rules 可省略；此处显式传入
# 只是示范「可注入自定义规则表」这一测试缝。
result = run_backtest(
    prices,
    symbol="sh600000",
    strategy=BuyAndHold,
    cash=100_000.0,
    rules=r"D:\rules\a_share.toml",
    commission=0.0003,       # 手续费率（券商约定）
    commission_min=5.0,      # 单笔最低手续费
    commission_mode="all_in",  # 该费率是「全佣」（已含经手费与证管费）；「净佣」用 "net"
    slippage=0.002,          # 滑点
)

result.equity_curve  # 净值曲线：交易日为索引，组合总资产为值
result.trades        # 成交明细：每笔成交一行
result.final_value   # 期末总资产
```

### 复权（回测前应做）

`run_backtest` **不会自动复权**——它用的就是传给它的价格。除权日不复权会在回测里留下
假跳空，故**回测应传后复权视图**：

```python
from mbt.data import GbbqDataSource, TdxDataSource, backward_adjusted

prices = TdxDataSource(r"D:\Tools\tdx\vipdoc").daily("sh600000")
# 权息文件位于 T0002\hq_cache\，不在 vipdoc 之内，故单独注入路径
events = GbbqDataSource(r"D:\Tools\tdx\T0002\hq_cache\gbbq").events("sh600000")

prices = backward_adjusted(prices, events)   # 后复权：回测用
# forward_adjusted(prices, events)           # 前复权：只用于展示（最新价不变）

result = run_backtest(prices, symbol="sh600000", strategy=BuyAndHold)
```

- 两种视图都**用时计算**，不落盘任何已复权的价格序列（ADR-0003）——前复权价会随未来每一次
  除权整体重算，存下来就不可复现。
- 事件按**评估日**过滤，默认取序列末根 K 线；「已公告、未除权」的未来事件一律不纳入（ADR-0006）。
- **序列之外的除权事件不参与**：要算它的因子，需要的前收盘价不在序列里；而它对整条序列
  只贡献一个常数倍数。故后复权视图的绝对尺度是相对于**序列首根**的，不必与行情软件一致
  ——与行情软件可比的是前复权视图。
- 解密为**自研**，不引入运行期第三方解析库；与独立 oracle（`pytdx`）对账全表 199,800 条
  逐字段一致，该 oracle **仅作测试对照**、不是依赖（ADR-0008）。

### 交易制度规则表

制度参数以「生效日期 × 板块」的规则表建模，查表按**成交日**取当日生效的那一条
（ADR-0002）。规则表是**可配置数据文件**，新制度到来时不必改代码：

- 出厂表：`src/mbt/rules/a_share.toml`——已落地，**每条数值都带出处**
  （2015–2026 的涨跌幅、ST 限幅、印花税、过户费），逐条来源与不确定项见
  `docs/research/a-share-trading-rules.md`。数值改动请连同出处一起改。
- 测试缝：`run_backtest(..., rules=<路径>)` 可注入任意规则表，测试因此指向
  `tests/fixtures/rules/` 下的夹具，不依赖出厂数值。出厂表自身由
  `tests/test_shipped_rules.py` 按变更日边界逐条锁定。

## 测试

```powershell
.venv\Scripts\python.exe -m pytest          # 全量
.venv\Scripts\python.exe -m pytest -m "not realmdata"   # 不依赖本机数据
```

测试自足：所有 fixture 都提交在 `tests/fixtures/` 下，不依赖本机通达信安装路径——
含日线 `.day` 与权息 `gbbq` 的**真实文件切片**，后者另附 `pytdx` 生成的 `records_oracle.csv`
逐字段对账（oracle 只在开发期跑过，运行测试只读这份固化结果）。
标记为 `realmdata` 的测试使用本机真实数据验证格式假设，数据缺席时自动跳过
（可用 `MBT_TDX_ROOT` 指定 `vipdoc` 路径、`MBT_TDX_GBBQ` 指定 `gbbq` 文件路径）。
