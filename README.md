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
  （见「用法」）。
- **行情异常检测已实现**（同上票据）：把「越出涨跌停带的跳空」区分为**公司行为**与
  **坏数据**——有除权除息事件可解释的放过，无事件解释的报错（ADR-0005）。判据是比
  **限价带**而不是比涨跌幅比率：实测 460 次超 10% 的日间变动中 **444 次合法**（含四舍五入
  效应与「收在涨停但盘中未锁死」），照字面判比率会把正常交易日全数误报。它由数据层入口
  `load_market_data` **强制**执行；`run_backtest` 只管撮合一张给定的表，不负责质检。
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
- **信号层已实现**（票据 [#5](https://github.com/sunziming001/MyBacktrader/issues/5)）：项目的架构中枢。
  指标、过滤信号、排序因子一律算成日期 × 标的的**标的宽表**，回测按列取时序、选股按行取截面
  ——同一个函数，两个消费方向（ADR-0001）。全部是纯函数：无 I/O、无状态、不读全局配置。
  - 指标：`sma`、`rolling_max`、`volume_ratio`（成交量比）、`atr`（Wilder 口径，见下）
  - 过滤信号：`new_high`、`volume_surge`、`ma_cross_up`、`rising_streak`、`above_ma`
  - 排序因子：`momentum`、`distance_to_high`
  - **行情面板已实现**（票据 [#14](https://github.com/sunziming001/MyBacktrader/issues/14)）：
    `Panel` + `assemble_panel`。ATR 一行内要用 `high` / `low` / `close` 且必须同日对齐，而标的
    宽表一格只能放一个数，面板补上这一层。它是**按需组装**的显式结构，不落盘、不常驻；字段
    由调用方显式声明，组装时强制各字段同日对齐。正名与理由见
    [ADR-0009](docs/adr/0009-panel-and-frame-naming.md)：裸用「宽表」一词在本项目里被禁止，
    因为**字段宽表**与**标的宽表**形状同形而语义相反。
  - 术语提醒：`volume_ratio` **不是**交易所口径的「量比」（后者是盘中、5 日、每分钟口径，
    日线无法忠实复现），本项目刻意不实现量比、也不用那个词。

- **组合回测与股票池已实现**（票据 [#6](https://github.com/sunziming001/MyBacktrader/issues/6)）：
  - `run_portfolio_backtest` 收一组已质检行情，多标的**共享资金池**，有最大持仓数与持仓分配
    （默认等权：每个持仓名额拿到等额的**目标仓位**，见 `mbt.backtest.sizing`）。单标的入口
    `run_backtest` 是它的退化情形，**不存在第二套引擎路径**。
  - **股票池**是 boolean 标的宽表，按每个调仓日的当日状态计算：品种只收股票、板块可配置、
    默认排除次新股（`mbt.universe`）。撮合层**强制**池外不可买、**卖出不受限**。
  - **品种判定**（`mbt.data.instrument_type`）按市场 + 代码前缀区分股票 / 指数 / 可转债 /
    基金 / 其他；判不出归「其他」而**不报错**（错在「不收」这个安全方向）。
- **没有选股能力**（待票据 [#7](https://github.com/sunziming001/MyBacktrader/issues/7)），
  也不会评估基准、回撤等指标。

用当前代码得出的任何收益数字都只能视为**管道连通性的证据**，不能作为策略好坏的依据。

> **与 ADR-0002 的冲突已消除**：[ADR-0002](docs/adr/0002-a-share-trading-constraints.md)
> 决策「回测从第一天就实现 A 股交易制度约束」。费用、撮合、复权与行情异常检测均已落地。
>
> 一处取舍要说明：异常检测放在**数据层入口** `load_market_data` 上，而不是 `run_backtest`
> 内部。原因是判定「跳空是公司行为还是坏数据」需要权息事件，而 `run_backtest` 拿到的只是
> 一张价格表，无从判断；更糟的是，若它对照后复权价做这件检查，就会把复权已经抹平的跳空
> 再折算一次，**凭空造出假异常**（详见 `src/mbt/data/market.py` 的模块说明）。故「质检」
> 与「撮合」分层：`load_market_data` 的返回值只可能是已质检的原始价，`run_backtest` 则是
> 哑引擎。绕过入口直接构造价格表，就绕过了质检。

> **ST 状态的已知局限（本工具当前最大的制度盲区）**：涨跌幅限制需要知道标的当日是否 ST
> （沪深主板 ST 为 5%，自 2026-07-06 起上调为 10%；创业板 ST 自 2020-08-24 起为 20%），
> 但本地价格数据不含股票名称，无法回溯历史上的 ST 状态。规则表因此以
> `[[st_period]]` 显式登记 ST 期间，**未登记即视为非 ST**。
>
> 出厂表**不登记任何 ST 期间**，故那些 ST 限幅行目前**永远不会被选中**——一律按非 ST 取值。
> 这是宁漏不错的取舍：漏登记会让限幅偏大（放过本不该成交的交易），而不是凭空造出停牌。
> 等到带生效日期的 ST 名称历史到位，出厂表里的 ST 限幅即自动生效。
> 详见 [`docs/research/tdx-halt-and-limit-representation.md`](docs/research/tdx-halt-and-limit-representation.md)。

> **数据早于规则表窗口时会报错，而不是静默放过**：出厂表的涨跌幅覆盖自 1996-12-16（沪深主板；
> 该日之前是「无涨跌幅 / 5% / 1% 等」的多段试验期）与 2021-11-15（北交所；其前身新三板精选层
> 自 2020-07-27 起同为 30%，但过户费与经手费口径不同，故本表不把两者合并）。早于此的 K 线
> 会让 `load_market_data` 抛 `RuleTableError`——实测本机 5,602 只股票中 186 只如此（几乎全是
> 北交所 2021-11-15 之前的历史）。要回测这些标的的早期区间，请显式切片；本工具不会替你「跳过」
> 无法判定的日子。

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

> **买入未按「一手 = 100 股」整手取整**：持仓分配算出的股数可能不是 100 的倍数，而现实中
> 买入必须是 100 股的整数倍。这是一处**已知的乐观偏差**（真实下单会被拒或改单），量级与
> 「一手」相对单个标的目标仓位的比例同阶，未建模。

> **「上市不足 60 个交易日」=「本地行情不足 60 根 K 线」**：股票池的次新股门槛度量的是
> **本地数据的可得量**，不是真实上市日。对本机数据窗口（约 2015-01 起）之前上市的老股，
> 其首根即窗口起点，故该门槛永不触发——结果正确（它们确实不是次新），但它**不等于**
> 「自 IPO 起已满 60 日」。真实上市日需要数据源提供带日期的上市信息，一期不含。

> **停牌日不可成交，且挂单会失效**：多标的回测中，停牌标的的 K 线是**陈旧**的——若只看
> 行情本身，`volume` 看着完全正常，订单会按停牌前的价格成交。撮合层因此用一张逐日的
> **可交易掩码**判定「这根 K 线是不是今天的」，标的连续无 K 线超过 `order_expiry_ticks`
> （默认 5）个交易日后挂单作废。策略读 `data.close[0]` 做决策前应先查
> `self.broker.tradability_mask`。

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

取数请走**正门** `load_market_data`：它把行情、权息事件与规则表组合起来，**并在返回前做完
越界检查**（ADR-0005）——坏数据在这里报错，而不是被静默喂进回测。

```python
import backtrader as bt

from mbt.backtest import run_backtest
from mbt.data import load_market_data

# 权息文件位于 T0002\hq_cache\，不在 vipdoc 之内，故单独注入路径
market = load_market_data(
    "sh600000",
    tdx_root=r"D:\Tools\tdx\vipdoc",
    gbbq_path=r"D:\Tools\tdx\T0002\hq_cache\gbbq",
)

prices = market.backward_adjusted()   # 后复权：回测用，除权日不再有假跳空


class BuyAndHold(bt.Strategy):
    def next(self):
        if not self.position:
            self.buy()


# 出厂规则表已落地（src/mbt/rules/a_share.toml），故 rules 可省略；此处显式传入
# 只是示范「可注入自定义规则表」这一测试缝。
result = run_backtest(
    prices,
    symbol=market.symbol,
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

`load_market_data` 只覆盖**股票**：指数、基金、债券没有涨跌幅限价制度，传进去会报错
（硬套股票的带才是错的）。全市场扫描若想「跳过坏标的、继续扫」，请在调用处捕获
`MarketDataError` 并记下该标的存疑，而不是让它以干净的名义混进样本。

### 行情异常检测（取数即做，无需额外调用）

判据是**比限价带**，不是比涨跌幅比率。两者看似等价，实测却差很多：本机 16 只主板股票的
43,982 个相邻交易日里，日间变动超 10% 的有 460 次，其中 **444 次合法**——既含四舍五入
效应（`11.26 × 1.1 = 12.386` 进位到 12.39，涨 10.04%），也含「收在涨停但盘中未锁死」。
照字面「涨跌幅超限即报错」会把正常交易日全数误报。故限价带用**十进制半进位**算到分，
且与撮合层共用同一个算法（`mbt.rules.limit_price`），不会出现「撮合认为没触限、质检
却报异常」。

归因则用权息事件解释跳空，而不是见到事件就豁免：除权除息日的限幅以**除权除息参考价**
为基数，故那一天的合法区间会整体下移。检查按「前一根 K 线之后、当根 K 线之前（含当根）」
取全量事件并**跨日链式**折算——长期停牌期间的权息事件在复牌日才体现为跳空，只查当根
K 线会漏掉它们。若没有事件而价格越界，或按事件重算的带仍被越出，则报错。

**已知误报源：制度空窗**。全市场实测（9,720 个文件、1,135 万根 K 线）检出 761 处越界，
**没有一处**是「无制度空窗可解释」的坏数据——761 处全部落在规则表已声明的「有意省略」里：

- **序列前 5 根**（601 处）：新股上市初期不设涨跌幅；
- **北交所**（76 处，集中在 2021–2023）：本地 `bj/` 目录混存北交所与新三板，`920xxx` 的历史
  回溯到其上市前的**新三板挂牌期**——那段时期没有硬性涨跌幅限制，只有 ±30%/±60% 的盘中
  临时停牌；
- **其余约 82 处**：股改复牌首日不设限（上证交字〔2005〕7 号）、重大资产重组/借壳复牌、
  退市整理期，以及缩股/债转股等**非除权除息类**公司行为（`gbbq` 类别 2 及以后，按 ADR-0003
  不参与价格复权，质检因此看不到它们）。这批**未逐例定案**——定案需要「上市日 / 退市日 /
  重组停复牌标记」，而 `.day` 不含。

因此这条判据的定位是**「报错让人看」，不是无人值守的批量过滤**：它不放过坏数据（限价若算错
会在除权日与涨跌停日大面积误报，而实测误报密度仅 0.0067%），但会把制度空窗误报成坏数据。
做全市场选股时，请在调用处捕获 `MarketDataError` 并把该标的记为**存疑**、再人工抽查，而不是
静默丢弃或当它干净。逐条数字与分类见
[`docs/research/tdx-halt-and-limit-representation.md`](docs/research/tdx-halt-and-limit-representation.md) 结论五的补记。

### 复权（回测前应做）

`run_backtest` **不会自动复权**——它用的就是传给它的价格。除权日不复权会在回测里留下
假跳空，故**回测应传后复权视图**（走正门时即 `market.backward_adjusted()`）。若你手上
已经有价格表与事件，也可直接用复权层的函数：

```python
from mbt.data import GbbqDataSource, TdxDataSource, backward_adjusted

prices = TdxDataSource(r"D:\Tools\tdx\vipdoc").daily("sh600000")
# 权息文件位于 T0002\hq_cache\，不在 vipdoc 之内，故单独注入路径
events = GbbqDataSource(r"D:\Tools\tdx\T0002\hq_cache\gbbq").events("sh600000")

prices = backward_adjusted(prices, events)   # 后复权：回测用
# forward_adjusted(prices, events)           # 前复权：只用于展示（最新价不变）

result = run_backtest(prices, symbol="sh600000", strategy=BuyAndHold)
```

> 这条路径**绕过了数据质检**（越界检查需要规则表）。它适用于你已自行确认过数据质量的
> 场合；一般情况下请用 `load_market_data`。

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
