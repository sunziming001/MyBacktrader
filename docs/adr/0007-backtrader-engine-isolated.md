# 回测引擎用 backtrader 1.9.78.123，隔离在自有接口之后

回测引擎采用 `backtrader` **1.9.78.123**（2023-04-19 发布，界面冻结），并在 `mbt.backtest` 内以自有协议（Protocol）隔离。数据层与信号层**不得**依赖 backtrader，它们只使用 pandas。

## Context

实测事实：`backtrader` 上游自 2023-04-19 起 `master` 无提交、无 GitHub Release、63 个未关 issue。但在目标环境（Python 3.10.6 + pandas 2.3.3 + numpy 2.2.6）上**实测跑通**——`PandasData`、`PandasDirectData`、周线重采样、日期作列/作索引全部通过。pandas 2.0 移除的 `iteritems` / `append` / `Int64Index`，其 feed 源码一个都未调用；核心引擎不 import numpy。全部 256 个 issue/PR 中无一条提及 pandas 2 或 numpy 2——**上游无需修，因为它本就不受影响**。

唯一实测缺陷是 Python 3.10 移除 `collections.Iterable` 导致 `lineiterator.py` 的 `bindlines()` / `bind2lines()` 抛 `AttributeError`（修复存在于仍未合并的 PR #520）。普通回测不走该路径。

## Considered Options

- **backtrader + 隔离**（采纳）
- **`cloudQuant/backtrader` 活跃 fork**（v1.3.0, 2026-07-26）：维护活跃，但**未实测跑通**（源码下载超时）且**许可证未核实**。
- **自研事件驱动引擎**：最贵，且本项目价值在信号层与选股（ADR-0001），不在重写 Cerebro。
- **`vectorbt`**（否决）：**范式错配**，不只是次优。ADR-0002 要求的 T+1、涨跌停不可成交、停牌跳过全是路径依赖的顺序约束，向量化引擎天生难表达；且 `vectorbt` ≥1.1.0 要求 Python ≥3.11，会连带放弃成熟的 3.10 数据分析生态。

## 决策

采用 `backtrader` 1.9.78.123，**精确 pin**（不用 `~=`，停更库的任何版本变化都值得一次人工确认），并隔离在 `mbt.backtest` 的自有协议之后。

**明知停更仍然选择它，是刻意的**——读到这里的未来读者（包括作者本人）会本能地想「换掉这个三年未更新的库」，此 ADR 记录的正是「为什么明知如此还这么选」，以及爆炸半径被限制在哪里。

Python 3.10 的 `collections.Iterable` 缺陷以**包内兼容 shim** 修补，**并配一个真正调用 `bindlines` 的测试**——没有测试的 monkeypatch 会在依赖升级后静默失效。

## Consequences

核心收益是所有权边界：数据层与信号层只用 pandas，其稳定性比 backtrader 高一个数量级；Q10 的黄金用例因此写在**我们自己的接口**上，将来更换引擎时测试全部存活。

代价是永远无法向上游提 bug 期望修复，且需自行承担 Python 版本迁移风险（如将来升到 3.13 需重新验证）。
