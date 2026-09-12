"""指标：由净值曲线与成交明细算出可评估的数字（票据 #10）。

**纯函数层**：无 I/O、不读全局配置，与信号层同一种纪律，故它天然可测。渲染与落盘在
:mod:`mbt.report`。

## 口径即契约

AC 要求「每个指标的计算口径以公式写进文档，读者不必猜」——故每条口径都写在对应函数的
文档串里，且**有测试按独立的算式钉住**。三处最容易各算各的地方在此定死：

- 年化用**交易日数**、按**几何**方式；
- 夏普的无风险利率按**几何**折算到日频（不是除以 252），标准差用**样本**口径；
- 涨跌幅与回撤一旦没有波动或没有样本，返回**缺失值而不是无穷**——``inf`` 会污染比较与
  图表，而「算不出来」与「好得没边」是两件事。
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import pandas as pd

#: 一年的交易日数，年化口径的基准。A 股约 242–244，业界惯例用 252。
TRADING_DAYS_PER_YEAR = 252

#: 股数比较的容差。浮点累加后「剩余 0 股」往往是 1e-13 量级而非精确 0，故判「这一批是否
#: 已扣完」需要一个极小容差；它只用于**配对记账**，不参与任何价格或费用的计算。
_SHARE_EPSILON = 1e-9


@dataclass(frozen=True)
class Metrics:
    """一次回测的指标汇总。缺失一律是 ``nan``，原因见 :attr:`notes`。

    属性:
        total_return: 区间总收益 ``期末/期初 − 1``。
        annual_return: 几何年化收益（口径见 :func:`compute_metrics`）。
        sharpe: 年化夏普比率。
        max_drawdown: 最大回撤，**正值**表示深度（0.25 即从峰值回撤 25%）。
        win_rate: 胜率，基于**平仓交易**。
        payoff_ratio: 盈亏比 ``平均盈利 / 平均亏损``，基于**平仓交易**。
        turnover: 换手率，**单边**、**期间累计、不年化**。
        trading_days: 净值曲线的点数（交易日数）。
        closed_trades: 平仓交易笔数——胜率与盈亏比的分母。
        benchmark_annual_return: 基准的几何年化收益；未给基准时为 ``None``。
        excess_annual_return: 策略年化减基准年化；未给基准时为 ``None``。
        rejected_orders: **未成交而终结**的订单笔数（资金不足、出池、挂单过期等）。
        rejection_rate: 拒单率 ``拒单笔数 /（成交笔数 + 拒单笔数）``。它高说明大部分下单
            都没成——那种情形下净值曲线与指标都**不代表策略**，故必须看得见。
        notes: 每条缺失值的**原因**说明（如「无平仓交易」）。空元组表示没有缺失。
    """

    total_return: float
    annual_return: float
    sharpe: float
    max_drawdown: float
    win_rate: float
    payoff_ratio: float
    turnover: float
    trading_days: int
    closed_trades: int
    benchmark_annual_return: float | None = None
    excess_annual_return: float | None = None
    rejected_orders: int = 0
    rejection_rate: float = 0.0
    notes: tuple[str, ...] = field(default_factory=tuple)

    def as_dict(self) -> dict:
        """扁平字典，供落盘与打印。``notes`` 转为列表以便 JSON 序列化。"""
        payload = dict(self.__dict__)
        payload["notes"] = list(self.notes)
        return payload


def compute_metrics(
    equity: pd.Series,
    trades: pd.DataFrame | None = None,
    *,
    periods_per_year: int = TRADING_DAYS_PER_YEAR,
    risk_free: float = 0.0,
    benchmark_equity: pd.Series | None = None,
    rejected: pd.DataFrame | None = None,
) -> Metrics:
    """由净值曲线与成交明细算出指标。

    参数:
        equity: 净值曲线，交易日为索引（升序）、组合总资产为值。
        trades: 成交明细，列同 ``mbt.backtest.TRADE_COLUMNS``（``date`` / ``symbol`` /
            ``size`` / ``price`` / ``value`` / ``commission``）。``None`` 或空表示没有成交
            ——那是合法情形，只有依赖成交的指标会变成缺失。``symbol`` 列是**必需的**：平仓
            配对按标的分别做，缺了它会把一个标的的买入当成另一个标的卖出的对手方。
        periods_per_year: 年化口径的每年期数，默认 252（交易日）。
        risk_free: **年化**无风险利率（如 0.02 表示 2%）。默认 0，即不扣无风险收益。
        benchmark_equity: 基准的净值/价格序列（与 ``equity`` 同一区间口径）。给了它才算
            基准与超额收益。
        rejected: **未成交而终结**的订单明细（``mbt.backtest.BacktestResult.rejected``）。它
            不参与任何指标计算，只用来报出拒单笔数与拒单率——因为**高拒单率下上面的指标都
            不代表策略**（净值曲线可能只是一条平线），而那件事必须看得见。

    返回:
        :class:`Metrics`。

    抛:
        ValueError: 净值曲线为空，或它带有缺失值——**净值曲线空缺的地方不能就地补**
            （ADR-0005 的缺口纪律），它意味着这一跑没有真正跑起来。

    各指标口径：

    - **总收益** ``V_end / V_start − 1``；**年化** ``(1 + 总收益)^(periods_per_year / 期数) − 1``，
      其中期数为 ``交易日数 − 1``。用**交易日**而非日历天：净值只在交易日有定义，按日历天
      年化等于假设周末也在涨。
    - **夏普** ``mean(r − rf_d) / std(r, ddof=1) × sqrt(periods_per_year)``，其中
      ``rf_d = (1 + risk_free)^(1 / periods_per_year) − 1``——**几何**折算，不是除以 252。
      样本标准差（``ddof=1``）。收益率全等（无波动）或样本不足 2 个时返回缺失，**不返回无穷**。
    - **最大回撤** ``max_t (peak_t − V_t) / peak_t``，``peak`` 为净值的**运行最大值**，
      报为**正值**（图上则画负值，见 :mod:`mbt.report`）。
    - **胜率 / 盈亏比**基于**平仓交易**（FIFO 配对，见 :func:`closed_trade_pnls`）：
      胜率 ``盈利笔数 / 平仓笔数``；盈亏比 ``mean(盈利) / |mean(亏损)|``。盈亏比是 **payoff
      ratio**，**不是** profit factor（后者是「总盈利/总亏损」，本工具不输出以免混淆）。
      无平仓交易或无亏损笔时相应指标缺失。
    - **换手率** ``Σ|成交金额| / 2 / mean(净值)``。**单边**口径（买卖各一次算一轮），
      **期间累计、不年化**——年化换手率极易被误读。
    """
    if equity.empty:
        raise ValueError("净值曲线为空——这一跑没有真正跑起来，指标无从谈起")
    if equity.isna().any():
        raise ValueError(
            "净值曲线含有缺失值。净值空缺不能就地补（ADR-0005 的缺口纪律）："
            "补一条等于凭空造出一天的资产，而它不会在图上显示为异常。"
        )

    notes: list[str] = []
    periods = len(equity) - 1
    start, end = float(equity.iloc[0]), float(equity.iloc[-1])

    total_return = end / start - 1.0 if start > 0 else math.nan
    annual_return = _annualize(total_return, periods, periods_per_year)
    if math.isnan(total_return):
        notes.append("总收益缺失：期初资产不为正，比值无定义")
    elif math.isnan(annual_return):
        notes.append(
            f"年化收益缺失：只有 {len(equity)} 个点（不足一个收益期），或已亏至 0 而无法开方"
        )

    returns = equity.pct_change().dropna()
    sharpe = _sharpe(returns, periods_per_year, risk_free)
    if math.isnan(sharpe):
        notes.append(
            "夏普缺失：收益率无波动或样本不足两期，此时比值无定义（不报无穷，那会污染比较）"
        )

    max_drawdown = _max_drawdown(equity)

    pnls = closed_trade_pnls(trades)
    win_rate, payoff_ratio = _win_rate_and_payoff(pnls)
    if not pnls:
        notes.append("胜率与盈亏比缺失：没有平仓交易（只建仓未卖出也算）")
    elif not any(pnl < 0 for pnl in pnls):
        notes.append("盈亏比缺失：没有亏损的平仓交易，平均值无定义")

    turnover = _turnover(trades, equity)
    if trades is None or trades.empty:
        notes.append("换手率为 0：区间内没有任何成交")

    benchmark_annual_return = None
    excess = None
    if benchmark_equity is not None and len(benchmark_equity) >= 2:
        benchmark_total = float(benchmark_equity.iloc[-1]) / float(benchmark_equity.iloc[0]) - 1.0
        benchmark_annual_return = _annualize(
            benchmark_total, len(benchmark_equity) - 1, periods_per_year
        )
        if not math.isnan(annual_return) and not math.isnan(benchmark_annual_return):
            excess = annual_return - benchmark_annual_return
        elif not math.isnan(benchmark_annual_return):
            notes.append("超额收益缺失：策略年化本身算不出来，无从相减")
    elif benchmark_equity is not None:
        notes.append("基准年化缺失：基准序列不足两个点")

    filled = 0 if trades is None else len(trades)
    rejected_count = 0 if rejected is None else len(rejected)
    attempted = filled + rejected_count
    rejection_rate = (rejected_count / attempted) if attempted else 0.0

    return Metrics(
        total_return=total_return,
        annual_return=annual_return,
        sharpe=sharpe,
        max_drawdown=max_drawdown,
        win_rate=win_rate,
        payoff_ratio=payoff_ratio,
        turnover=turnover,
        trading_days=len(equity),
        closed_trades=len(pnls),
        benchmark_annual_return=benchmark_annual_return,
        excess_annual_return=excess,
        rejected_orders=rejected_count,
        rejection_rate=rejection_rate,
        notes=tuple(notes),
    )


def closed_trade_pnls(trades: pd.DataFrame | None) -> list[float]:
    """把成交明细配成**平仓交易**，返回每笔的已实现盈亏（元）。

    配对规则是 **FIFO、按股数、且按标的分别进行**：买入进队列（成本含手续费），卖出从队首
    扣减，逐段实现。**每笔卖出产生一条**平仓记录，其盈亏是本次卖出的各段之和（部分平仓因此
    只算一笔）。

    「按标的分别」这件事**必须有**：组合里 A 的买入与 B 的卖出在时间上相邻，若共用一条队列，
    A 的买入就会被当成 B 卖出的对手方，于是算出一个**无意义的盈亏**——而胜率与盈亏比都建在
    它上面。故 ``trades`` 必须带 ``symbol`` 列（:data:`mbt.backtest.TRADE_COLUMNS` 的契约）。

    卖出量超过持仓时**报错**——那说明成交明细本身不可信，而在此静默按可用量截断，会让
    胜率的分母凭空变小。

    .. note::

        未平仓的持仓**不进结果**——拿浮动盈亏当一笔「赢」是两类完全不同的统计对象
        （``CONTEXT.md`` 的**持仓**与净值曲线同理不在同一层）。
    """
    if trades is None or trades.empty:
        return []

    if "symbol" not in trades.columns:
        raise ValueError(
            "成交明细缺少 symbol 列，无法按标的分别配对——"
            "缺了它会把一个标的的买入当成另一个标的卖出的对手方，"
            "算出的胜率与盈亏比没有意义。"
            "列应如 ('date', 'symbol', 'size', 'price', 'value', 'commission')"
        )

    pnls: list[float] = []
    ordered = trades.sort_values("date")
    for symbol, group in ordered.groupby("symbol", sort=False):
        pnls.extend(_pair_one_symbol(group, symbol))
    return pnls


def _pair_one_symbol(trades: pd.DataFrame, symbol) -> list[float]:
    """单个标的的 FIFO 配对（口径见 :func:`closed_trade_pnls`）。"""
    open_lots: list[list[float]] = []  # [剩余股数, 每股成本(含费)]
    pnls: list[float] = []

    for row in trades.itertuples(index=False):
        size = float(row.size)
        value = float(row.value)
        commission = float(row.commission or 0.0)

        if size > 0:
            unit_cost = (value + commission) / size
            open_lots.append([size, unit_cost])
            continue

        remaining = -size
        available = sum(lot[0] for lot in open_lots)
        if remaining > available + _SHARE_EPSILON:
            raise ValueError(
                f"{symbol} 卖出 {remaining:g} 股超过当时持仓 {available:g} 股——"
                f"成交明细不可信，配对算出的胜率也就没有意义"
            )

        unit_proceeds = (abs(value) - commission) / remaining
        pnl = 0.0
        while remaining > 0 and open_lots:
            lot = open_lots[0]
            take = min(lot[0], remaining)
            pnl += (unit_proceeds - lot[1]) * take
            lot[0] -= take
            remaining -= take
            if lot[0] <= _SHARE_EPSILON:
                open_lots.pop(0)
        pnls.append(pnl)

    return pnls


def _annualize(total_return: float, periods: int, periods_per_year: int) -> float:
    """``(1 + 总收益)^(periods_per_year / 期数) − 1``；期数不足一期时无法年化。

    期数为 0（只有一个点）或无收益可年化（``1 + 总收益 ≤ 0``，即亏光）时返回缺失。
    """
    if periods < 1 or math.isnan(total_return) or total_return <= -1.0:
        return math.nan
    return (1.0 + total_return) ** (periods_per_year / periods) - 1.0


def _sharpe(returns: pd.Series, periods_per_year: int, risk_free: float) -> float:
    """年化夏普；无波动或样本不足两期时返回缺失（**不返回无穷**）。"""
    if len(returns) < 2:
        return math.nan

    std = float(returns.std(ddof=1))
    if not math.isfinite(std) or std == 0.0:
        return math.nan

    # 无风险利率按**几何**折算到日频：除以 252 会把 2% 算成 0.00794%，与实际复利不符。
    daily_risk_free = (1.0 + risk_free) ** (1.0 / periods_per_year) - 1.0
    excess = returns - daily_risk_free

    return float(excess.mean()) / std * math.sqrt(periods_per_year)


def _max_drawdown(equity: pd.Series) -> float:
    """``max_t (peak_t − V_t) / peak_t``，**正值**。"""
    peak = equity.cummax()
    drawdown = (peak - equity) / peak
    return float(drawdown.max())


def _win_rate_and_payoff(pnls: list[float]) -> tuple[float, float]:
    """胜率与盈亏比；无样本或无亏损笔时相应项为缺失。"""
    if not pnls:
        return math.nan, math.nan

    wins = [pnl for pnl in pnls if pnl > 0]
    losses = [pnl for pnl in pnls if pnl < 0]

    win_rate = len(wins) / len(pnls)
    if not losses:
        return win_rate, math.nan

    mean_win = sum(wins) / len(wins) if wins else 0.0
    mean_loss = -sum(losses) / len(losses)
    return win_rate, mean_win / mean_loss


def _turnover(trades: pd.DataFrame | None, equity: pd.Series) -> float:
    """``Σ|成交金额| / 2 / mean(净值)``——单边、期间累计。

    除以 2 是把「买 + 卖」折算成**单边**口径（一次完整往返算一轮）；平均净值作为分母
    而不是期初，是因为区间内资金规模会变。
    """
    if trades is None or trades.empty:
        return 0.0

    mean_equity = float(equity.mean())
    if mean_equity <= 0:
        return math.nan

    gross = float(trades["value"].abs().sum())
    return gross / 2.0 / mean_equity
