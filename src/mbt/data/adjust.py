"""复权：由原始价与除权除息事件算出后复权 / 前复权视图（ADR-0003、ADR-0006）。

落盘只存**原始价 + 复权事件**，任何已复权的价格序列都不保存——前复权价会随未来
每一次除权整体重算，存下来就不可复现（ADR-0003）。视图因此一律**用时计算**。

## 两种视图及其锚点

=====  ==================================  ==============================
视图   因子                                 锚点（谁的价格保持原值）
=====  ==================================  ==============================
后复权 ``raw × Π_{t_i ≤ d} (1/r_i)``        序列**第一根** K 线 → 回测用
前复权 ``raw × F(d) / F(末根)``             序列**最后一根** K 线 → 展示用
=====  ==================================  ==============================

前复权的锚点是**末根 K 线、而非「最新那次除权」**。两者通常一致，但一旦权息文件里
存在「已公告、未除权」的未来事件（实测 gbbq 的事件日期上界就晚于最新行情），
按「最新事件」锚定就会把尚未生效的除权**折算进当前价**，等价于提前知道未来——
正是 ADR-0006 要防的前视偏差。锚在末根 K 线则天然免疫：未落到任何 K 线上的事件
既进不了因子，也改不了锚点。

## 序列之外的除权事件不参与

事件早于序列首根时，算 r 需要的**除权日前一交易日收盘价**不在序列里。猜一个基数是
ADR-0005 明令禁止的。这类事件对整条序列只贡献一个**常数倍数**，而后复权视图的绝对
尺度本就任意（回测只关心收益率与相对形状），故略去它不影响回测结论。

代价要说清楚：本模块的「后复权」与全历史口径的后复权**相差一个常数**，数值不必与
行情软件一致；与行情软件可比的是前复权视图（其最新价不变是硬性性质）。
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass

import numpy as np
import pandas as pd

from .errors import MarketDataError

#: 会被复权改写价格的列。成交量与成交额**不参与**——复权是价格口径的换算，
#: 不是把成交也折算一遍（份额折算另属 ETF 的类别 11，本项目不含 ETF）。
PRICE_COLUMNS = ("open", "high", "low", "close")


@dataclass(frozen=True)
class AdjustmentEvent:
    """一条**除权除息**事件。

    四个数值字段对应权息文件的 f1..f4，此处按「除权除息」这一类别的语义命名。
    原始文件存的是 float32，故数值带约 7 位有效数字的表示误差（如 4.20 会读成
    4.199999809265137）。**不做四舍五入**——那等于替公告猜一个位数（ADR-0005）；
    误差量级约 1e-8，对复权因子没有实际影响。
    """

    symbol: str
    ex_date: dt.date
    cash_per_10: float = 0.0
    rights_price: float = 0.0
    bonus_per_10: float = 0.0
    rights_per_10: float = 0.0

    @property
    def cash_per_share(self) -> float:
        """每股分红（元）。文件里存的是「每 10 股」。"""
        return self.cash_per_10 / 10.0

    @property
    def bonus_per_share(self) -> float:
        """每股送转股（股）。送股与转增对价格的作用相同，故合并。"""
        return self.bonus_per_10 / 10.0

    @property
    def rights_per_share(self) -> float:
        """每股配股（股）。"""
        return self.rights_per_10 / 10.0

    def reference_price(self, prev_close: float) -> float:
        """除权除息参考价（标准公式）。

        ``ref = (前收盘 − 每股分红 + 配股价 × 每股配股) / (1 + 每股送转 + 每股配股)``

        分母是除权后的股本放大倍数；分子把现金分红从价格里剔除、再把配股缴款加回。
        不按分位取整——交易所公布的参考价会取整，但复权因子用的是**未取整**的理论值，
        取整会引入每笔不足一分的偏差，与「连续性」这一目标冲突。
        """
        rights = self.rights_per_share
        return (prev_close - self.cash_per_share + self.rights_price * rights) / (
            1.0 + self.bonus_per_share + rights
        )

    def factor(self, prev_close: float) -> float:
        """除权因子 ``r = ref / 前收盘``。

        正常情形 ``r < 1``（除权后价格下降）。取到非正数说明事件数据本身不成立
        （如分红高于前收盘加配股缴款），按缺口纪律报错而不是算出一个负价格。
        """
        if prev_close <= 0:
            raise MarketDataError(
                f"{self.symbol} 在 {self.ex_date.isoformat()} 的前收盘价 {prev_close} 非正，"
                f"无法计算复权因子"
            )
        factor = self.reference_price(prev_close) / prev_close
        if not factor > 0:
            raise MarketDataError(
                f"{self.symbol} 在 {self.ex_date.isoformat()} 的除权因子为 {factor}，"
                f"事件数据不成立（前收盘 {prev_close}，每 10 股分红 {self.cash_per_10}、"
                f"配股价 {self.rights_price}、每 10 股配股 {self.rights_per_10}）"
            )
        return factor


def adjustment_factors(
    prices: pd.DataFrame,
    events,
    as_of: dt.date | dt.datetime | None = None,
) -> pd.Series:
    """累积复权因子 ``F(d) = Π_{t_i ≤ d} (1 / r_i)``，索引与 ``prices`` 相同。

    后复权价即 ``原始价 × F``；前复权价即 ``原始价 × F / F(末根)``。

    参数:
        prices: 原始价宽表，须含 ``close``（复权因子要用除权日前一交易日的收盘价）
            且索引为**升序**的日期索引。
        events: :class:`AdjustmentEvent` 序列。
        as_of: 评估日，之后（含「已公告未除权」）的事件一律不纳入（ADR-0006）。
            默认取序列最后一根 K 线的日期。
    """
    if "close" not in prices.columns:
        raise ValueError("价格表必须含 close 列：复权因子需要除权日前一交易日的收盘价")
    if not prices.index.is_monotonic_increasing:
        raise ValueError("价格表的日期索引必须升序——因子按日期定位，乱序会算错")

    cutoff = _cutoff(prices, as_of)
    index = prices.index
    closes = prices["close"].to_numpy(dtype="float64")

    # 每个事件只影响「从除权日起的每一根 K 线」，故在除权日那一根上放一个乘数，
    # 再取累积乘积——这样同一根 K 线上有多个事件时也自然合并。
    steps = np.ones(len(index), dtype="float64")
    for event in events:
        if event.ex_date > cutoff:
            continue
        position = int(index.searchsorted(pd.Timestamp(event.ex_date), side="left"))
        # position == 0：事件早于序列首根，没有前一交易日收盘价（见模块文档）。
        # position == len：事件晚于序列末根，落不到任何 K 线上。
        if position == 0 or position >= len(index):
            continue
        steps[position] *= 1.0 / event.factor(closes[position - 1])

    return pd.Series(np.cumprod(steps), index=index, name="adjustment_factor")


def backward_adjusted(
    prices: pd.DataFrame,
    events,
    as_of: dt.date | dt.datetime | None = None,
) -> pd.DataFrame:
    """**后复权**视图：回测用。

    首根 K 线的价格保持原始值，其后按 ``× Π(1/r)`` 放大，故除权日不再出现假跳空，
    且不引入序列之外的任何未来信息。
    """
    return _apply(prices, events, as_of, anchor="first")


def forward_adjusted(
    prices: pd.DataFrame,
    events,
    as_of: dt.date | dt.datetime | None = None,
) -> pd.DataFrame:
    """**前复权**视图：展示用。

    以序列**末根 K 线**为锚，故最新价等于原始价，历史价被下调。新的一次除权会让整条
    历史重算，因此这个视图只用于展示，不落盘、不进回测（ADR-0003）。
    """
    return _apply(prices, events, as_of, anchor="last")


def _apply(prices: pd.DataFrame, events, as_of, anchor: str) -> pd.DataFrame:
    out = prices.copy()
    if len(prices) == 0:
        return out

    factors = adjustment_factors(prices, events, as_of=as_of)
    if anchor == "last":
        factors = factors / factors.iloc[-1]

    for column in PRICE_COLUMNS:
        if column in out.columns:
            out[column] = out[column] * factors
    return out


def _cutoff(prices: pd.DataFrame, as_of) -> dt.date:
    if as_of is None:
        return prices.index[-1].date()
    if isinstance(as_of, dt.datetime):  # pd.Timestamp 是 datetime 的子类
        return as_of.date()
    if isinstance(as_of, dt.date):
        return as_of
    raise TypeError(f"as_of 应是 date / datetime，收到 {as_of!r}")
