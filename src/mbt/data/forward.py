"""估值的前瞻分支（票据 #50）：把「一致预期」按通达信原式接进来。

## 只能在**非回溯**场景使用

    **这一模块给的是「现在这一版」一致预期，不是「评估日当时」的一致预期。**

一致预期的来源 :mod:`mbt.data.gpone` 是**快照**：每个 ``(代码, 字段)`` 只有一条记录，
文件里**没有日期字段**。机构上一回给出的预测已被下一回覆盖——不是「旧记录被挤掉」，是
**根本没有旧记录**（这是 ADR-0006 修订二登记的缺口）。

所以：**不得**把本模块的读数用于任何声称时点正确的回测。它适用于「今天的池子、今天的读数」
这类非回溯场景。这也是本模块**没有** ``as_of`` 参数的原因——不是漏了，是不给这个口子。

正因如此，本模块**刻意不接进** :func:`mbt.data.valuation.build_valuation`：那条路按公告日
点取（ADR-0006），把一份「最新值」混进去会让整张估值表静默失去时点正确性。有一条测试钉住
这件事（``test_valuation_does_not_reach_for_the_forward_branch``）。

## 原式（票据 #50 正文）

```
财年T := GPONEDAT(4);                     K线年 := YEAR;
用前瞻 := 财年T>0 AND K线年=财年T;

PE前瞻 := IF(EPS_T>0, C/EPS_T, IF(PE预期>0, PE预期, 0));
   其中 EPS_T := GPONEDAT(5)   PE预期 := GPONEDAT(23)
增预期 := IF(基期净利万>0 AND 净利T>0, (净利T-基期净利万)/基期净利万*100, 0);
   其中 净利T := GPONEDAT(8)   基期净利万 := 基期年报归母净利

选用PE := IF(用前瞻, PE前瞻, PE历史) NODRAW;
选用增 := IF(用前瞻, 增预期, 增历史);      ← PE 与增长率**成对切换**
```

**「成对切换」在本模块是结构性保证，不是约定**：:func:`choose` 一次返回 PE 与增长率两个值
（:class:`Chosen`），没有「只换 PE 不换增长」的写法。

单位与缺失照原式：``净利T`` 与 ``基期净利万`` 都是**万元**；两者任一不大于 0 时 ``增预期``
取 **0**（原式如此，不是缺失）。``C`` 用**原始价**——与 :mod:`mbt.data.valuation` 同一个理由：
PE 要的是当时的成交价，用后复权价会整体放大（实测 `sh600000` 放大 13.38 倍）。
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import pandas as pd

#: 取「基期年报」的字段名：归母净利润（元，累计）。年末那一期即全年数。
_BASE_PROFIT_FIELD = "net_profit_ytd"

#: 元 → 万元。``GPONEDAT(8)`` 的单位就是万元，故基期净利要换算过来才能相减。
_YUAN_PER_WAN = 1e4

#: 字段号（见 :data:`mbt.data.gpone.FIELDS`）。写成常量免得正文里散着魔数。
FIELD_FISCAL_YEAR = 4
FIELD_EPS_T = 5
FIELD_NET_PROFIT_T = 8
FIELD_PE_EXPECTED = 23

#: 调用方要一并交代给用户的那句话。非回溯场景里也该让人知道读到的是什么。
FORWARD_CAVEAT = (
    "这是「现在这一版」一致预期，不是评估日当时的；本地没有带发布日的一致预期历史，"
    "故不能用于声称时点正确的回测。"
)


@dataclass(frozen=True)
class ForwardReading:
    """某标的**今天**这一版的前瞻原料。

    属性:
        symbol: 标的。
        fiscal_year: ``GPONEDAT(4)`` 财年T；没有时 0（原式：「空值显示为 0」）。
        eps_t: ``GPONEDAT(5)`` 一致预期T年每股收益（元）。
        net_profit_t: ``GPONEDAT(8)`` 一致预期T年净利润（**万元**）。
        pe_expected: ``GPONEDAT(23)`` 一致预期T年PE。
        base_net_profit: 基期年报归母净利（**万元**）；拿不到时报缺失。
    """

    symbol: str
    fiscal_year: float
    eps_t: float
    net_profit_t: float
    pe_expected: float
    base_net_profit: float | None

    @property
    def growth_expected(self) -> float:
        """``增预期``（**百分数**）。两者任一不大于 0 时取 **0**（原式如此，不是缺失）。"""
        if not self.base_net_profit or self.base_net_profit <= 0 or self.net_profit_t <= 0:
            return 0.0
        return (self.net_profit_t - self.base_net_profit) / self.base_net_profit * 100.0

    def pe_forward(self, close: float) -> float:
        """``PE前瞻 = IF(EPS_T>0, C/EPS_T, IF(PE预期>0, PE预期, 0))``。

        参数:
            close: **原始价**（当时的成交价），不是后复权价。
        """
        if self.eps_t > 0:
            return close / self.eps_t
        if self.pe_expected > 0:
            return self.pe_expected
        return 0.0


@dataclass(frozen=True)
class Chosen:
    """「选用PE / 选用增」这一对值。

    两个值**同出一处**是有意的：原式要求 PE 与增长率成对切换，而这个类型没有「只给一个」的
    构造方式，故调用方拿不到「换了 PE 没换增长」的组合。
    """

    pe: float
    growth: float
    used_forward: bool


def uses_forward(fiscal_year: float, bar_year: int) -> bool:
    """``用前瞻 := 财年T>0 AND K线年=财年T``。

    两条缺一不可：``财年T`` 为 0 表示这只票没有一致预期；年份不匹配表示预期落在**别的年份**
    （例如到了次年，T 还是去年那个）——那时用预期口径会拿旧预测配新价格。
    """
    return fiscal_year > 0 and int(fiscal_year) == bar_year


def choose(
    reading: ForwardReading,
    *,
    close: float,
    bar_year: int,
    trailing_pe: float,
    trailing_growth: float,
) -> Chosen:
    """按原式在历史与前瞻之间**成对**切换。

    参数:
        reading: 该标的的前瞻原料。
        close: **原始价**。
        bar_year: ``K线年``，即这根 K 线所属年份。
        trailing_pe: ``PE历史``（:func:`mbt.data.valuation.price_earnings_ratio`）。
        trailing_growth: ``增历史``（:data:`mbt.data.fundamental` 的 ``growth_ytd``）。
    """
    if uses_forward(reading.fiscal_year, bar_year):
        return Chosen(reading.pe_forward(close), reading.growth_expected, True)
    return Chosen(trailing_pe, trailing_growth, False)


def reading(
    symbol: str,
    *,
    gpone,
    financials,
    fields: tuple[int, ...] = (
        FIELD_FISCAL_YEAR,
        FIELD_EPS_T,
        FIELD_NET_PROFIT_T,
        FIELD_PE_EXPECTED,
    ),
) -> ForwardReading:
    """读一只标的的前瞻原料。

    参数:
        symbol: 标的。
        gpone: :class:`~mbt.data.gpone.GponeDataSource`。
        financials: :class:`~mbt.data.fundamental.CwDataSource` 之类，提供 ``records(symbol)``。
            用来取**基期年报**的归母净利（``净利T`` 是预测值，增长率要与**已公告的实际值**比）。
        fields: 要读的字段号，默认就是原式那四个。
    """
    values = {field: (gpone.value(symbol, field) or 0.0) for field in fields}
    return ForwardReading(
        symbol=symbol,
        fiscal_year=values[FIELD_FISCAL_YEAR],
        eps_t=values[FIELD_EPS_T],
        net_profit_t=values[FIELD_NET_PROFIT_T],
        pe_expected=values[FIELD_PE_EXPECTED],
        base_net_profit=_base_annual_profit(financials, symbol),
    )


def readings(symbols, *, gpone, financials) -> dict[str, ForwardReading]:
    """一批标的的前瞻原料。缺数据的标的**不出现**在结果里（不补 0 假装有预期）。"""
    out: dict[str, ForwardReading] = {}
    for symbol in symbols:
        try:
            gpone.path_for(symbol)
        except Exception:  # noqa: BLE001 — 不覆盖的市场（如北交所）直接跳过
            continue
        out[symbol] = reading(symbol, gpone=gpone, financials=financials)
    return out


def _base_annual_profit(financials, symbol: str) -> float | None:
    """基期年报归母净利（**万元**）：最近一期**可用的**年末财报。

    只取 ``12-31`` 那一期——``net_profit_ytd`` 是**累计**数，一季报给的是「年初到 3 月底」，
    拿它当「基期年报」会把增长率算成一个与全年无关的数。

    ``usable`` 那道门（公告日 > 报告期、且年份 ≥ 2005）照旧要过：2005 年之前的公告日是
    「等于报告期」的占位符，用它会提前数月知道尚未公布的财报（ADR-0006）。
    """
    base = None
    for record in financials.records(symbol):
        if record.report_period.month != 12 or record.report_period.day != 31:
            continue
        if not record.usable:
            continue
        base = record  # records() 是升序的，最后一条即最近一期
    if base is None:
        return None
    return base.values[_BASE_PROFIT_FIELD] / _YUAN_PER_WAN


def readings_frame(readings_by_symbol: Mapping[str, ForwardReading]) -> pd.DataFrame:
    """把原料摊成一张表（一行一标的），便于盘后筛一遍。索引是标的。"""
    return pd.DataFrame(
        {
            symbol: {
                "fiscal_year": item.fiscal_year,
                "eps_t": item.eps_t,
                "net_profit_t": item.net_profit_t,
                "pe_expected": item.pe_expected,
                "base_net_profit": item.base_net_profit,
                "growth_expected": item.growth_expected,
            }
            for symbol, item in readings_by_symbol.items()
        }
    ).T
