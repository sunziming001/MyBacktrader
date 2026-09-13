"""行情面板：**字段 → 标的宽表**的映射（ADR-0009）。

ATR 这类信号在一行内同时用 ``high``、``low``、``close`` 且必须**同日对齐**，而标的宽表的
一格只能放一个数，故它写不出输入。行情面板补上这一层：把若干个字段的标的宽表绑成一个
一起消费的整体。

## 它是按需组装的显式结构，不是规范表示

逐标的的**字段宽表**仍是磁盘与引擎边界的规范形态（见 :mod:`mbt.data.tdx`）。面板只在需要
跨字段计算的地方现组，**不落盘、不常驻**：全市场全字段面板约 12,000 个标的 × 6,400 根
× 5 字段 ≈ 3.9 亿格，float64 下约 3.1 GB，且必须一次性急加载——与 ``.day`` 的逐标的文件
布局及增量更新正面冲突。

## 校验只做它自己那一件

面板校验的是**字段之间的对齐**（各字段的 ``index`` 与 ``columns`` 完全一致），因为跨字段
计算的全部正确性都建立在同日对齐上，而对齐一旦不成立，结果是**静默错位**而非报错。

它**不**校验索引是否为升序 ``DatetimeIndex``：那是**标的宽表**自身的契约，由消费它的信号层
在接缝上强制（:func:`mbt.signals._symbol_frame.check_symbol_frame`）。划分的依据是各管一事——
面板管「字段之间是否对齐」，信号层管「什么算合法标的宽表」——好让后者的定义在全项目里
只有一处。
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from types import MappingProxyType

import pandas as pd

from .errors import MarketDataError
from .market import MarketData


class Panel:
    """**字段 → 标的宽表**的映射（ADR-0009）。由 :func:`assemble_panel` 或直接构造。

    直接构造的路径是给测试与手工装配用的：它只要求「字段 → 数据帧」这一映射与字段间的
    对齐，不要求这些帧来自磁盘。

    属性:
        fields: 字段名到标的宽表的只读映射。
    """

    def __init__(self, fields: Mapping[str, pd.DataFrame]) -> None:
        if not fields:
            raise MarketDataError("行情面板至少要有一个字段")

        for name, frame in fields.items():
            if not isinstance(frame, pd.DataFrame):
                raise MarketDataError(
                    f"字段 {name!r} 必须是 DataFrame（标的宽表），收到 {type(frame).__name__}"
                )

        reference_name, reference = next(iter(fields.items()))
        for name, frame in fields.items():
            if not frame.index.equals(reference.index):
                raise MarketDataError(
                    f"字段 {name!r} 与 {reference_name!r} 的 index 不一致。"
                    f"跨字段计算要求各字段同日对齐，否则会静默错位"
                )
            if not frame.columns.equals(reference.columns):
                raise MarketDataError(
                    f"字段 {name!r} 与 {reference_name!r} 的 columns 不一致。"
                    f"跨字段计算要求各字段的标的集合相同，否则会静默错位"
                )

        duplicated = reference.columns[reference.columns.duplicated()]
        if len(duplicated) > 0:
            raise MarketDataError(
                f"标的宽表的 columns 有重复：{list(duplicated)}。" f"同一标的不应在面板里出现两次"
            )

        self._fields = MappingProxyType(dict(fields))

    @property
    def fields(self) -> Mapping[str, pd.DataFrame]:
        """字段名到标的宽表的只读映射。"""
        return self._fields

    @property
    def field_names(self) -> tuple[str, ...]:
        """面板声明的字段名，按构造时的顺序。"""
        return tuple(self._fields)

    @property
    def symbols(self) -> pd.Index:
        """面板覆盖的标的。各字段的标的集合相同（构造时已校验）。"""
        return next(iter(self._fields.values())).columns

    def __getitem__(self, field: str) -> pd.DataFrame:
        try:
            return self._fields[field]
        except KeyError:
            raise KeyError(f"面板没有字段 {field!r}；已声明的字段为 {list(self._fields)}") from None

    def __contains__(self, field: object) -> bool:
        return field in self._fields

    def __repr__(self) -> str:
        return (
            f"Panel(fields={list(self._fields)}, "
            f"symbols={len(self.symbols)}, bars={len(next(iter(self._fields.values())))})"
        )


def clip_fields(
    fields: dict[str, pd.DataFrame],
    markets,
    *,
    start=None,
    end=None,
) -> dict[str, pd.DataFrame]:
    """把算好的信号截到回测区间，并对齐到 ``markets`` 的标的与顺序。

    **为什么必须先算后截**：回看类信号（估值百分位要看 1000 个交易日、跌幅要看 252 个）若拿
    **已截断**的行情去算，区间开头的窗口就只剩几天——截短了**不会报错**，只会让开头那段日子的
    值失真、或整段变成缺失而被过滤条件静默排除。故顺序是：用**完整历史**算，再截到回测区间。

    它同时服务两类信号（估值来自财务、跌幅来自行情），故住在**面板**这一层而不是某一类信号的
    模块里。

    截断口径与 :func:`~mbt.data.loader.slice_markets` 一致（``.loc`` 两端含），并保留
    ``markets`` 的列顺序——引擎那边的行情面板正是按那个顺序组装的，顺序不同会被
    :class:`Panel` 判为不一致（这是对的：顺序不同往往意味着两份数据来源对不上，而静默重排
    会掩盖它）。
    """
    lower = pd.Timestamp(start) if start else None
    upper = pd.Timestamp(end) if end else None
    symbols = [market.symbol for market in markets]

    clipped = {}
    for name, frame in fields.items():
        one = frame.loc[lower:upper] if lower is not None or upper is not None else frame
        missing = [symbol for symbol in symbols if symbol not in one.columns]
        if missing:
            raise MarketDataError(
                f"信号 {name!r} 缺少标的 {missing[:3]}…——它与行情不是同一次取数的产物"
            )
        clipped[name] = one[symbols]
    return clipped


def with_signals(panel: Panel, signals: Mapping[str, pd.DataFrame] | None) -> Panel:
    """把**额外信号**并进面板，供选股规则取用；``None`` 或空则原样返回。

    选股规则吃的是 :class:`Panel`，而估值这类信号不在 ``.day`` 行情里（它们来自财务数据），
    故必须并进去才能被过滤器取用。

    对齐**必须**成立：:class:`Panel` 只允许各字段同日同标的，而两个来源分别由「行情」与
    「行情 + 财务」算出——任一处口径不同就会静默错位，故这里不做任何形状修补，让 ``Panel``
    的校验直接报错。

    与行情字段**重名**要报错而不是覆盖：覆盖会让行情字段静默消失，而下游会以为拿到的还是行情。
    """
    if not signals:
        return panel

    fields = {name: panel[name] for name in panel.field_names}
    for name, frame in signals.items():
        if name in fields:
            raise ValueError(f"信号字段 {name!r} 与行情字段重名，会静默覆盖行情")
        fields[name] = frame
    return Panel(fields)


def assemble_panel(markets: Sequence[MarketData], fields: Sequence[str]) -> Panel:
    """把若干个**已质检**的行情组建成面板，只取显式声明的字段。

    参数:
        markets: 一组 :class:`~mbt.data.market.MarketData`。收它是为了让面板的存在即
            「已质检」的证据（与 ``MarketData`` 自身的做法一致）——面板因此**不再重复**
            越界检查，也不该允许绕过那条正门。
        fields: 要装进面板的字段，**由调用方显式声明**。刻意不推断：``amount`` 之类与
            信号无关的字段没有理由被无声带进计算，而「给了哪些字段」应当是一件被声明的
            意图，不是数据源实现的副产品。

    返回:
        含这些字段的 :class:`Panel`。

    抛:
        MarketDataError: 标的为空、字段为空、或某个标的缺某个字段。标的重复由 :class:`Panel`
            的列唯一性校验报出（见下）。

    各标的的交易日**允许不同**（停牌、次新股都会造成差异）：结果按各字段的日集合并集对齐，
    缺处为缺失值。这不是填充——缺失仍是缺失，只是换了个位置表达（ADR-0005）。
    """
    if not markets:
        raise MarketDataError("组装面板至少要有一个标的")
    if not fields:
        raise MarketDataError("组装面板至少要声明一个字段")

    symbols = [market.symbol for market in markets]

    built: dict[str, pd.DataFrame] = {}
    for field in fields:
        series: list[pd.Series] = []
        for market in markets:
            if field not in market.prices.columns:
                raise MarketDataError(
                    f"标的 {market.symbol} 的行情缺少字段 {field!r}；"
                    f"它有 {list(market.prices.columns)}"
                )
            series.append(market.prices[field])
        # 用 concat + keys 而非字典：字典遇到重复标的会**悄悄覆盖**前一份，而重复应当由
        # Panel 的列唯一性校验报错，不该在这里被吞掉。
        built[field] = pd.concat(series, axis=1, keys=symbols)

    return Panel(built)
