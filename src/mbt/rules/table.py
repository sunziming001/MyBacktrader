"""规则表的加载与查表。

查表语义只有一条：给定 (键, 成交日)，取**生效日期 ≤ 成交日**中生效日期最大的那一条。
变更日**当天**即生效。若成交日早于该键最早的生效日期，报错而**不猜**——缺口纪律
（ADR-0005）在规则表上的对应：默认一个限幅等于凭空造出一条规则。
"""

from __future__ import annotations

import datetime as dt
from bisect import bisect_right
from pathlib import Path

import tomli

from .board import board_of
from .errors import RuleTableError

#: 过户费的收取方向取值。
_FEE_SIDES = ("both", "buy", "sell", "none")


class _Series:
    """某个参数在同一键下的生效日期序列，按生效日期升序。"""

    def __init__(self, entries, value_field: str, what: str):
        entries = sorted(entries, key=lambda e: e["effective_from"])
        self._starts = [e["effective_from"] for e in entries]
        self._values = [e[value_field] for e in entries]
        self._what = what

    @property
    def starts(self):
        return self._starts

    def at(self, on: dt.date, key_desc: str) -> object:
        i = bisect_right(self._starts, on) - 1
        if i < 0:
            raise RuleTableError(
                f"{self._what}未覆盖 {on.isoformat()}{key_desc}："
                f"该键最早的生效日期是 {self._starts[0].isoformat()}"
            )
        return self._values[i]


class RuleTable:
    """交易制度规则表。

    参数:
        root: 解析自 TOML 的原始结构。通常不要直接构造，用 :meth:`load`。
    """

    def __init__(self, raw: dict):
        self._price_limit = {}
        self._st_price_limit = {}
        self._transfer_fee = {}
        self._transfer_fee_sides = {}

        for entry in raw.get("price_limit", []):
            self._price_limit.setdefault(entry["board"], []).append(entry)
        for entry in raw.get("st_price_limit", []):
            self._st_price_limit.setdefault(entry["board"], []).append(entry)
        for entry in raw.get("transfer_fee", []):
            sides = entry["sides"]
            if sides not in _FEE_SIDES:
                raise RuleTableError(
                    f"过户费的 sides 取值 {sides!r} 不合法，应为 {_FEE_SIDES} 之一"
                )
            self._transfer_fee.setdefault(entry["board"], []).append(entry)
            self._transfer_fee_sides.setdefault(entry["board"], []).append(entry)

        self._price_limit = {
            k: _Series(v, "limit", f"{k} 的涨跌幅限制") for k, v in self._price_limit.items()
        }
        self._st_price_limit = {
            k: _Series(v, "limit", f"{k} 的 ST 涨跌幅限制") for k, v in self._st_price_limit.items()
        }
        self._transfer_fee = {
            k: _Series(v, "rate", f"{k} 的过户费") for k, v in self._transfer_fee.items()
        }
        self._transfer_fee_sides = {
            k: _Series(v, "sides", f"{k} 的过户费收取方向")
            for k, v in self._transfer_fee_sides.items()
        }

        self._stamp_duty = _Series(raw.get("stamp_duty", []), "sell_rate", "印花税")

        # ST 期间：登记了才认。
        self._st_periods: dict[str, list[tuple[dt.date, dt.date | None]]] = {}
        for entry in raw.get("st_period", []):
            self._st_periods.setdefault(entry["symbol"], []).append(
                (entry["start"], entry.get("end"))
            )

    @classmethod
    def load(cls, path) -> RuleTable:
        """从 TOML 文件加载规则表。"""
        path = Path(path)
        if not path.is_file():
            raise RuleTableError(f"规则表文件不存在：{path}")
        try:
            with path.open("rb") as f:
                raw = tomli.load(f)
        except tomli.TOMLDecodeError as exc:
            raise RuleTableError(f"规则表 {path} 不是合法的 TOML：{exc}") from exc
        return cls(raw)

    # --- 查表 ---

    def board_of(self, symbol: str) -> str:
        """标的所属**板块**（主板 / 创业板 / 科创板 / 北交所）。"""
        return board_of(symbol)

    def is_st(self, symbol: str, on: dt.date) -> bool:
        """标的在成交日是否处于 ST 期间。

        **只有登记在 ``[[st_period]]`` 里的才认**——未登记即视为非 ST。
        这是必须显式声明的局限：本地价格数据不含股票名称，无法回溯历史上的 ST 状态。
        """
        for start, end in self._st_periods.get(symbol, ()):
            if start <= on and (end is None or on <= end):
                return True
        return False

    def limit_for(self, symbol: str, on: dt.date) -> float:
        """标的在成交日的**涨跌幅限制**，已按需叠加 ST 覆盖。

        ST 限幅本身按板块取值（主板 5%，而创业板 ST 仍与创业板普通股相同），
        因此它是规则表里的**数据**，不是代码里的分支。
        """
        board = self.board_of(symbol)
        if self.is_st(symbol, on):
            if board not in self._st_price_limit:
                raise RuleTableError(
                    f"标的 {symbol} 在 {on.isoformat()} 属于 ST，"
                    f"但规则表中没有{board}的 ST 涨跌幅限制——"
                    f"静默退回普通限幅会放过本不该成交的交易"
                )
            return self._st_price_limit[board].at(on, "")
        return self.price_limit(board, on)

    def price_limit(self, board: str, on: dt.date) -> float:
        """某板块在成交日的**涨跌幅限制**（如 0.10 表示 10%），不含 ST 覆盖。"""
        return self._series(self._price_limit, board, on, f"板块 {board!r}")

    def stamp_duty_rate(self, on: dt.date) -> float:
        """成交日的**印花税**卖出费率。印花税仅卖出方缴纳。"""
        return self._stamp_duty.at(on, "")

    def transfer_fee_rate(self, board: str, on: dt.date) -> float:
        """某板块在成交日的**过户费**费率，按成交金额计。"""
        return self._series(self._transfer_fee, board, on, f"板块 {board!r}")

    def transfer_fee_sides(self, board: str, on: dt.date) -> str:
        """某板块在成交日的过户费**收取方向**：``both`` / ``buy`` / ``sell`` / ``none``。"""
        return self._series(self._transfer_fee_sides, board, on, f"板块 {board!r}")

    @staticmethod
    def _series(series: dict, board: str, on: dt.date, key_desc: str):
        if board not in series:
            raise RuleTableError(f"规则表中没有{key_desc}的任何规则")
        return series[board].at(on, "")
