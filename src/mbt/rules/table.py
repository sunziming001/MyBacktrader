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

#: 出厂规则表路径。与本模块同目录，故「出厂表在哪」只有这一处定义
#: （数据层入口、回测入口与测试都经由 :meth:`RuleTable.load` 取用）。
SHIPPED_RULES_PATH = Path(__file__).resolve().parent / "a_share.toml"

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

    def at_or(self, on: dt.date, default):
        """同 :meth:`at`，但**没有适用条目时返回 ``default``** 而不报错。

        用于「查不到本身就是一个答案」的参数——例如新股上市初期的不设涨跌幅天数：
        「该上市日没有这条制度」是事实，不是我们不知道（ADR-0013）。别处仍应走
        :meth:`at`，那里的「查不到」意味着规则表不覆盖那段历史，必须报出来。
        """
        i = bisect_right(self._starts, on) - 1
        return default if i < 0 else self._values[i]


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
        self._handling_fee = {}
        self._regulatory_fee = {}

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

        for entry in raw.get("handling_fee", []):
            self._handling_fee.setdefault(entry["board"], []).append(entry)
        for entry in raw.get("regulatory_fee", []):
            self._regulatory_fee.setdefault(entry["board"], []).append(entry)

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

        self._handling_fee = {
            k: _Series(v, "rate", f"{k} 的经手费") for k, v in self._handling_fee.items()
        }
        self._regulatory_fee = {
            k: _Series(v, "rate", f"{k} 的证管费") for k, v in self._regulatory_fee.items()
        }

        self._stamp_duty = _Series(raw.get("stamp_duty", []), "sell_rate", "印花税")

        #: 上市初期不设涨跌幅的天数。**键是板块**，而查表时传的是**上市日**（ADR-0013）。
        _no_limit = {}
        for entry in raw.get("new_listing_no_limit", []):
            _no_limit.setdefault(entry["board"], []).append(entry)
        self._new_listing_no_limit = {
            k: _Series(v, "days", f"{k} 的新股上市初期不设涨跌幅天数") for k, v in _no_limit.items()
        }

        # ST 期间：登记了才认。
        self._st_periods: dict[str, list[tuple[dt.date, dt.date | None]]] = {}
        for entry in raw.get("st_period", []):
            self._st_periods.setdefault(entry["symbol"], []).append(
                (entry["start"], entry.get("end"))
            )

    @classmethod
    def load(cls, path=None) -> RuleTable:
        """从 TOML 文件加载规则表。``path`` 省略或为 ``None`` 时取出厂表。

        出厂表与本模块同目录，故它的位置只有这一处定义——调用方（数据层入口、回测入口、
        测试）都不必各自拼一次路径，也就不会各自拼错。
        """
        path = SHIPPED_RULES_PATH if path is None else Path(path)
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

    def has_st_period(self, symbol: str) -> bool:
        """该标的在规则表里是否**登记过** ST 期间。

        供股票池先做一次廉价的筛除，免得对四千多个标的逐日查 ST。把这件事放在规则表自己
        身上，是因为「登记了什么」是它的内部状态——让调用方去摸它的私有字段会让两者绑死。
        """
        return bool(self._st_periods.get(symbol))

    def st_period_count(self) -> int:
        """登记了多少条 ST 期间（跨标的合计）。供产物记录「这张表认识多少 ST」。"""
        return sum(len(spans) for spans in self._st_periods.values())

    def with_st_periods(self, periods) -> RuleTable:
        """返回一张**叠加**了这些 ST 期间的新表；**不改动原表**。

        为什么返回新表而不就地修改：规则表是共享对象（出厂表由 :meth:`load` 每次读盘新建，
        但调用方常把它传来传去）。就地修改会让「谁改了它」变成跨模块的隐式耦合，而回测的
        可复现性依赖于「同一份规则 => 同一个结果」。

        参数:
            periods: ``{符号: [(起, 止), …]}``。``止`` 为 ``None`` 表示延续到数据末端。
        """
        clone = RuleTable({})
        clone.__dict__.update(self.__dict__)
        merged = {symbol: list(spans) for symbol, spans in self._st_periods.items()}
        for symbol, spans in periods.items():
            merged.setdefault(symbol, []).extend(spans)
        clone._st_periods = merged
        return clone

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

    def new_listing_no_limit_days(self, board: str, listing_date: dt.date) -> int:
        """某板块在**该上市日**适用的「上市初期不设涨跌幅」交易日数；无适用条目则 ``0``。

        与 :meth:`price_limit` 有两处根本差别，都是刻意的（ADR-0013）：

        1. **查表的键是上市日，不是成交日。** 想知道「某根 K 线那天有没有涨跌幅限制」，
           要看的是这家公司**什么时候上市**——制度按上市时点适用，之后不再变。拿 K 线
           日期去查会把「2024 年上市的老制度股」错判成注册制新股。
        2. **查不到不报错，返回 0。** 别处「查不到」意味着「我们不知道那天的制度」，
           故报错（ADR-0005）。这里「查不到」本身就是答案：该上市日没有这条制度，
           故不豁免。把「不知道」与「没有」混为一种处置，会让 2023 年前上市的老股
           全部无法加载。
        """
        series = self._new_listing_no_limit.get(board)
        if series is None:
            return 0
        return int(series.at_or(listing_date, 0))

    def stamp_duty_rate(self, on: dt.date) -> float:
        """成交日的**印花税**卖出费率。印花税仅卖出方缴纳。"""
        return self._stamp_duty.at(on, "")

    def transfer_fee_rate(self, board: str, on: dt.date) -> float:
        """某板块在成交日的**过户费**费率，按成交金额计。"""
        return self._series(self._transfer_fee, board, on, f"板块 {board!r}")

    def transfer_fee_sides(self, board: str, on: dt.date) -> str:
        """某板块在成交日的过户费**收取方向**：``both`` / ``buy`` / ``sell`` / ``none``。"""
        return self._series(self._transfer_fee_sides, board, on, f"板块 {board!r}")

    def handling_fee_rate(self, board: str, on: dt.date) -> float:
        """某板块在成交日的**经手费**费率，按成交金额计、买卖双向。

        经手费由交易所收取，**总是发生**。它是否与 ``commission`` 重复，取决于用户的报价
        口径（「全佣」已含、「净佣」未含），故由调用方决定是否叠加。
        """
        return self._series(self._handling_fee, board, on, f"板块 {board!r}")

    def regulatory_fee_rate(self, board: str, on: dt.date) -> float:
        """某板块在成交日的**证管费**费率，按成交金额计、买卖双向。

        与 :meth:`handling_fee_rate` 同理：总是发生，是否叠加取决于佣金口径。
        """
        return self._series(self._regulatory_fee, board, on, f"板块 {board!r}")

    @staticmethod
    def _series(series: dict, board: str, on: dt.date, key_desc: str):
        if board not in series:
            raise RuleTableError(f"规则表中没有{key_desc}的任何规则")
        return series[board].at(on, "")
