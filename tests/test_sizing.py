"""持仓分配：**买入数量必须把交易费用算进去**（票据 #29）。

本文件的核心是 `test_a_lone_position_is_not_rejected_for_want_of_fees`。在它之前，
默认 sizer 用 ``int(现金 ÷ 名额 ÷ 价格)``，**不给费用留余量**；于是单标的（名额 = 1）时
买入金额几乎正好等于现金，加上**过户费**即超出 → 每一笔买单都以 ``Margin`` 被拒 →
**零成交，而退出码仍是 0**。

这类失败最难发现：产物里 `equity.csv` 是一条平线、`closed_trades` 为 0，看着像个
「本来就没信号」的正常结果。故这里的断言刻意做成**两条互相独立**的：

1. 用**规则表**（而非 sizer 的公式）独立算出当日该付的费用，验证「成交金额 + 费用 ≤ 现金」；
2. 提高费率必须**买到更少**——钉住「费用真的被算进去了」，而不是碰巧躲过了一次超支。

费用一律走 ``synthetic_rules``（沪主板过户费 0.00002、双向），它是夹具表、数值是编造的，
但**收取方式**与出厂表一致：与佣金无关，买入即收。
"""

from __future__ import annotations

import backtrader as bt
import pandas as pd
import pytest

from mbt.backtest import run_portfolio_backtest
from mbt.universe import UniverseRules

#: `synthetic_rules` 里沪主板的过户费率：双向收取（``sides = "both"``）。
TRANSFER_FEE_RATE = 0.00002

#: `synthetic_rules` 里沪主板的涨跌幅限制。**注意它在 2023-06-01 从 0.10 改成 0.12**——夹具
#: 故意设了这次变更（用来验证「按成交日取当日生效规则」），而夹具数据在 2024 年，故生效值是
#: 0.12。sizer 按 ``定量价 × (1 + 它)`` 留余地，因为订单在下一根成交。
PRICE_LIMIT = 0.12

PRICE = 10.0
CASH = 100_000.0


class BuyOnce(bt.Strategy):
    """第一根 K 线后对每个标的各下一张买单，其后不动。"""

    def next(self):
        if len(self) == 1:
            for data in self.datas:
                self.buy(data=data)


class BuyThenSell(bt.Strategy):
    """买入，下一根再卖出全部持仓——用来确认 sizer 不插手卖出。"""

    def next(self):
        if len(self) == 1:
            for data in self.datas:
                self.buy(data=data)
        elif len(self) == 2:
            for data in self.datas:
                self.close(data=data)


def _run(make_market, make_prices, synthetic_rules, **overrides):
    markets = [make_market("sh600000", make_prices([PRICE] * 6))]
    options = dict(
        cash=CASH,
        max_positions=None,
        rules=synthetic_rules,
        universe_rules=UniverseRules(min_trading_days=0),
    )
    options.update(overrides)
    return run_portfolio_backtest(markets, BuyOnce, **options)


# --- 本票的核心：单标的买入不再因费用被拒 -----------------------------------


@pytest.mark.parametrize("max_positions", [None, 1])
def test_a_lone_position_is_not_rejected_for_want_of_fees(
    make_market, make_prices, synthetic_rules, max_positions
):
    """``max_positions`` 省略或为 1 时，买单必须**成交**，而不是以 ``Margin`` 被拒。

    修掉之前这里恒为零成交：``int(100000 ÷ 1 ÷ 10) = 10000`` 股，成交金额恰好 100000，
    再加 2 元过户费就超出现金。两个参数值都测，因为「省略」是最自然的用法——而它恰好
    是最坏的那个。
    """
    result = _run(make_market, make_prices, synthetic_rules, max_positions=max_positions)

    assert len(result.trades) == 1, "这笔买单应当成交"
    assert "Margin" not in set(result.rejected["status"]), "不该再因资金不足被拒"


def test_the_buy_leaves_room_for_the_fill_price_and_the_transfer_fee(
    make_market, make_prices, synthetic_rules
):
    """**用规则表独立核算**：按最坏成交价算，成交金额 + 过户费 ≤ 现金。

    刻意不从 sizer 里取数（那是自己验自己），而是照规则表再算一遍。

    「最坏成交价」= ``定量价 × (1 + 当日板块限幅)``：订单在本根收盘定量、在**下一根**成交，
    而下一根的价格受「前收盘 × (1 + 限幅)」约束。``synthetic_rules`` 的沪主板限幅是 0.10。
    """
    result = _run(make_market, make_prices, synthetic_rules)
    fill = result.trades.iloc[0]

    worst_price = PRICE * (1 + PRICE_LIMIT)
    gross = fill["size"] * worst_price
    assert gross + gross * TRANSFER_FEE_RATE <= CASH

    # 让这条断言有意义：**留余地之前**的算法在这里本就会超支。
    naive = int(CASH / PRICE)
    naive_gross = naive * worst_price
    assert naive_gross + naive_gross * TRANSFER_FEE_RATE > CASH, "旧算法本就会超，故本测试有意义"
    assert fill["size"] == int(CASH / worst_price), "按最坏成交价算出的上界"


def test_a_rising_next_bar_does_not_reject_the_buy(make_market, make_prices, synthetic_rules):
    """**本票的核心**：下一根价格比定量价高时，买单仍应成交。

    构造：定量价 10.0，下一根 10.5（+5%，在沪主板 10% 限幅内，故不是一字板、不会被挡）。
    不留余地时按 ``int(100000 / 10) = 10000`` 股定量，而成交价 10.5 使金额达 105000 →
    ``Margin`` 被拒。留余地后按 ``10.0 × 1.10`` 定量，成交在 10.5 时仍在预算内。

    两条都跑，好让「留余地确实解决了问题」有对照，而不是只断言现状。对照那条用默认 sizer
    加 ``headroom=0.0``——``sizer_options`` 在默认 sizer 那条路径上同样生效，故留余地的比例
    是可配置的。
    """
    markets = [make_market("sh600000", make_prices([10.0, 10.5, 10.5, 10.5, 10.5, 10.5]))]

    def run(**sizer_options):
        return run_portfolio_backtest(
            markets,
            BuyOnce,
            cash=CASH,
            max_positions=None,
            rules=synthetic_rules,
            universe_rules=UniverseRules(min_trading_days=0),
            sizer_options=sizer_options or None,
        )

    without = run(headroom=0.0)
    with_headroom = run()  # 默认：按规则表查当日板块限幅

    assert "Margin" in set(without.rejected["status"]), "不留余地时本就该被拒，故对照成立"
    assert len(with_headroom.trades) == 1, "留余地后应当成交"
    assert "Margin" not in set(with_headroom.rejected["status"])

    fill = with_headroom.trades.iloc[0]
    assert fill["price"] == 10.5, "成交价确实是下一根的价格"
    expected = int(CASH / (10.0 * (1 + PRICE_LIMIT)))
    assert fill["size"] == expected, "按最坏成交价定量"
    assert fill["size"] * fill["price"] <= CASH


def test_a_higher_commission_buys_strictly_fewer_shares(make_market, make_prices, synthetic_rules):
    """买入数量对费率**单调敏感**——钉住「费用真被算进去了」。

    只断言「不超支」是不够的：把数量一律减半也能不超支。真正要钉的是**它随费率变化**。
    """
    free = _run(make_market, make_prices, synthetic_rules)
    charged = _run(
        make_market,
        make_prices,
        synthetic_rules,
        commission=0.01,
        commission_mode="all_in",
    )

    assert len(free.trades) == 1 and len(charged.trades) == 1
    assert charged.trades.iloc[0]["size"] < free.trades.iloc[0]["size"]


def test_a_price_that_cannot_be_afforded_yields_no_order(make_market, make_prices, synthetic_rules):
    """一股都买不起时下 0 股的单——不提交订单，也不报错。"""
    markets = [make_market("sh600000", make_prices([PRICE] * 6))]

    result = run_portfolio_backtest(
        markets,
        BuyOnce,
        cash=5.0,  # 不足一股，连费用都不用谈
        max_positions=None,
        rules=synthetic_rules,
        universe_rules=UniverseRules(min_trading_days=0),
    )

    assert len(result.trades) == 0
    assert len(result.rejected) == 0


def test_orders_pending_in_the_same_bar_still_count_as_taken_slots(
    make_market, make_prices, zero_cost_rules
):
    """**同一 tick 里连下几笔买单时，挂单中的也要占名额。**

    只数已成交持仓是不够的：持仓要等成交后才更新，于是同一根 K 线里的每笔订单都会看到同一个
    ``held``、各自按「剩余名额」足额定量，合计必然超出现金——最先提交的成交，其余全部以
    ``Margin`` 被拒。

    实测（名额 2、同一 tick 提交 4 笔、现金 10 万）：每笔定 45,450 元 → 合计 181,800 →
    2 笔成交、**2 笔 `Margin`**。真实回测里这类拒单占全部下单的 **61%**。

    修好之后：第 3、4 笔看到名额已满，返回 0 股 → **不提交订单**，于是既不成交也不被拒。
    """
    import backtrader as bt

    from mbt.data.market import MarketData

    class BuyAll(bt.Strategy):
        """每个 tick 对所有未持仓的标的下单——组合策略的自然写法。"""

        def next(self):
            for data in self.datas:
                if self.getposition(data).size:
                    continue
                self.buy(data=data)

    markets = [
        MarketData(symbol=symbol, prices=make_prices([10.0] * 10), events=())
        for symbol in ("sh600000", "sh600001", "sh600002", "sh600003")
    ]

    result = run_portfolio_backtest(
        markets,
        BuyAll,
        cash=100_000.0,
        max_positions=2,
        rules=zero_cost_rules,
        universe_rules=UniverseRules(min_trading_days=0),
    )

    assert len(result.trades) == 2, "名额 2，就只该成交 2 笔"
    assert len(result.rejected) == 0, "多出来的两笔应当**不下单**，而不是下了被拒"

    spent = result.trades[result.trades["size"] > 0]["value"].sum()
    assert spent <= 100_000.0, "两笔合计不得超过现金"


def test_a_pending_buy_does_not_block_the_next_bar(make_market, make_prices, zero_cost_rules):
    """挂单**成交之后**就不再占名额——下一根仍能继续建仓。

    这条防的是「把挂单数记死了」这类改法：`broker.orders` 里保留着已成交的历史订单，故必须
    用 ``alive()`` 过滤，否则名额会被永久占满、策略从此再也买不进。
    """
    import backtrader as bt

    from mbt.data.market import MarketData

    class BuyOnePerBar(bt.Strategy):
        """每根只买一只还没持仓的——多根 K 线逐步建仓。"""

        def next(self):
            for data in self.datas:
                if self.getposition(data).size:
                    continue
                self.buy(data=data)
                return

    markets = [
        MarketData(symbol=symbol, prices=make_prices([10.0] * 10), events=())
        for symbol in ("sh600000", "sh600001", "sh600002")
    ]

    result = run_portfolio_backtest(
        markets,
        BuyOnePerBar,
        cash=100_000.0,
        max_positions=3,
        rules=zero_cost_rules,
        universe_rules=UniverseRules(min_trading_days=0),
    )

    assert len(result.trades) == 3, "三只都该买到（每根买一只）"


# --- 卖出仍由策略决定 ----------------------------------------------------------


def test_staggered_entries_are_equal_weight_not_back_loaded(
    make_market, make_prices, zero_cost_rules
):
    """**分次建仓时每个名额的分到的钱应当相同**——不能把剩下的现金全压给最后一个名额。

    构造：5 只标的、`max_positions=5`，但每根 K 线只买一只（于是现金逐次减少）。

    旧口径 ``现金 ÷ 剩余名额`` 在这里会失效：最后一笔看到「剩 1 个名额、手上还有 4/5 的
    现金」，于是**把全部剩余现金压上去**——实测单只占到组合的 **36.8%**，而等权应为 20%。
    新口径按 ``组合总值 ÷ max_positions`` 定额，故每笔都约 20%。
    """
    import backtrader as bt

    from mbt.data.market import MarketData

    class OnePerBar(bt.Strategy):
        """每根 K 线只买一只还没持仓的——多根 K 线逐步建仓。"""

        def next(self):
            for data in self.datas:
                if self.getposition(data).size:
                    continue
                self.buy(data=data)
                return

    symbols = ["sh600000", "sh600001", "sh600002", "sh600003", "sh600004"]
    markets = [
        MarketData(symbol=symbol, prices=make_prices([10.0] * 20), events=()) for symbol in symbols
    ]

    result = run_portfolio_backtest(
        markets,
        OnePerBar,
        cash=100_000.0,
        max_positions=5,
        rules=zero_cost_rules,
        universe_rules=UniverseRules(min_trading_days=0),
    )

    equity = result.equity_curve
    buys = result.trades[result.trades["size"] > 0].sort_values("date")
    assert len(buys) == 5, "五只都该买到"

    weights = []
    for row in buys.itertuples(index=False):
        stamp = pd.Timestamp(row.date)
        weights.append(abs(float(row.value)) / float(equity.asof(stamp)))

    assert min(weights) > 0.15, f"最小的仓位 {min(weights):.1%} 太小——该约 20%"
    assert max(weights) < 0.25, f"最大的仓位 {max(weights):.1%} 太大——该约 20%"
    # 新口径的特征是**平坦**：每个名额分到等额一份，故逐笔差异很小。
    assert (
        max(weights) - min(weights) < 0.03
    ), f"各笔仓位该基本相等，实测 {[round(w, 4) for w in weights]}"


def test_the_legacy_budget_back_loads_the_last_slot(make_market, make_prices, zero_cost_rules):
    """对照：旧口径 ``equal_weight=False`` **确实**会把钱压给最后一个名额。

    留着这条是为了让「为什么改口径」有据可查——否则新旧差异无从归因（正像阈值那几轮）。
    """
    import backtrader as bt

    from mbt.data.market import MarketData

    class OnePerBar(bt.Strategy):
        def next(self):
            for data in self.datas:
                if self.getposition(data).size:
                    continue
                self.buy(data=data)
                return

    markets = [
        MarketData(symbol=symbol, prices=make_prices([10.0] * 20), events=())
        for symbol in ("sh600000", "sh600001", "sh600002", "sh600003", "sh600004")
    ]

    result = run_portfolio_backtest(
        markets,
        OnePerBar,
        cash=100_000.0,
        max_positions=5,
        rules=zero_cost_rules,
        universe_rules=UniverseRules(min_trading_days=0),
        sizer_options={"equal_weight": False},
    )

    equity = result.equity_curve
    buys = result.trades[result.trades["size"] > 0].sort_values("date")
    weights = [
        abs(float(row.value)) / float(equity.asof(pd.Timestamp(row.date)))
        for row in buys.itertuples(index=False)
    ]

    # 旧口径的特征是**逐笔递增**（后面的名额分到更多），而不是某个具体阈值——阈值依赖
    # 现金与价格路径，构造起来脆弱。实测这一组是 18.2% → 21.9%，单调上升。
    assert weights == sorted(weights), f"旧口径该逐笔递增，实测 {weights}"
    assert weights[-1] > weights[0] * 1.15, f"最后一笔该明显大于第一笔，实测 {weights}"


def test_selling_is_still_left_to_the_strategy(make_market, make_prices, synthetic_rules):
    """sizer 不插手卖出：数量由策略决定（这里是全部持仓）。

    这条防的是「修买入时顺手把卖出也改了」——卖出数量与预算无关。
    """
    markets = [make_market("sh600000", make_prices([PRICE] * 6))]

    result = run_portfolio_backtest(
        markets,
        BuyThenSell,
        cash=CASH,
        max_positions=None,
        rules=synthetic_rules,
        universe_rules=UniverseRules(min_trading_days=0),
    )

    fills = result.trades.sort_values("date")
    assert len(fills) == 2
    bought, sold = fills.iloc[0], fills.iloc[1]
    assert sold["size"] == -bought["size"], "卖出数量应当正好是买入的全部持仓"


def test_the_boundary_is_exactly_the_affordable_maximum(make_market, make_prices, synthetic_rules):
    """边界精确：按**最坏成交价**再多买**一股**就买不起。

    这条把「二分找的是最大可行整数」钉住——若有人改回「乘 0.999 的安全系数」，这里会因为
    少买而失败。
    """
    result = _run(make_market, make_prices, synthetic_rules)
    size = result.trades.iloc[0]["size"]
    worst_price = PRICE * (1 + PRICE_LIMIT)

    def affordable(n):
        gross = n * worst_price
        return gross + gross * TRANSFER_FEE_RATE <= CASH

    assert affordable(size), "当前数量必须买得起"
    assert not affordable(size + 1), "再多一股就该买不起了——故取的是上界"
