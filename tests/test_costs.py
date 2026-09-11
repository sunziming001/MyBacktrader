"""验证费用模型：手续费、印花税、过户费、滑点。

关键是费用必须**在撮合时**进入现金流，而不是事后从净值里扣——否则可用资金算错，
持仓数量会随之算错，回测结果与实际不符。

期望值全部手算。价格恒为 10.00 以剔除行情损益，于是期末资金 = 期初资金 − 全部费用。
费率是百分数，浮点下「精确相等」无意义，故用 approx 并给出容差。
"""

import backtrader as bt
import pandas as pd
import pytest

from mbt.backtest import run_backtest

SYMBOL = "sh600000"  # 沪主板

# 费率取自合成规则表（tests/fixtures/rules/synthetic.toml）
COMMISSION = 0.0003
COMMISSION_MIN = 5.0


class BuyThenSell(bt.Strategy):
    """第 0 根买、第 2 根卖。日线下分别在第 1、3 根成交。"""

    def __init__(self):
        self.bar = 0

    def next(self):
        if self.bar == 0:
            self.buy(size=100_000)
        elif self.bar == 2:
            self.sell(size=100_000)
        self.bar += 1


def _run(start, make_prices, synthetic_rules, cash=1_100_000.0, **kwargs):
    prices = make_prices([10.0] * 6, start=start)
    # 默认按「全佣」口径：本文件的期望值都是「费率 + 印花税 + 过户费」，
    # 不含经手费与证管费。net 口径由专门测试覆盖。
    kwargs.setdefault("commission_mode", "all_in")
    return run_backtest(
        prices,
        symbol=SYMBOL,
        strategy=BuyThenSell,
        cash=cash,
        rules=synthetic_rules,
        commission=COMMISSION,
        commission_min=COMMISSION_MIN,
        **kwargs,
    )


def test_costs_before_the_regime_changes(make_prices, synthetic_rules):
    """2022 年初：过户费 0.00002、印花税 0.001。

    买入（第 1 根，2022-01-04）：手续费 0.0003×1e6=300，过户费 0.00002×1e6=20，共 320。
    卖出（第 3 根，2022-01-06）：手续费 300 + 印花税 0.001×1e6=1000 + 过户费 20，共 1320。
    价格恒定，故期末资金 = 1,100,000 − 320 − 1320 = 1,098,360。
    """
    result = _run("2022-01-03", make_prices, synthetic_rules)

    assert result.final_value == pytest.approx(1_098_360.0, abs=1e-6)

    buy, sell = result.trades.iloc[0], result.trades.iloc[1]
    assert buy["date"] == pd.Timestamp("2022-01-04")
    assert buy["commission"] == pytest.approx(320.0, abs=1e-6)
    assert sell["date"] == pd.Timestamp("2022-01-06")
    assert sell["commission"] == pytest.approx(1320.0, abs=1e-6)


def test_costs_after_the_regime_changes(make_prices, synthetic_rules):
    """2024 年：过户费降至 0.00001、印花税减半为 0.0005。

    买入（2024-01-04）：300 + 10 = 310。
    卖出（2024-01-08）：300 + 500 + 10 = 810。
    期末资金 = 1,100,000 − 310 − 810 = 1,098,880。

    与上一支测试的差额即为「按成交日查表」的直接后果：同样的交易，日期不同，费用不同。
    """
    result = _run("2024-01-03", make_prices, synthetic_rules)

    assert result.final_value == pytest.approx(1_098_880.0, abs=1e-6)

    buy, sell = result.trades.iloc[0], result.trades.iloc[1]
    assert buy["commission"] == pytest.approx(310.0, abs=1e-6)
    assert sell["commission"] == pytest.approx(810.0, abs=1e-6)


def test_stamp_duty_is_charged_on_sell_only(make_prices, synthetic_rules):
    """印花税只由卖出方缴纳，买入记录里不得含它。"""
    result = _run("2024-01-03", make_prices, synthetic_rules)

    buy, sell = result.trades.iloc[0], result.trades.iloc[1]
    # 买入只有手续费 + 过户费；卖出多出印花税 500
    assert sell["commission"] - buy["commission"] == pytest.approx(500.0, abs=1e-6)


def test_commission_minimum_applies_to_small_trades(make_prices, synthetic_rules):
    """小额成交按单笔最低手续费收取，而非按费率。

    买 100 股 × 10.00 = 成交额 1000：费率算出 0.3，但最低 5 元生效。
    过户费 0.00002×1000 = 0.02。故费用合计 5.02。

    该策略只买不卖，故期末仍持仓：净值 = 1,000,000 − 5.02（费用），
    买入占用的 1000 只是现金换成持仓，不减少净值。
    """

    class SmallBuy(bt.Strategy):
        def next(self):
            if not self.position:
                self.buy(size=100)

    prices = make_prices([10.0] * 3, start="2022-01-03")
    result = run_backtest(
        prices,
        symbol=SYMBOL,
        strategy=SmallBuy,
        cash=1_000_000.0,
        rules=synthetic_rules,
        commission=COMMISSION,
        commission_min=COMMISSION_MIN,
        commission_mode="all_in",
    )

    assert result.trades.iloc[0]["commission"] == pytest.approx(5.02, abs=1e-9)
    assert result.final_value == pytest.approx(1_000_000.0 - 5.02, abs=1e-6)


def test_slippage_worsens_the_fill_price(synthetic_rules):
    """滑点让成交价劣化：买入价高于开盘价。

    K 线必须有振幅——引擎会对滑点后的价格做当日上下限校验（``slip_match``），
    以高于当日最高价的价位成交是不允许的，故 ``o==h==l==c`` 的数据测不出滑点。
    """

    class BuyOnce(bt.Strategy):
        def next(self):
            if not self.position:
                self.buy(size=100)

    prices = pd.DataFrame(
        {
            "open": [10.0, 10.0, 10.0],
            "high": [10.5, 10.5, 10.5],
            "low": [9.8, 9.8, 9.8],
            "close": [10.2, 10.2, 10.2],
            "volume": [1000, 1000, 1000],
        },
        index=pd.bdate_range("2022-01-03", periods=3),
    )
    result = run_backtest(
        prices,
        symbol=SYMBOL,
        strategy=BuyOnce,
        cash=1_000_000.0,
        rules=synthetic_rules,
        slippage=0.01,
    )

    # 第 1 根开盘 10.00，滑点 1% → 10.10，未越过当日最高 10.50，故成交于 10.10
    assert result.trades.iloc[0]["price"] == pytest.approx(10.10, abs=1e-9)


# --- 佣金口径：全佣 / 净佣 ---


def test_commission_mode_is_required_when_commission_is_positive(make_prices, synthetic_rules):
    """填了费率就必须说明口径——两种口径的成本不同，替你猜会静默算错。"""
    with pytest.raises(ValueError, match="commission_mode"):
        run_backtest(
            make_prices([10.0] * 3, start="2022-01-03"),
            symbol=SYMBOL,
            strategy=BuyThenSell,
            cash=1_000_000.0,
            rules=synthetic_rules,
            commission=COMMISSION,
        )


def test_commission_mode_is_not_needed_when_commission_is_zero(make_prices, synthetic_rules):
    """零费率时无口径可言，不得强迫调用方传参。"""
    result = run_backtest(
        make_prices([10.0] * 3, start="2022-01-03"),
        symbol=SYMBOL,
        strategy=BuyThenSell,
        cash=1_000_000.0,
        rules=synthetic_rules,
    )

    # 价格恒定且费用为零，故净值不动——这条测试只关心「不传口径也能跑起来」
    assert result.final_value == pytest.approx(1_000_000.0)


def test_net_mode_adds_handling_and_regulatory_fees(make_prices, synthetic_rules):
    """净佣口径下另行叠加经手费与证管费；全佣口径下不得叠加（否则重复计费）。

    合成表里沪主板经手费 0.00005、证管费 0.00002，合计 0.00007。
    成交额 1e6 时，买入多出 70，卖出再多出 70。
    """
    all_in = _run("2022-01-03", make_prices, synthetic_rules, commission_mode="all_in")
    net = _run("2022-01-03", make_prices, synthetic_rules, commission_mode="net")

    buy_all_in, sell_all_in = all_in.trades.iloc[0], all_in.trades.iloc[1]
    buy_net, sell_net = net.trades.iloc[0], net.trades.iloc[1]

    assert buy_net["commission"] - buy_all_in["commission"] == pytest.approx(70.0, abs=1e-6)
    assert sell_net["commission"] - sell_all_in["commission"] == pytest.approx(70.0, abs=1e-6)

    # 净佣口径的期末资金正好少了两笔规费
    assert all_in.final_value - net.final_value == pytest.approx(140.0, abs=1e-6)
