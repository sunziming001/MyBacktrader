"""过滤信号：对**单一标的**的是/否判断（``CONTEXT.md``），票据 #5。

两条约定在此锁定：

1. 输出是**纯 ``bool``** 的标的宽表，缺失一律取 ``False``——语义是「不合格」。
   不引入 nullable boolean：下游（回测、选股）要的是「能不能买」这个二值答案，
   把「未知」再传下去只会让每个消费者各写一遍填充逻辑。
2. 判断只在**已知**上成立。窗口不足或输入缺失时不给乐观答案，一律 ``False``。
"""

from __future__ import annotations

import pandas as pd
import pytest

from mbt.signals import above_ma, ma_cross_up, new_high, rising_streak, volume_surge


def test_new_high_is_strict_so_exactly_matching_the_prior_high_does_not_count(symbol_frame):
    """恰好等于前 n 日最高价**不算**新高——边界在此，差一分才算越出。"""
    prices = symbol_frame({"sh600000": [10.0, 11.0, 12.0, 12.0, 12.01]})

    got = new_high(prices, n=3)["sh600000"]

    assert got.tolist() == [False, False, False, False, True]


def test_new_high_never_fires_when_the_window_has_no_known_price(symbol_frame):
    """全为缺失值的窗口不产生信号，也不被当成「低于现值」而放行（缺口纪律）。"""
    nan = float("nan")
    prices = symbol_frame({"sh600000": [nan, nan, nan, nan]})

    got = new_high(prices, n=3)

    assert got["sh600000"].tolist() == [False, False, False, False]


def test_one_missing_bar_inside_the_window_suppresses_the_signal(symbol_frame):
    """窗口里**有一根**缺失即不足以判定，故不给信号。

    这一条比「全为缺失」更紧：它同时挡住「把缺失填成 0」与「把缺失跳过、用其余 n-1 根
    凑一个最大值」两种写法——后者会把停牌期当成「没有更高的价」，从而凭空造出新高。
    """
    nan = float("nan")
    prices = symbol_frame({"sh600000": [10.0, nan, nan, 20.0]})

    got = new_high(prices, n=3)["sh600000"]

    assert got.tolist() == [False, False, False, False]


def test_above_ma_compares_the_close_with_its_own_trailing_average(symbol_frame):
    """站上均线：当根收盘高于 n 日均线；窗口未满时不给答案。"""
    prices = symbol_frame({"sh600000": [10.0, 11.0, 12.0, 11.0]})

    got = above_ma(prices, n=2)["sh600000"]

    assert got.tolist() == [False, True, True, False]


def test_ma_cross_up_fires_on_the_crossing_bar_only(symbol_frame):
    """均线上穿是**事件**而非状态：越过的那一根为真，其后继续在上方则为假。"""
    prices = symbol_frame({"sh600000": [10.0, 9.0, 8.0, 9.5, 9.6]})

    got = ma_cross_up(prices, n=2)["sh600000"]

    assert got.tolist() == [False, False, False, True, False]


def test_rising_streak_needs_n_increases_hence_n_plus_one_bars(symbol_frame):
    """连续上涨 n 日 = 最近 n 个交易日**每一次**收盘都高于前一日，故需要 n+1 根 K 线。

    取 n=3、六根 K 线：[1,2,3,4,4,5] 的日间变动是 [—,↑,↑,↑,平,↑]。第一根合格的
    K 线在第 4 根（索引 3，"1→4" 那一段），此前不足以判定，索引 4 因持平而中断。
    """
    prices = symbol_frame({"sh600000": [1.0, 2.0, 3.0, 4.0, 4.0, 5.0]})

    got = rising_streak(prices, n=3)["sh600000"]

    assert got.tolist() == [False, False, False, True, False, False]


def test_volume_surge_is_strict_so_exactly_k_times_the_average_does_not_count(symbol_frame):
    """放量 k 倍：当根量必须**严格高于**前 n 日均量的 k 倍——恰好 k 倍不算。

    第 4 根 200 恰为前 3 日均量 100 的 2 倍，故为假；第 5 根 300 高于其基准
    mean(100,100,200)=133.3 的 2 倍（266.7），故为真。
    """
    volumes = symbol_frame({"sh600000": [100.0, 100.0, 100.0, 200.0, 300.0]})

    got = volume_surge(volumes, k=2.0, n=3)["sh600000"]

    assert got.tolist() == [False, False, False, False, True]


def test_volume_surge_uses_the_prior_average_not_one_including_the_current_bar(symbol_frame):
    """基准取自**前 n 根**：含当根会把当根的放量本身算进基准，放量越猛越难触发。"""
    volumes = symbol_frame({"sh600000": [100.0, 100.0, 100.0, 400.0]})

    got = volume_surge(volumes, k=3.0, n=3)["sh600000"]

    # 400 > 3 × 100 → 真。若基准含当根则为 mean(100,100,400)=200，400 > 600 为假。
    assert got.tolist() == [False, False, False, True]


def test_filters_return_plain_bools_never_missing(symbol_frame):
    """过滤器输出纯 ``bool``：缺失被解释为「不合格」，不是「未知」。"""
    nan = float("nan")
    prices = symbol_frame({"sh600000": [nan, nan, 12.0, 13.0], "sz000001": [1.0, 2.0, 3.0, 4.0]})

    got = above_ma(prices, n=2)

    assert list(got.columns) == ["sh600000", "sz000001"]
    assert got.index.equals(prices.index)
    assert all(dtype.kind == "b" for dtype in got.dtypes)
    assert bool(got.loc[got.index[0], "sz000001"]) is False
    assert bool(got.loc[got.index[0], "sh600000"]) is False


def test_a_filter_rejects_a_non_monotonic_index_like_the_indicators_do(symbol_frame):
    """契约由所有公开函数共同遵守，过滤器也不例外。"""
    idx = pd.to_datetime(["2024-01-03", "2024-01-02", "2024-01-04"])
    prices = pd.DataFrame({"sh600000": [10.0, 11.0, 12.0]}, index=idx)

    with pytest.raises(ValueError, match="升序"):
        new_high(prices, n=2)
