"""验证 Python 3.10 上 backtrader 的 ``collections.Iterable`` 缺陷已被修补。

背景：backtrader 1.9.78.123 的 ``lineiterator.py`` 在 ``bindlines()`` 中直接引用
``collections.Iterable``，而该别名在 Python 3.10 已被移除，导致这个公开 API 抛出
``AttributeError: module 'collections' has no attribute 'Iterable'``。

``bindlines`` 在 backtrader 内部无人调用，它是供自定义指标作者使用的公开 API，
因此本测试以一个自定义指标的形式使用它——即真实的公开用法。
"""

import backtrader as bt

import mbt.backtest  # noqa: F401  导入即应用兼容修补


class Binder(bt.Indicator):
    """自定义指标：把自己的线绑定到所属层。这是 ``bindlines`` 的公开用法。"""

    lines = ("probe",)

    def __init__(self):
        self.bindlines()


class Host(bt.Strategy):
    def __init__(self):
        Binder(self.data)


def test_custom_indicator_can_bind_lines(make_prices):
    """自定义指标调用 bindlines 后能正常跑完——修补前此处抛 AttributeError。"""
    cerebro = bt.Cerebro()
    cerebro.adddata(bt.feeds.PandasData(dataname=make_prices([10.0, 11.0, 12.0])))
    cerebro.addstrategy(Host)

    cerebro.run()


def test_removed_collections_alias_is_restored():
    """被 Python 3.10 移除的别名已可解析——``bindlines`` 依赖它。"""
    import collections
    import collections.abc

    assert collections.Iterable is collections.abc.Iterable
