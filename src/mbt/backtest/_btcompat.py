"""Python 3.10 兼容修补：恢复 backtrader 依赖的 ``collections`` 别名。

backtrader 1.9.78.123 的 ``lineiterator.py`` 在公开 API ``bindlines()`` 中引用
``collections.Iterable``，而该别名在 Python 3.10 已被移除，导致调用抛
``AttributeError: module 'collections' has no attribute 'Iterable'``。
上游修复（PR #520）至今未合并，故由本项目在包内修补（ADR-0007）。

只补被移除**且被 backtrader 实际引用**的别名。经全量扫描确认仅 ``Iterable`` 一个：
``cerebro.py`` 与 ``writer.py`` 中的同类引用自带版本分支（``collectionsAbc``），无需修补；
其余 ``collections`` 用法（``OrderedDict`` / ``defaultdict`` / ``deque``）在 3.10 均仍存在。

修补是条件式的：在更早的 Python 上为无操作。
"""

import collections
import collections.abc

if not hasattr(collections, "Iterable"):
    collections.Iterable = collections.abc.Iterable
