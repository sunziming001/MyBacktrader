"""标的宽表契约（ADR-0009）。

信号层的每个公开函数都遵守同一形状：``DataFrame[日期 × 标的] → DataFrame[日期 × 标的]``。
回测按列取时序、选股按行取截面——**同一个函数，两个消费方向**（ADR-0001）。

注意它与数据层的**字段宽表**（交易日为索引、**字段**为列，一个标的一份）形状同形而语义相反，
故裸用「宽表」一词在本项目里是被禁止的（ADR-0009）：那种同名曾让 ``sma(单标的 OHLCV, n=2)``
静默返回一张把 5 个*字段*当成 5 个*标的*的均值表。

契约集中在这里而不是散在各函数开头，为的是让「什么算合法标的宽表」只有一处定义；将来加严
（如要求索引无重复）也只需改这一处。
"""

from __future__ import annotations

import pandas as pd


def check_symbol_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """校验并返回标的宽表本身。

    要求索引是**升序**的 ``DatetimeIndex``。乱序不是小事：信号层的时点正确性建立在
    「只用当根及其之前」之上，而 ``rolling`` 之类算子按**位置**推进——索引一旦乱序，
    算出来的就是别的日子。故报错而不是就地排序：静默重排会让调用方以为一切正常。

    刻意**不**校验缺失值：缺失是正常状态（停牌、次新股），由各函数按自己的语义处理
    （指标保留缺失，过滤器取「不合格」），而不是在入口一刀切。
    """
    if not isinstance(frame.index, pd.DatetimeIndex):
        raise ValueError(
            f"标的宽表的索引必须是 DatetimeIndex（交易日），收到 {type(frame.index).__name__}"
        )
    if not frame.index.is_monotonic_increasing:
        raise ValueError("标的宽表的索引必须按交易日升序；乱序会破坏「只用过去」的保证")
    return frame
