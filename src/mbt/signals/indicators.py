"""指标：由**单一标的**的价量序列算出、本身不含任何判断的数值序列（``CONTEXT.md``）。

它们是过滤信号与排序因子的**构件**——「放量」与「站上均线」都组合这里的滚动窗口，而不是
各写一份。窗口一律取「当根及其之前 n-1 根」且 ``min_periods=n``：既不含未来信息，也不把
不足窗口的短缺当成数值——缺失就让它缺失（ADR-0005 的缺口纪律）。

本模块多数函数取**单字段**的标的宽表；``atr`` 与 ``kdj`` 需要**跨字段**，故它们取
:class:`~mbt.data.panel.Panel`。跨字段的信号**返回标的宽表**（ADR-0009）：消费方向与其余的
指标完全一样（列 = 时序、行 = 截面），只是输入多了两张表。
"""

from __future__ import annotations

from typing import NamedTuple

import numpy as np
import pandas as pd

from mbt.data.panel import Panel
from mbt.signals._symbol_frame import check_symbol_frame


def sma(prices: pd.DataFrame, n: int) -> pd.DataFrame:
    """n 日简单移动平均，含当根；窗口不足处为缺失值。"""
    return check_symbol_frame(prices).rolling(n, min_periods=n).mean()


def _smooth_frame(prices: pd.DataFrame, alpha: float) -> pd.DataFrame:
    """对整张标的宽表做**分段递推平滑**：段首 ``Y = X``，其后 ``Y = α·X + (1 − α)·Y_prev``。

    缺失把序列切成若干段，每段以该段**第一个可用值**播种；缺失处一律是缺失。

    **为什么按行推进，不按列。** 递推只能沿时间走，但**列之间互不相干**，故按行推进时每一行
    可以一次算完全部标的（numpy 向量），Python 层的迭代因此从「列数 × 行数」降到「行数」。
    原先按列做（``DataFrame.apply``）时，5,451 个标的各要一次 pandas 重活（``notna`` /
    ``cumsum`` / ``groupby`` / ``ewm``），实测单层 EMA 在白线（双层）上就是分钟量级。
    实测白线（467 列 × 7,206 行）：**3.64s → 0.18s（20×）**。

    **数值上不是逐位相同，但等价到末位。** ``α·x + (1−α)·y`` 与 pandas ``ewm`` 的内部写法
    在最后一两个 ULP 上不同：实测最大相对差 **4.7e-16**（白线）、**3.6e-16**（单层 EMA），
    而 B1 的 ``trend`` 门（白线 > 黄线）在 285,768 格上**零翻转**、``low_j`` 门亦零翻转
    （见 ``.scratch/smooth_dump_or_compare.py``）。这是**接受**的：递推给的是同一序列，
    差在浮点舍入，而判据是「比大小」不是「逐位比对」——**接受它的必要条件**是判据零翻转，
    该取舍见 ``docs/adr/0015-recursive-indicators-vectorize-with-ulp-drift.md``。

    .. note::

        不能整列调一次 ``ewm(adjust=False)``：它遇到缺失会**沿用前值**——``[1, 2, NaN, 4, 5]``
        在 ``span=2`` 下给出 ``[1, 1.667, 1.667, 3.667, 4.556]``，第 3 位不是缺失而是上一根的
        值。那等于把停牌日伪造成一个**有**指标值的交易日（ADR-0005）。故段必须分开。
    """
    frame = check_symbol_frame(prices)
    values = frame.to_numpy(dtype=float)
    rows, width = values.shape
    out = np.full((rows, width), np.nan)
    # `prev` 为 NaN 表示「当前处在段首」（段首或刚跨过一个缺口）。
    prev = np.full(width, np.nan)
    for t in range(rows):
        row = values[t]
        seeded = np.where(np.isnan(prev), row, alpha * row + (1.0 - alpha) * prev)
        prev = np.where(np.isnan(row), np.nan, seeded)
        out[t] = prev
    return pd.DataFrame(out, index=frame.index, columns=frame.columns, dtype=float)


def ema(prices: pd.DataFrame, n: int) -> pd.DataFrame:
    """n 日指数移动平均，**通达信口径**：``EMA_t = (2·C_t + (n−1)·EMA_{t−1}) ÷ (n+1)``。

    与 :func:`sma` 的口径差别是**刻意的，不要对齐**：

    - ``sma`` 是尾随窗口均值，窗口不满处为**缺失**（``min_periods=n``）；
    - ``ema`` 是递推平滑，**没有「窗口填满」这一刻**。通达信与行情软件都以**首根**播种
      （``EMA₁ = C₁``）并一路递推；要对齐图上的线就必须照此口径。若也按 ``min_periods=n``
      处理，前 n−1 根会缺失，且首个非缺失值也与图上不同。

    代价要写明：它在前若干根的取值受**序列起点**影响——那不是数据的性质，是本函数的口径。
    故凡与长窗口均线（如 114 日均线）同时出现的信号，其可用起点由**那个长窗口**决定，
    本函数不额外设门槛。

    缺失处一律是缺失：没有收盘价就没有 EMA。缺口把序列切成若干段，每段以该段**第一个可用值**
    播种——与 :func:`atr` 是同一条缺口纪律，而**不是**「沿用前值续算」：把跨越停牌的递推连起来
    等于假装停牌期间也有收盘价（ADR-0005）。行情软件上没有缺口行，故只在**无缺口**的序列上
    两种口径一致——也就是说「字面复刻图上的线」这件事，在有空缺的标的身上并不成立。
    """
    return _smooth_frame(prices, 2.0 / (n + 1))


def rolling_max(prices: pd.DataFrame, n: int) -> pd.DataFrame:
    """n 日最高价，含当根；窗口不足处为缺失值。"""
    return check_symbol_frame(prices).rolling(n, min_periods=n).max()


def rolling_min(prices: pd.DataFrame, n: int) -> pd.DataFrame:
    """n 日最低价，含当根；窗口不足处为缺失值。

    与 :func:`rolling_max` 对称。喂 ``low`` 字段即得「近 n 根的**前低**」——止损位要用的量
    （见 ``CONTEXT.md`` 的**前低**）；喂收盘价得的是收盘价的低点，两者不是一回事。
    """
    return check_symbol_frame(prices).rolling(n, min_periods=n).min()


def white_line(prices: pd.DataFrame, n: int) -> pd.DataFrame:
    """**白线**：``EMA(EMA(C, n), n)``——双重指数平滑，行情软件图上的一条快线。

    名字取自图上线的颜色，不是算法性质；同名术语见 ``CONTEXT.md``。``n`` 由调用方给定
    （常见取 10）而不设默认值：默认值会让「用了哪个口径」从调用处消失。

    外层 EMA 吃到内层在缺口处留下的缺失，故缺口会把两层的递推都切开——与 :func:`ema`
    是同一条缺口纪律。
    """
    return ema(ema(prices, n), n)


def yellow_line(prices: pd.DataFrame, windows: tuple[int, ...]) -> pd.DataFrame:
    """**黄线**：若干条 ``MA(C, w)`` 的**等权均值**，行情软件图上的一条慢线。

    名字取自图上线的颜色，不是算法性质；同名术语见 ``CONTEXT.md``。``windows`` 由调用方
    给定（常见取 ``(14, 28, 57, 114)``）而不设默认值，理由同 :func:`white_line`。

    均值用**简单平均**而非加权：图上那四条线是等权的。任一条 MA 在当根缺失（窗口不满或
    数据有缺口），结果即缺失——NaN 会自然沿加法传播，不必额外判空。
    """
    if not windows:
        raise ValueError("windows 不能为空：至少需要一条均线才谈得上均值")
    return sum(sma(prices, w) for w in windows) / len(windows)


class KDJ(NamedTuple):
    """KDJ 的三条线，各是一条标的宽表。

    三条线**各自**都是合法的指标（列 = 时序、行 = 截面），故这个结构不破坏「指标返回
    标的宽表」的契约（ADR-0009）——它只是把一次计算里的三个结果一起交回，省得调用方
    为了拿 ``j`` 而把 ``k``、``d`` 算两遍。
    """

    k: pd.DataFrame
    d: pd.DataFrame
    j: pd.DataFrame


def kdj(panel: Panel, n: int, m1: int, m2: int) -> KDJ:
    """KDJ 指标，**通达信口径**：

    .. code-block:: text

        RSV_t = (C_t − LLV(L, n)) ÷ (HHV(H, n) − LLV(L, n)) × 100
        K_t   = SMA(RSV, m1, 1)      # 即 α = 1/m1 的递推平滑
        D_t   = SMA(K,   m2, 1)
        J_t   = 3·K_t − 2·D_t

    其中 ``SMA(X, N, M) = (M·X + (N−M)·Y_prev) ÷ N`` 是通达信的递推均值，与本模块的
    :func:`ema` 是同一族（``α = M/N``）——**不是**移动平均。用简单均线算出的 K 与图上不同，
    这条差别已被测试钉住。

    参数:
        panel: 须含 ``high`` / ``low`` / ``close`` 三个字段，且三者同日对齐
            （对齐由 :class:`~mbt.data.panel.Panel` 保证）。
        n / m1 / m2: 三个窗口，由调用方给出；KDJ 的常见取值为 ``9 / 3 / 3``，但这里不设
            默认值——默认值会让「用了哪个口径」从调用处消失。

    返回:
        :class:`KDJ`（``k`` / ``d`` / ``j`` 各为一条标的宽表）。``j`` 不属于 ``[0, 100]``，
        会跌破 0、也会超过 100，这是定义如此而非瑕疵。

    三类缺失一律**缺失**，不填充：

    - **窗口不满**（``min_periods=n``）：不足 n 根就无从谈最高/最低，故 RSV 缺失。与图上
      的差别要写明：行情软件的 ``LLV``/``HHV`` 在不足 n 根时按**已有的**几根算，故它从首根
      就有值；本项目的缺口纪律不给这种乐观答案。这段差异只影响序列开头 n−1 根，且递推会
      在约 m1×5 根内把差别衰减掉。
    - **分母为 0**：窗口内最高价等于最低价（一字板、长期停牌复牌）时 RSV 无定义。此处判
      缺失而**不是**沿用前值——与图上「平盘时 KDJ 保持前值」的观感不同，取的是「不猜」。
    - **数据缺口**：    缺失把序列切成若干段，每段以第一个可用值播种，缺口处保持缺失
    （与 :func:`_smooth_frame` 同一纪律）。
    """
    high = check_symbol_frame(panel["high"])
    low = check_symbol_frame(panel["low"])
    close = check_symbol_frame(panel["close"])

    highest = high.rolling(n, min_periods=n).max()
    lowest = low.rolling(n, min_periods=n).min()
    span = highest - lowest
    # 分母为 0 时 RSV 无定义：先算出比值、再把分母非正的位置判为缺失，
    # 免得 0/0 的 NaN 与「窗口不满」的 NaN 混为一谈。
    rsv = (close - lowest) / span * 100.0
    rsv = rsv.where(span > 0)

    k = _smooth_frame(rsv, 1.0 / m1)
    d = _smooth_frame(k, 1.0 / m2)
    return KDJ(k=k, d=d, j=3.0 * k - 2.0 * d)


#: 公式里写死的窗口与偏移。它们**刻意不是参数**：砖型图是一条具体的公式，换掉其中任何一个数
#: 就是**另一条线**，而两条线同名会让「用的哪个口径」从调用处消失。见 :func:`brick_line`。
_BRICK_WINDOW = 4
_BRICK_FAST_N = 4
_BRICK_SLOW_N = 6
_BRICK_FLOOR = 4.0


class BrickLine(NamedTuple):
    """砖型图连同它那两个**派生读数**，各为一条标的宽表。

    只有 ``line`` 是 ``CONTEXT.md`` 的**砖型图**；另外两条由它算出，是各自的术语：

    ===============  ==================================================================
    字段              术语（``CONTEXT.md``）与读法
    ===============  ==================================================================
    ``line``          **砖型图**：公式末句减 4、与 0 取大之后的那条线
    ``size``          **砖的大小**：``|line[t] − line[t−1]|``，两砖相等时为 0
    ``size_ratio``    ``size[t] ÷ size[t−1]``；前一根没有砖（大小为 0）时为**缺失**
    ===============  ==================================================================

    三条都**不含判断**（判据在 :func:`~mbt.signals.filters.red_brick` 与
    :func:`~mbt.signals.filters.green_brick`），故照常遵循指标契约；方向各不相同，故
    **不要**直接当排序因子用之前先看清哪一条朝哪边。

    ``size`` 取**绝对值**，故「砖的大小」恒为正——绿砖（严格下降）也不例外。这条不是顺手
    取的：它使「红砖的大小 ÷ 绿砖的大小」的分母**不可能为零**，而那个比值是排序因子之一。

    ``size_ratio`` 是**机械**读数，**不**先判红绿：只有当 ``line[t−1]`` 是绿砖、``line[t]``
    是红砖时，它才等于「红砖的大小 ÷ 绿砖的大小」。其余形态下它算的是别的比值——判据会把那些
    格子筛掉，故这里**不**替它预设形态（预设等于把判断塞进指标里）。前一根大小为 0（两根持平、
    或前一根在缺口上）时比值**无定义**，取缺失而**不**取无穷：那条砖根本不存在，除法没有对象。
    """

    line: pd.DataFrame
    size: pd.DataFrame
    size_ratio: pd.DataFrame


def brick_line(panel: Panel) -> BrickLine:
    """**砖型图**，照行情软件那条公式字面实现。

    公式（``VAR`` 编号照原样保留，便于与图上逐句对读）::

        VAR1A := (HHV(H, 4) − C) ÷ (HHV(H, 4) − LLV(L, 4)) × 100 − 90
        VAR2A := SMA(VAR1A, 4, 1) + 100
        VAR3A := (C − LLV(L, 4)) ÷ (HHV(H, 4) − LLV(L, 4)) × 100
        VAR4A := SMA(VAR3A, 6, 1)
        VAR5A := SMA(VAR4A, 6, 1) + 100
        VAR6A := VAR5A − VAR2A
        砖型图 := IF(VAR6A > 4, VAR6A − 4, 0)

    两个位置读数（``VAR1A`` 量「离窗口最高价多远」、``VAR3A`` 量「离窗口最低价多远」）各做
    通达信递推均值后相减，故这条线与 :func:`~mbt.signals.indicators.kdj` 是**同一族**、用
    同一个零件（:func:`_smooth_frame`）。

    参数:
        panel: 须含 ``high`` / ``low`` / ``close``，三者同日对齐（由 :class:`~mbt.data.panel.Panel`
            保证）。**不需要**成交量——量那一条是排序因子，不在本函数里。

    返回:
        :class:`BrickLine`。

    .. note::

        **两处 ``+100`` 相消，故这里直接相减。** ``VAR5A − VAR2A`` 展开后那两个 ``+100``
        一正一负消掉，故本函数算的是``SMA(SMA(VAR3A,6,1),6,1) − SMA(VAR1A,4,1)``。这不是
        化简求快——先加 100 再减回来会白白丢掉两个数的低位精度，而数学上两式恒等。

    .. note::

        **末句那个「与 0 取大」照字面保留**（ADR-0016）。它是本条线定义的一部分：接近底部时
        ``VAR6A ≤ 4`` 一律记 0，于是图上那段**既不红也不绿**，而一根从 0 抬到 3 的红砖与一根
        8 → 11 的红砖被判为**同高**（``size`` 都是 3）。这两条后果是认下的，不是瑕疵。

    .. note::

        **缺口处缺失，不取 0——这是对公式字面的**一处刻意偏离**。** 行情软件上没有缺口行，
        故它的 ``IF`` 把「算不出」判成假、给出 0；本项目按缺口纪律取**缺失**（ADR-0005）。
        差别是实质性的：取 0 会把一个停牌日伪造成「落在 0 底的一天」，于是复牌那天凭空多出
        一根砖，而它不会报错。

        同理，窗口内最高价等于最低价（一字板、长期停牌复牌）时分母为 0，那两个位置读数一律
        缺失——不是 0。

    .. note::

        前 3 根缺失：``HHV``/``LLV`` 都要 4 根，窗口不满即缺失，故递推从第 4 根播种。这里
        **不**照行情软件在不足 4 根时按已有的几根算——那是缺口纪律的另一面（``kdj`` 同）。

        递推的记忆按 ``(5/6)^t`` 衰减，故它自带一段约百根的暖机；序列起点对末位的影响到那里
        已落进浮点噪声。这段深度由**消费它的选股规则**自己声明——历史够不够深是规则的属性，
        本函数不猜。
    """
    high = check_symbol_frame(panel["high"])
    low = check_symbol_frame(panel["low"])
    close = check_symbol_frame(panel["close"])

    highest = high.rolling(_BRICK_WINDOW, min_periods=_BRICK_WINDOW).max()
    lowest = low.rolling(_BRICK_WINDOW, min_periods=_BRICK_WINDOW).min()
    span = highest - lowest

    near_high = (highest - close) / span * 100.0 - 90.0
    near_low = (close - lowest) / span * 100.0
    # 分母为 0 时两个读数都无定义：先算比值、再把分母非正的位置判为缺失，
    # 免得 0/0 的 NaN 与「窗口不满」的 NaN 混为一谈（与 `kdj` 同一处置）。
    near_high = near_high.where(span > 0)
    near_low = near_low.where(span > 0)

    var6 = _smooth_frame(_smooth_frame(near_low, 1.0 / _BRICK_SLOW_N), 1.0 / _BRICK_SLOW_N) - (
        _smooth_frame(near_high, 1.0 / _BRICK_FAST_N)
    )

    # 照字面的 IF(VAR6A>4, VAR6A−4, 0)：先减 4、再与 0 取大。
    floored = (var6 - _BRICK_FLOOR).where(var6 > _BRICK_FLOOR, 0.0)
    # 再补回缺口：`where` 的假分支（包括 NaN 处的假）已把缺失写成 0，故这一步不可省。
    line = floored.where(var6.notna())

    size = line.diff().abs()
    denominator = size.shift(1)
    # 前一根没有砖（大小为 0 或缺失）时比值无定义——取缺失，不取无穷。
    size_ratio = (size / denominator).where(denominator > 0)

    return BrickLine(line=line, size=size, size_ratio=size_ratio)


def volume_ratio(volumes: pd.DataFrame, n: int) -> pd.DataFrame:
    """成交量比：``当根成交量 ÷ 前 n 根均量``。

    基准取**前 n 根**而非含当根：含当根会把当根的放量本身算进基准，于是放量越猛基准越高、
    比值越小——正好与「放量」的语义相反（``volume_surge`` 判定放量即复用它）。

    .. warning::

        **它不是交易所口径的「量比」。** 官方量比是「当日开盘后**每分钟平均**成交量 ÷ 过去
        **5 个**交易日每分钟平均成交量」，是个**盘中**指标，用日线数据**无法忠实复现**。
        本项目刻意不实现量比，也刻意不用那个词——用一个有法定含义的词去指另一件事，正是
        本项目反复警惕的失真。
    """
    volumes = check_symbol_frame(volumes)
    prior_mean = volumes.rolling(n, min_periods=n).mean().shift(1)
    return volumes / prior_mean


def _wilder_smooth(true_range: pd.Series, n: int) -> pd.Series:
    """对一个标的的真实波幅做**分段** Wilder 平滑（``atr`` 的零件）。

    缺失把序列切成若干段，每段各自播种：一段需**连续** n 个真实波幅，其简单均值放在该段
    第 n 根，此前缺失；其后走 Wilder 递推。不足 n 个真实波幅的段整段缺失。

    这里**逐段**调用 ``ewm`` 而不是整列调用一次，是因为整列调用会连缺失处的毛病一起带来：
    ``ewm(adjust=False)`` 遇到缺失会**沿用前值**（把缺失位置填成上一次的状态），且会把缺口
    之后的第一个原始真实波幅当成新的起点——那等于报出一个未经平滑的值。逐段处理两个问题
    都不存在，且段内的递推仍由 ``ewm`` 在 C 层完成。
    """
    valid = true_range.notna()
    if not valid.any():
        return true_range

    run_id = (~valid).cumsum()
    out = pd.Series(np.nan, index=true_range.index, dtype=float)
    for _, run in true_range[valid].groupby(run_id[valid]):
        if len(run) < n:
            continue
        seeded = run.copy()
        seeded.iloc[: n - 1] = np.nan
        seeded.iloc[n - 1] = run.iloc[:n].mean()
        out.loc[run.index] = seeded.ewm(alpha=1.0 / n, adjust=False).mean()
    return out


def atr(panel: Panel, n: int) -> pd.DataFrame:
    """平均真实波幅（ATR），**Wilder 平滑**口径。

    真实波幅（TR）逐根为下三者的最大：

    .. code-block:: text

        high − low
        |high − 前收|
        |low  − 前收|

    真实波幅的参照收盘价取**最近一个可得**的收盘价。前一根停牌时它没有收盘价，而无论
    「凭空当 0」还是「让整根作废」都会丢掉复牌日的跳空——那恰恰是真实波幅要度量的东西。
    这**不是**被禁止的填充：价格序列本身不变、输出在缺失处也仍然缺失，这里只是给真实波幅
    挑一个参照。序列开头没有参照收盘价，退回 ``high − low``。

    ATR 以**连续 n 个真实波幅的简单均值**播种于该段的第 n 根，其后走 Wilder 递推：

    .. code-block:: text

        ATR_t = (ATR_{t-1} × (n − 1) + TR_t) ÷ n

    **缺失把序列切成若干段，每段各自播种。** 一段不足 n 个真实波幅即整段缺失；缺口之后
    需重新积累 n 个连续值才重新播种。缺失处一律是缺失——**不沿用**前值，因为那会把停牌日
    伪造成一个有波幅的交易日（ADR-0005 的缺口纪律）。

    口径必须写明而不是默认，因为「ATR」不加限定会被读成简单均线版，而两者数值不同：
    在 n=3 的样本上手算，Wilder 口径后两值为 1.3889 / 1.7593，简单均线版为 1.5 / 1.8333。

    参数:
        panel: 须含 ``high`` / ``low`` / ``close`` 三个字段，且三者同日对齐
            （对齐由 :class:`~mbt.data.panel.Panel` 保证）。
        n: 平滑窗口。

    返回:
        标的宽表。形态与 :func:`sma` 相同，故它可直接当排序因子用。
    """
    high = check_symbol_frame(panel["high"])
    low = check_symbol_frame(panel["low"])
    close = check_symbol_frame(panel["close"])

    reference_close = close.ffill().shift(1)
    true_range = np.maximum(
        np.maximum(high - low, (high - reference_close).abs()),
        (low - reference_close).abs(),
    )
    # 序列开头没有参照收盘价：三者的最大值全为缺失，故退回 high − low。
    true_range = (high - low).where(reference_close.isna(), true_range)

    return true_range.apply(_wilder_smooth, n=n)
