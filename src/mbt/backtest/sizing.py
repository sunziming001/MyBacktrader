"""持仓分配：决定每个标的该分到多少资金（票据 #6）。

AC 要求「多标的共享资金池，有最大持仓数与仓位分配」，但**没有定义分配规则**。故这里给出
一个精确定义、能手算的默认实现，并把机制做成可插拔——「按排序因子加权」之类要等 #7 的
选股规则定了形状再说，现在猜就是空想。

这里的词按 `CONTEXT.md` 的分工用：**持仓**是「组合拿着哪些标的」，**仓位**是「其中每一只
占多大比例」。

## 默认口径：等权 + 按限幅留余地

设最大持仓（标的数）为 ``M``、**组合总值**为 ``V``、已占用（已成交持仓 + 挂单中的买单）为
``H``，本单可用资金为 ``C``（= 现金 − 挂单已占用），则本次买入的目标金额为::

    min(V ÷ M, C)

即**每个持仓名额分到等额一份**（``V ÷ M``），且不超过手上能用的现金。方向是「**买齐**」：
既有持仓涨了，这一份就跟着变大；跌了就变小——所以每个名额的**市值**趋于相等，而不是
「投入金额」相等。

之前的口径是 ``C ÷ 剩余名额``，它**只在同一 tick 一次买满时**才等于等权。分次建仓（某只
平仓后腾出名额）时它会把手上的**全部**现金压给最后一个名额——实测单只占到组合的 **36.8%**，
而等权应为 20%。传 ``equal_weight=False`` 可回到那个旧口径，便于对照。

**买入数量再按「最坏成交价」求上界**——不是按当根收盘价。理由：订单在**本根收盘定量、在下一根
成交**，而下一根的价格受「前收盘 × (1 + 涨跌幅限制)」约束，故最坏成交价是
``收盘价 × (1 + 当日板块限幅)``，限幅从规则表按板块与成交日查出（票据 #32）。

**代价是系统性少买约「限幅」那么多**（主板约 9%、北交所约 23%），这是**刻意承担**的：那份代价
是确定的、可解释的，而「因下一根涨价而整笔被拒」是不确定且静默的。留余地的比例可用
``headroom`` 覆盖，传 ``0.0`` 即回到按收盘价定量（那时请自己盯着拒单率）。

## 尚未建模

- **A 股「买入必须是 100 股整数倍」**（一手 = 100 股）。本模块不做整手取整，故产出的股数
  可能不是 100 的倍数。这是一处**已知的乐观偏差**（真实下单会被拒或改单），已在 README 的
  风险清单里登记。
- 单笔上限、按波动率加权、行业分散等更复杂的分配规则。
- **卖出挂单不释放名额**：一笔尚未成交的卖单本会腾出一个名额，但 sizer 不把它算进去。按
  「更少下单」估是稳妥的一侧（宁可少买，不可超支），代价是某些调仓日会晚一天才补上仓位。
"""

from __future__ import annotations

import backtrader as bt


class AStockSizer(bt.Sizer):
    """本项目两个 sizer 的公共底座：**挂单占用**、**最坏成交价**、**买得起的股数**。

    抽出来是因为它们与「每个名额分多少钱」无关——那是唯一的差别（见 :class:`EqualWeightSizer`
    与 :class:`FixedAmountSizer`，两个子类各自只裁决 :meth:`_slots` 与 :meth:`_budget`）。
    而这三段每一段都是踩出来的（下面各自有实测记录），分成两份必然漂开，且漂开时**两边都不会
    报错**。

    它同时是一个**标记**：引擎据此知道这个 sizer 接受 ``rules`` 参数，于是把规则表递进来。
    别的 sizer（如 ``bt.sizers.FixedSize``）不认这个参数，递进去会当场报错。
    """

    params = (("rules", None), ("headroom", None))

    def _getsizing(self, comminfo, cash, data, isbuy):
        if not isbuy:
            # 卖出交由策略决定数量（通常是全部持仓），sizer 不插手。
            return self.broker.getposition(data).size

        held = sum(1 for position in self.broker.positions.values() if position.size)
        # **挂单中的买单既占名额、也占现金。** 只数已成交持仓是不够的：同一根 K 线里连下几笔
        # 买单时，持仓要等成交后才更新，于是每笔都会看到同一个 `held` 与同一个 `cash`、各自
        # 按「剩余名额」足额定量，合计必然超出——结果是最先提交的成交，其余以 ``Margin`` 被拒。
        #
        # 实测（名额 2、同一 tick 提交 4 笔、现金 10 万）：每笔定 45,450 元 → 合计 181,800
        # → 2 笔成交、**2 笔 ``Margin``**。真实回测里这类拒单占全部下单的 **61%**。
        #
        # 卖出**不**算释放名额或现金：挂单未成交前它并不确定会成交，按「更少下单」估是稳妥的
        # 一侧（宁可少买，不可超支）。
        pending = 0
        committed = 0.0
        for order in self.broker.orders:
            if not order.isbuy() or not order.alive():
                continue
            pending += 1
            committed += self._pending_spend(order)

        slots = self._slots(held, pending)
        if slots <= 0:
            return 0

        price = data.close[0]
        if price <= 0:
            return 0

        budget = self._budget(cash, committed, slots, held + pending)
        if budget <= 0:
            return 0

        # 按**最坏成交价**定量，而不是按当根收盘价——见 `_headroom`。
        worst = price * (1.0 + self._headroom(data))
        return self._affordable(comminfo, budget, worst)

    def _slots(self, held: int, pending: int) -> int:
        """本单还有几个名额可占——子类各按自己的口径回答。"""
        raise NotImplementedError

    def _budget(self, cash: float, committed: float, slots: int, already: int) -> float:
        """本单该投多少钱——子类各按自己的口径回答。"""
        raise NotImplementedError

    def _pending_spend(self, order) -> float:
        """一笔挂单中的买单预计占用多少现金（按它自己的**最坏成交价**估）。

        与 :meth:`_affordable` 同一口径（含涨跌幅留余地），否则预留额会小于实际成交额，
        超支只是从「这笔」挪到「下一笔」。未成交部分才计入——部分成交的单已经花掉的钱在
        ``cash`` 里扣过了。
        """
        data = order.data
        if data is None or len(data) == 0:
            return 0.0
        remaining = abs(order.size) - abs(order.executed.size)
        if remaining <= 0:
            return 0.0
        price = float(data.close[0])
        if price <= 0:
            return 0.0
        return remaining * price * (1.0 + self._headroom(data))

    def _headroom(self, data) -> float:
        """成交价相对**定量价**的预留比例（见 :meth:`_affordable` 的说明）。

        ``headroom`` 显式给了就用它；否则按规则表查该标的在成交日的**涨跌幅限制**——因为
        订单在本根的收盘定量、在**下一根**成交，而下一根的价格受「前收盘 × (1 + 限幅)」
        约束，所以限幅是那个缺口的**可靠上界**。

        用 :meth:`~mbt.rules.RuleTable.limit_for` 而非 ``price_limit``：前者已按需叠加 ST
        覆盖，故对 ST 标的给出的仍是上界（且更紧）。``rules`` 未给则**不留余地**——这是刻意
        的向后兼容，调用方要么给规则表、要么自己给 ``headroom``。
        """
        if self.p.headroom is not None:
            return self.p.headroom
        if self.p.rules is None:
            return 0.0
        return self.p.rules.limit_for(data._mbt_symbol, data.datetime.date(0))

    @staticmethod
    def _affordable(comminfo, budget, price) -> int:
        """在 ``budget`` 之内买得起的最大股数——**含交易费用**。

        为什么不能只算 ``budget / price``：撮合层要求「成交金额 + 费用 ≤ 现金」，而费用是
        **制度必收**的（过户费与佣金无关，``commission_mode="net"`` 时还有经手费与证管费）。
        于是 `budget / price` 几乎正好用光预算时，加上费用就超——**每一笔买单都以 ``Margin``
        被拒**。实测单标的（``max_positions`` 省略或为 1）时零成交，而退出码仍是 0：不报错，
        只是回测空转。**这是本项目最防的那类失败。**

        做法是**二分**：``size * price + fee(size)`` 对 ``size`` 严格单调递增（``size*price``
        递增，``fee`` 单调不减），故在 ``[0, int(budget/price)]`` 上二分即可求出满足约束的
        最大整数，精确且无需知道费率。

        刻意**不**乘一个 0.999 之类的安全系数：那既盖不住 ``commission=0.3%`` 这类配置，也会
        在费率很小时白白少买，而两者都不会报错。

        为什么要用 ``comminfo`` 而不用自己的费率常量：backtrader 的 ``Sizer.getsizing`` 会先调
        ``broker.getcommissioninfo(data)``（本项目的 ``AStockBroker`` 在那里把**成交日与板块**
        注入费用对象），**再**调 ``_getsizing``。所以这里的 ``comminfo`` 已能算出当日当板块的
        真实费用——照抄一份费率表出来只会与之漂移（Shotgun Surgery）。

        .. note::

            传入的 ``price`` 是**最坏成交价**而非当根收盘价（见 :meth:`_headroom`）。代价是
            系统性少买约「限幅」那么多（主板约 9%、北交所约 23%），**这是刻意承担的**：那份
            代价是确定的、可解释的，而「整笔被拒」是不确定且静默的。

            另两条路各自的问题：开 ``coc``／``coo`` 能让成交价等于定量价，但会改动成交时点
            语义、让回测偏乐观；让撮合层在成交时**缩减**订单则更省资金，但要改 backtrader 的
            订单模型。两条都留待将来，不在本口径内。
        """
        low, high = 0, int(budget / price)
        while low < high:
            mid = (low + high + 1) // 2
            if mid * price + comminfo.getcommission(mid, price) <= budget:
                low = mid
            else:
                high = mid - 1
        return low


class EqualWeightSizer(AStockSizer):
    """等权 sizer：把可用资金摊给剩余的持仓名额（见模块说明）。

    每个持仓名额的目标**仓位**相同（``1/M``），故依次买入时自然把可用资金摊开。

    参数:
        max_positions: 最大持仓**标的数**。``None`` 表示不限——此时每个名额的目标仓位无从
            谈起，故本次买入用尽可用资金。那会让后续买入因资金不足被拒，而那些拒单会记进
            ``BacktestResult.rejected``，不会悄无声息地发生。
    """

    params = (("max_positions", None), ("equal_weight", True))

    def _slots(self, held: int, pending: int) -> int:
        if self.p.max_positions is None:
            return 1
        return self.p.max_positions - held - pending

    def _budget(self, cash: float, committed: float, slots: int, already: int) -> float:
        """本单该投多少钱——**等权**，见模块文档的「口径」一节。

        默认口径（``equal_weight=True``）按**组合总值 ÷ 名额总数**算：每个持仓名额分到等额
        一份，本单可花的不超过「本名额的那一份」，且不超过「可用现金 − 已挂单占用」。

        另一种口径（``equal_weight=False``）是 `可用资金 ÷ 剩余名额`——它**只在同一 tick
        一次买满时**才等于等权，分次建仓时会把全部现金压给最后一个名额（实测单只占到组合的
        **36.8%**，而等权应为 20%）。保留它是为了让「改口径」这件事可被对照。
        """
        free = max(cash - committed, 0.0)
        if not self.p.equal_weight:
            return free / slots

        total_value = float(self.broker.getvalue())
        unit = total_value / self.p.max_positions if self.p.max_positions else total_value
        # 「已占用的名额」= 已成交持仓 + 挂单中的买单（口径与 `slots` 同一处）。
        return max(min(unit, free), 0.0)


class FixedAmountSizer(AStockSizer):
    """**每笔固定金额**的 sizer：每个持仓名额投入 ``amount`` 元，而不是「组合总值的一份」。

    与 :class:`EqualWeightSizer` 只差 ``_budget`` 与 ``_slots`` 两处裁决（见
    :class:`AStockSizer`），外加一个「必须给正金额」的构造守卫：那三段踩出来的账（挂单占用、
    最坏成交价、含费定量）完全共用。

    什么时候要它：策略的仓位大小**不随净值漂移**时。等权口径按「组合总值 ÷ 名额」算，跑赢之后
    一笔就是 20 万往上；而需求写的是「每笔 20 万」，那是**固定金额**。

    **资金不够时少买，不跳过**：``min(amount, 可用现金 − 挂单占用)``。空着的名额比买不足更浪费。

    参数:
        amount: 每个名额的目标**金额**（元）。**必须为正**——负值是「卖空」或「不买」，
            两种意图都与本类无关，故报错而不是猜。
        max_positions: 最大持仓**标的数**。``None`` 表示不限名额，此时只受 ``amount`` 与
            可用现金约束。
    """

    params = (("amount", None), ("max_positions", None))

    def __init__(self):
        super().__init__()
        if self.p.amount is None or self.p.amount <= 0:
            raise ValueError(
                f"FixedAmountSizer 需要一个正的 amount（每笔的目标金额），收到 {self.p.amount!r}。"
                "不设默认值是刻意的：「每笔多少钱」是策略的属性，藏进默认值里就没人看得见它。"
            )

    def _slots(self, held: int, pending: int) -> int:
        if self.p.max_positions is None:
            return 1
        return self.p.max_positions - held - pending

    def _budget(self, cash: float, committed: float, slots: int, already: int) -> float:
        """本单该投多少钱：``min(amount, 可用现金)``——**与已有几笔无关**。

        这是本类与等权口径的**唯一**实质差别：等权要按「已占用名额」摊，而固定金额每一笔都
        是同一个数，故 ``slots`` 与 ``already`` 在这里不参与（名额的上限仍由 :meth:`_slots`
        管，那一步在 ``_getsizing`` 里已经判过）。
        """
        free = max(cash - committed, 0.0)
        return max(min(float(self.p.amount), free), 0.0)
