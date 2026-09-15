"""品种判定（票据 #6，AC 1）。

**品种**（这是什么：股票 / 指数 / 可转债 / 基金）与**板块**（适用哪套交易制度）不在同一层，
`CONTEXT.md` 中「品种」的 `_Avoid_` 明确列了「板块」，两者不可混用。故本模块落在数据层，
不依赖规则表也无从依赖——判一个代码「是不是股票」不需要知道任何制度参数。

判不出时归入**「其他」而不报错**，与 `mbt.rules.board_of` 的报错刻意不对称：板块判错会让
**规则取值失真**（错得无声），而品种归入「其他」只是**不被股票池收**，错在安全方向。
"""

from __future__ import annotations

import pytest

from mbt.data import instrument_type
from mbt.data.instrument import bare_code
from mbt.rules import RuleTableError, board_of

#: 每个品种各取一组**真实前缀**的样本。编码的是「这些代码是这种东西」这一事实，
#: 不是实现细节——故用真实代码而非编造的。
STOCKS = [
    "sh600000",  # 沪主板
    "sh603288",
    "sh688981",  # 科创板
    "sz000001",  # 深主板
    "sz002594",
    "sz300750",  # 创业板
    "sz301236",
    "bj920819",
    "bj430047",
]
INDEXES = ["sh000001", "sh000300", "sz399001", "sz399006"]
CONVERTIBLE_BONDS = ["sh110043", "sh113050", "sz123456", "sz127045", "sz128136"]
FUNDS = ["sh510300", "sh588000", "sh512880", "sz159915", "sz161725"]
OTHERS = ["sh900901", "sh204001", "sz200011", "sh120001", "sz101651"]


@pytest.mark.parametrize("symbol", STOCKS)
def test_stock_prefixes_are_stocks(symbol):
    assert instrument_type(symbol) == "股票"


@pytest.mark.parametrize("symbol", INDEXES)
def test_index_prefixes_are_indexes(symbol):
    """指数与股票共用 ``000`` / ``300`` 这类数字，靠**市场前缀**区分：

    ``sh000001`` 是上证指数，``sz000001`` 是平安银行——只按代码前缀判必错一个。
    """
    assert instrument_type(symbol) == "指数"


@pytest.mark.parametrize("symbol", CONVERTIBLE_BONDS)
def test_convertible_bond_prefixes_are_convertible_bonds(symbol):
    assert instrument_type(symbol) == "可转债"


@pytest.mark.parametrize("symbol", FUNDS)
def test_fund_prefixes_are_funds(symbol):
    assert instrument_type(symbol) == "基金"


@pytest.mark.parametrize("symbol", OTHERS)
def test_unrecognised_prefixes_fall_back_to_other_without_raising(symbol):
    """判不出归「其他」而**不报错**：真实目录里一定混着 B 股、逆回购、企业债，
    让扫描因一个陌生前缀整体失败，比少收几只更糟。"""
    assert instrument_type(symbol) == "其他"


def test_an_index_only_becomes_other_when_it_has_no_known_prefix():
    """「其他」不是兜底垃圾桶——它只接住真正判不出的代码。

    这条同时说明「其他」为何不能与「指数」合并：合并会让 ``sh999999`` 这种不存在的代码
    看起来像指数，而股票池的排除理由就变得不准确。
    """
    assert instrument_type("sh999999") == "其他"
    assert instrument_type("sh999999") != "指数"


def test_kechuangboard_etf_is_a_fund_while_kechuangboard_stock_is_a_stock():
    """``588xxx`` 是科创板 ETF（基金），``688xxx`` 是科创板股票——前缀差一位，品种不同。

    这是最容易写错的一对：只按「科创」二字归类会把 ETF 当成股票收进池子。
    """
    assert instrument_type("sh588000") == "基金"
    assert instrument_type("sh688981") == "股票"


def test_every_symbol_the_rule_table_accepts_is_a_stock_here():
    """两套前缀知识的一致性交叉检查。

    品种判定与板块判定是**两种**知识（`CONTEXT.md` 明确区分），故它们是两份代码、可能漂移。
    这条把重叠的部分钉住：规则表认得的标的，品种判定必须认它是股票——否则会出现
    「规则表能算它的费用，股票池却认为它不是股票」这种自相矛盾。
    """
    for symbol in STOCKS:
        assert board_of(symbol)  # 不抛错即说明规则层认它
        assert instrument_type(symbol) == "股票"


def test_token_that_is_too_short_falls_back_to_other():
    """残缺代码不该让整次扫描失败（它在文件名层面就可能是脏的）。"""
    assert instrument_type("sh") == "其他"
    assert instrument_type("") == "其他"


def test_an_index_is_accepted_by_instrument_type_but_rejected_by_board_of():
    """两个问题的答案本就不同：品种答「是什么」，板块答「适用哪套制度」。

    指数**有**品种而无板块——把这两件事合并成一个函数，就会被迫给指数编一个板块。
    """
    assert instrument_type("sh000001") == "指数"
    with pytest.raises(RuleTableError):
        board_of("sh000001")


# --- 裸代码：去掉市场那一段 ---------------------------------------------------


def test_bare_code_strips_the_market_prefix_off_real_codes():
    """通达信自己那几处文件格式（``gpcw`` 的按期财报、证券主表、导出的自选股）只认裸代码。

    取**真实代码**而不是编造的：编码的是「这些符号指这些股票」这一事实，不是实现细节。
    """
    assert bare_code("sh600000") == "600000"
    assert bare_code("sz000001") == "000001"
    assert bare_code("sz300888") == "300888"
    assert bare_code("bj920819") == "920819"


def test_bare_code_leaves_a_bare_code_alone():
    """已经是裸代码的原样返回——调用方不必先问「它带前缀吗」。"""
    assert bare_code("600000") == "600000"
    assert bare_code("300888") == "300888"


def test_bare_code_does_not_validate_the_prefix_against_the_code():
    """**不校验前缀与代码是否匹配**——那是 :func:`instrument_type` 的事。

    ``sh000001`` 是上证指数、``sz000001`` 是平安银行，两者都是合法的（市场, 代码）组合，
    去掉市场段之后**都是** ``000001``。本函数只回答「去掉市场那一段之后是什么」。
    """
    assert bare_code("sh000001") == bare_code("sz000001") == "000001"


def test_the_three_bare_code_entry_points_agree():
    """三处入口必须给出同一个答案——它们是同一件事，此前是三份拷贝。

    ``mbt.data.fundamental._core_code`` 与 ``mbt.data.master._bare`` 都是这段代码的副本
    （逐字相同），本函数是它们收敛出来的那一份。谁再各自演化，这条会先响。
    """
    from mbt.data.fundamental import _core_code
    from mbt.data.master import _bare

    samples = ["sh600000", "sz300888", "bj920819", "600000", "", "sh"]
    for symbol in samples:
        assert _core_code(symbol) == bare_code(symbol), f"{symbol!r} 在 fundamental 那边不一致"
        assert _bare(symbol) == bare_code(symbol), f"{symbol!r} 在 master 那边不一致"
