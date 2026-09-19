"""仓库根目录那两份日更脚本的契约：**纯 ASCII**、**路径只从环境变量取**、**评估日不手填**、
**各自那条规则要的那份数据才当硬条件**。

这不是「给字符串写测试」的洁癖——这几条都是**静默失效**的那种，改坏了不会有人告诉你：

- 一个中文字进了 .bat：cmd.exe 按 OEM 代码页（本机 936）读它，轻则显示乱码，重则
  **解析错自己的行**。那时脚本不是报错，而是少跑一截。
- 一个写死的默认路径：会让「路径指错」看起来像「数据不在」，而「缺失」被读成「正常」
  ——这条教训的来历见 README 的「真实数据冒烟测试」。
- 一个写死的评估日：数据没更新时它会**照常成功**，选出一个陈旧日子的名单。
- 把**用不着**的数据也当成硬条件（砖型不要财务数据，却要求 vipdoc 里的 cw 目录）：那会让没下这族
  数据的人**整个日更任务停摆**，而他本来只是没下一样自己不需要的东西。
- 两条规则**共用**一个自选股文件名：后跑的那条会静默覆盖先跑的那条，而通达信那边看不出
  任何异样（名字一样，内容换了）。

五者在评审时都看不出问题，故用测试钉住。手改 .bat 时若这条红了，先想清楚为什么。
"""

from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]

#: 两份脚本与它们各自那条规则：``(脚本, 规则名, 那条规则需要的可选数据)``。
SCRIPTS = (
    ("b1_daily.bat", "b1", "cw"),
    ("brick_daily.bat", "brick", None),
)

IDS = [name for name, _, _ in SCRIPTS]


def _text(name: str) -> str:
    """按 **ASCII** 读——非 ASCII 会被解码挡下来，这正是第一条契约要的效果。"""
    return (ROOT / name).read_text(encoding="ascii")


@pytest.mark.parametrize("name,rule,needs", SCRIPTS, ids=IDS)
def test_the_daily_script_exists_at_the_repository_root(name, rule, needs):
    assert (ROOT / name).is_file(), "README 让人双击它；挪走就得同时改 README"


@pytest.mark.parametrize("name,rule,needs", SCRIPTS, ids=IDS)
def test_the_daily_script_is_pure_ascii(name, rule, needs):
    """cmd.exe 按 OEM 代码页读 .bat，故文件本身不能出现非 ASCII。"""
    raw = (ROOT / name).read_bytes()
    offenders = sorted({byte for byte in raw if byte > 127})
    assert not offenders, f"{name} 必须是纯 ASCII，但有这些字节：{offenders}"


@pytest.mark.parametrize("name,rule,needs", SCRIPTS, ids=IDS)
def test_the_daily_script_never_defaults_a_data_path(name, rule, needs):
    """路径只从环境变量取；没设或指错就报错退出，不猜。"""
    text = _text(name)
    assert 'set "MBT_TDX_ROOT=' not in text, "不该给根路径兜底默认值"
    assert 'set "MBT_TDX_GBBQ=' not in text, "不该给权息路径兜底默认值"
    assert 'set "MBT_TDX_GPONE=' not in text, "不该给一致预期路径兜底默认值"
    assert "%MBT_TDX_ROOT%" in text
    assert "%MBT_TDX_GBBQ%" in text


@pytest.mark.parametrize("name,rule,needs", SCRIPTS, ids=IDS)
def test_the_daily_script_does_not_hand_type_an_evaluation_day(name, rule, needs):
    """评估日由 mbt screen 取「最近一个齐全的交易日」；写死会让它悄悄选错日子。"""
    assert "--as-of" not in _text(name)


@pytest.mark.parametrize("name,rule,needs", SCRIPTS, ids=IDS)
def test_the_daily_script_names_its_own_rule_over_the_whole_pool(name, rule, needs):
    text = _text(name)
    assert f"--screen {rule}" in text, "脚本要指名它那条规则"
    assert 'set "TOP_N=all"' in text, "默认全池，不截断"
    # python -m mbt 而非 mbt.exe：可编辑安装不保证落出 mbt.exe。
    assert "-m mbt screen" in text


def _screen_command(text: str) -> str:
    """脚本里那条 ``mbt screen`` 调用（从参数名到重定向）——契约管的是**这条命令**。

    刻意只看这一段而不是整个文件：注释里完全可以**解释**「为什么不要 cw」或「为什么不要
    compatible 的开关」，而那属于说明，不是调用。把整份文件当判据，就会把说明也当成违规。
    """
    return text[text.index("-m mbt screen") : text.index("2>&1")]


@pytest.mark.parametrize("name,rule,needs", SCRIPTS, ids=IDS)
def test_a_script_only_demands_the_data_its_own_rule_needs(name, rule, needs):
    """**用不着的数据不是硬条件。**

    B1 的最后两条过滤读 PE，故它要求 ``vipdoc\\cw``；而砖型一个字都不读财报——把 ``cw``
    也当成它的硬条件，会让没下那族数据的人整条日更任务停摆，而他本来只是没下一样自己不需要
    的东西。反过来也一样：一条读财报的规则不能少了那道检查。
    """
    text = _text(name)
    command = _screen_command(text)
    if needs == "cw":
        assert '--cw-root "%MBT_TDX_ROOT%\\cw"' in command, "它要读财报，就得把那个目录传下去"
        assert 'if not exist "%MBT_TDX_ROOT%\\cw"' in text, "缺了它要当场停下"
    else:
        assert "--cw-root" not in command, "这条规则不读财报，就不该要那个目录"
        assert (
            'if not exist "%MBT_TDX_ROOT%\\cw"' not in text
        ), "更不该把那个目录当成硬条件——缺了它会让整条日更任务停摆"


@pytest.mark.parametrize("name,rule,needs", SCRIPTS, ids=IDS)
def test_a_script_only_carries_the_forward_block_its_own_rule_uses(name, rule, needs):
    """前瞻（一致预期）那一套是 **B1 专属**：砖型不读 PE，故整块机关不该出现。

    这块机关的要害是「可选、只印 NOTE、绝不 ``exit /b``」（见下一条）。砖型用不着它，留着
    会让没下 ``MBT_TDX_GPONE`` 的人白看到一句与他无关的 NOTE。
    """
    text = _text(name)
    command = _screen_command(text)
    if rule == "b1":
        assert 'set "FORWARD=--forward-root ' in text, "它要用前瞻，就该把那个参数装起来"
        assert "%FORWARD%" in command, "而且真的传下去"
    else:
        assert 'set "FORWARD=' not in text, "这条规则用不着前瞻，不该有那段机关"
        assert "--forward-root" not in command
        assert "MBT_TDX_GPONE" not in text, "连那个环境变量都不该提——提了会让读者以为它有用"


def test_forward_root_is_optional_and_never_blocks_the_run():
    """一致预期目录没设、或设了但指错，都只能 **NOTE**，不能 ``exit /b``。

    这条是「可选」这个承诺**唯一**能被执行的地方：它一旦变成必填，最先受影响的是
    没下这族数据的人——他的日更任务会整个停摆，而他本来只是少一项好处。
    """
    text = _text("b1_daily.bat")
    assert "MBT_TDX_GPONE" in text, "环境变量名要对得上 mbt.data.gpone 与 README"
    assert 'set "FORWARD="' in text, "默认空：没给就不加参数，而不是加个假路径"

    # 从分支开头切到命令行：这段里不许出现 exit —— 否则「可选」就成了「必填」。
    start = text.index("if defined MBT_TDX_GPONE")
    forward_block = text[start : text.index("-m mbt screen")]
    assert "exit /b" not in forward_block, "一致预期缺失不该终止日更任务"
    assert "ERROR" not in forward_block, "缺它只是降级，不是错误"

    # 参数得真的传下去，否则「可选」变成「写了但没用」——比报错更难发现。
    command = _screen_command(text)
    assert "--forward-root" in forward_block
    assert "%FORWARD%" in command, "变量名要对得上，且必须就在这条 screen 命令里"


@pytest.mark.parametrize("name,rule,needs", SCRIPTS, ids=IDS)
def test_the_panel_window_covers_what_that_rules_declares(name, rule, needs):
    """面板窗口按**那条规则声明的深度**取，而不是所有人按最深的那条。

    B1 最深看 1000 根（PE 百分位窗口），故它取 1301；砖型自己声明只看 100 根（票据 #86），
    取 200 就有两倍余量。窗口写小了闸门会拒绝运行（那是好事），写大了白占内存。
    """
    from mbt.cli import PANEL_WARMUP_BARS
    from mbt.screen import brick_screen

    text = _text(name)
    line = next(row for row in text.splitlines() if row.startswith('set "PANEL_BARS='))
    value = int(line.split("=", 1)[1].rstrip('"'))

    if rule == "b1":
        assert value > PANEL_WARMUP_BARS, "B1 比最深的那条还浅，闸门会直接拒绝运行"
    else:
        declared = brick_screen().lookback_bars
        assert declared is not None, "这条规则该声明自己的深度"
        assert value >= 2 * declared, f"只给了 {value} 根，声明要 {declared} 根，留的余量不足"
        assert value < PANEL_WARMUP_BARS, "它本来就比最深的那条浅，取到那个数就没省下东西"


def test_the_two_scripts_write_differently_named_watchlist_files():
    """两条规则的自选股文件**名字必须不同**。

    通达信按**文件名**认自选股，而这份文件是「固定名、每天覆盖」的。两条规则共用一个名字时，
    后跑的那条会静默覆盖先跑的那条——用户在「每日选股」这个名字下看到的是砖型的名单，而
    通达信那边没有任何异样。故名字必须分得开。
    """
    from mbt.report import WATCHLIST_NAME, watchlist_name_for

    names = {watchlist_name_for(rule) for _, rule, _ in SCRIPTS}
    assert len(names) == len(SCRIPTS), f"两份自选股文件重名了：{names}"
    # B1 那一份**不动**：既有用户的自选股板块是按这个名字建的。
    assert watchlist_name_for("b1") == WATCHLIST_NAME
    assert watchlist_name_for("brick") != WATCHLIST_NAME


def test_an_unknown_rule_falls_back_to_the_default_name():
    """没登记过的规则用默认名——不报错，但共用同一个名字这件事要在 README 里看得见。"""
    from mbt.report import WATCHLIST_NAME, watchlist_name_for

    assert watchlist_name_for("momentum") == WATCHLIST_NAME
    assert watchlist_name_for("none") == WATCHLIST_NAME
