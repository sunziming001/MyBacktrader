"""``b1_daily.bat`` 的四条契约：**纯 ASCII**、**路径只从环境变量取**、**评估日不手填**、
**一致预期目录可以缺**。

这不是「给字符串写测试」的洁癖——这几条都是**静默失效**的那种，改坏了不会有人告诉你：

- 一个中文字进了 .bat：cmd.exe 按 OEM 代码页（本机 936）读它，轻则显示乱码，重则
  **解析错自己的行**。那时脚本不是报错，而是少跑一截。
- 一个写死的默认路径：会让「路径指错」看起来像「数据不在」，而「缺失」被读成「正常」
  ——这条教训的来历见 README 的「真实数据冒烟测试」。
- 一个写死的评估日：数据没更新时它会**照常成功**，选出一个陈旧日子的名单。
- 一致预期目录从「可缺」变成「必填」：那会让没下这族数据的人**整个日更任务停摆**，
  而他本来只是少了「按预测口径选股」这一项好处。

四者在评审时都看不出问题，故用测试钉住。手改 .bat 时若这条红了，先想清楚为什么。
"""

from pathlib import Path

BAT = Path(__file__).resolve().parents[1] / "b1_daily.bat"


def _text() -> str:
    return BAT.read_text(encoding="ascii")


def test_the_daily_script_exists_at_the_repository_root():
    assert BAT.is_file(), "README 让人双击它；挪走就得同时改 README"


def test_the_daily_script_is_pure_ascii():
    """cmd.exe 按 OEM 代码页读 .bat，故文件本身不能出现非 ASCII。"""
    raw = BAT.read_bytes()
    offenders = sorted({byte for byte in raw if byte > 127})
    assert not offenders, f"b1_daily.bat 必须是纯 ASCII，但有这些字节：{offenders}"


def test_the_daily_script_never_defaults_a_data_path():
    """路径只从环境变量取；没设或指错就报错退出，不猜。"""
    text = _text()
    assert 'set "MBT_TDX_ROOT=' not in text, "不该给根路径兜底默认值"
    assert 'set "MBT_TDX_GBBQ=' not in text, "不该给权息路径兜底默认值"
    assert 'set "MBT_TDX_GPONE=' not in text, "不该给一致预期路径兜底默认值"
    assert "%MBT_TDX_ROOT%" in text
    assert "%MBT_TDX_GBBQ%" in text
    # cw 由根路径推出，不另要一个环境变量——多一个变量就多一处能指错的地方。
    assert '--cw-root "%MBT_TDX_ROOT%\\cw"' in text


def test_the_daily_script_does_not_hand_type_an_evaluation_day():
    """评估日由 mbt screen 取「最近一个齐全的交易日」；写死会让它悄悄选错日子。"""
    assert "--as-of" not in _text()


def test_the_daily_script_screens_with_b1_over_the_whole_pool():
    text = _text()
    assert "--screen b1" in text
    assert 'set "TOP_N=all"' in text, "默认全池，不截断"
    # python -m mbt 而非 mbt.exe：可编辑安装不保证落出 mbt.exe。
    assert "-m mbt screen" in text


def test_forward_root_is_optional_and_never_blocks_the_run():
    """一致预期目录没设、或设了但指错，都只能 **NOTE**，不能 ``exit /b``。

    这条是「可选」这个承诺**唯一**能被执行的地方：它一旦变成必填，最先受影响的是
    没下这族数据的人——他的日更任务会整个停摆，而他本来只是少一项好处。
    """
    text = _text()
    assert "MBT_TDX_GPONE" in text, "环境变量名要对得上 mbt.data.gpone 与 README"
    assert 'set "FORWARD="' in text, "默认空：没给就不加参数，而不是加个假路径"

    # 从分支开头切到命令行：这段里不许出现 exit —— 否则「可选」就成了「必填」。
    start = text.index("if defined MBT_TDX_GPONE")
    forward_block = text[start : text.index("-m mbt screen")]
    assert "exit /b" not in forward_block, "一致预期缺失不该终止日更任务"
    assert "ERROR" not in forward_block, "缺它只是降级，不是错误"

    # 参数得真的传下去，否则「可选」变成「写了但没用」——比报错更难发现。
    command = text[text.index("-m mbt screen") : text.index("2>&1")]
    assert "--forward-root" in forward_block
    assert "%FORWARD%" in command, "变量名要对得上，且必须就在这条 screen 命令里"
