"""``python -m mbt`` 的入口。

存在的理由是**安装方式**：本仓库以 editable 装进 ``.venv``，而那条路径下
``[project.scripts]`` 声明的 ``mbt.exe`` 不一定被生成（本机实测就没有）。于是
``python -m mbt`` 是那个**总能工作**的调用方式，定时脚本靠它。

真正的入口是 :func:`mbt.cli._entry_point`——它只管收退出码，业务逻辑一概不在那里，
与 ``mbt`` 那条命令走的是同一个函数。
"""

from mbt.cli import _entry_point

if __name__ == "__main__":
    _entry_point()
