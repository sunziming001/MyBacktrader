"""真实数据的冒烟测试：用本机通达信文件验证格式假设。

标记为 ``realmdata``，且在本机数据缺席时**自动跳过**而非失败——这样换机器后
测试套件仍然全绿，而格式假设（32 字节定长记录）依然被真实文件验证过。

路径可用环境变量 ``MBT_TDX_ROOT`` 覆盖，默认指向本机安装。
"""

import os
from pathlib import Path

import backtrader as bt
import pytest

from mbt.backtest import run_backtest
from mbt.data import TdxDataSource

DEFAULT_ROOT = Path(r"D:\Tools\tdx\vipdoc")

pytestmark = pytest.mark.realmdata


@pytest.fixture
def real_root():
    root = Path(os.environ.get("MBT_TDX_ROOT", DEFAULT_ROOT))
    if not (root / "sh" / "lday").is_dir():
        pytest.skip(f"本机通达信数据不可用：{root}")
    return root


class BuyAndHold(bt.Strategy):
    def next(self):
        if not self.position:
            self.buy()


def test_real_day_file_covers_expected_history(real_root):
    df = TdxDataSource(real_root).daily("sh600000")

    assert len(df) > 2000, "应覆盖多年日线"
    assert df.index.is_monotonic_increasing
    assert (df["close"] > 0).all()


def test_real_day_file_length_is_record_multiple(real_root):
    """真实文件的长度必须能被记录长度整除——这是格式假设的直接验证。"""
    path = real_root / "sh" / "lday" / "sh600000.day"

    assert path.stat().st_size % 32 == 0


def test_backtest_runs_on_real_data(real_root):
    """端到端在真实数据上跑通（结果本身尚不可用于策略判断，见票据 02/03）。"""
    prices = TdxDataSource(real_root).daily("sh600000")

    result = run_backtest(prices, strategy=BuyAndHold, cash=100_000.0)

    assert len(result.equity_curve) == len(prices)
    assert len(result.trades) == 1
    assert result.final_value > 0
