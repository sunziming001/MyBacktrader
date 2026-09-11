"""通达信权息事件文件 ``gbbq`` 的解析与解密（ADR-0003、ADR-0008）。

## 文件结构

``4 字节小端计数 + N × 29 字节定长记录``，实测 ``(文件长度 − 4) / 29`` 恰等于计数。
每条记录**前 24 字节加密、后 5 字节明文**；明文解开后布局为：

========  ====  ============================================
偏移      宽度   字段
========  ====  ============================================
0         1     ``market``（0=深、1=沪、2=北）
1         7     ``code``（6 位 ASCII 数字 + NUL）
8         4     除权除息日 ``YYYYMMDD``（u32 小端）
12        1     事件类别（1..15）
13        4×4   f1..f4（四个 float32 小端）
========  ====  ============================================

## 加密

16 轮 Feistel 型分组密码，**密钥表 4176 字节内嵌于通达信程序、不在文件里**。每 8 字节
为一个分组，一条记录的 24 字节密文即 3 个分组。**分组之间、记录之间都没有链接模式**
——同一只股票的连续记录，密文前 8 字节逐字节相同，正是分组独立的直接证据。因此整表
可以对记录维度向量化解密，也可以从真实文件里任意截取片段作为 fixture。

密钥表的出处，以及 ADR-0008「不拷贝第三方代码」在这里怎么落实：

- 密钥表是**通达信文件格式的一部分**，不是某个第三方库的智慧财产（ADR-0008）。
  ``pytdx`` 与 ``mootdx2`` 各自转抄了它，两者逐字节相同——已核对。它无法从文件本身
  推导，只能作为格式常量引入。
- 但解密**实现**是自研的：算法以上面的格式描述为准，用 numpy 对记录维度向量化，
  没有移植任何第三方实现的控制流。``pytdx`` 仅作开发期 oracle——实测本模块与它对
  全部 199,800 条记录逐字段完全一致。
- 密钥表的正确性另有**不依赖 oracle 的自证**：解密后逐条校验 market / code / date /
  category 的结构约束（见 :func:`_validate`）。密钥哪怕错一位，几乎所有记录都会违反
  约束——实测 199,800 条零违规。

## 只有「除权除息」（类别 1）参与价格复权

类别 2–10 的四个浮点数是**股本数量（单位：万股）**，不是价格数据；类别 11/12/15 是
份额折算比率。本项目范围是个股与主要指数、不含 ETF，故只有类别 1 参与价格复权
（ADR-0003）。这个「只做一类」是刻意的，不是漏实现——文件里有 15 类，另 14 类与价格
无关。类别 5「股本变化」另有用途（估值因子需要每个评估日的总股本），届时再单独取用。
"""

from __future__ import annotations

import datetime as dt
import struct
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .adjust import AdjustmentEvent
from .errors import MarketDataError

#: 单条记录的字节数。
RECORD_SIZE = 29
#: 计数头的字节数。
HEADER_SIZE = 4
#: 每条记录中被加密的前缀字节数（其余 5 字节明文）。
ENCRYPTED_SIZE = 24

#: 「除权除息」事件类别——唯一参与价格复权的类别。
CATEGORY_DIVIDEND = 1

#: 真实文件里出现过的最大类别编号。超出即视为格式已变：报错，不静默当成无关事件。
MAX_CATEGORY = 15

#: 市场前缀（与 ``.day`` 文件名一致）→ gbbq 里的 ``market`` 编号。
_MARKET_BY_PREFIX = {"sz": 0, "sh": 1, "bj": 2}
_PREFIX_BY_MARKET = {v: k for k, v in _MARKET_BY_PREFIX.items()}


class GbbqError(MarketDataError):
    """权息事件文件不可信。"""


#: 16 轮分组密码的密钥表，4176 字节。出处见模块文档「加密」一节。
_KEY_TABLE_HEX = (
    "38a7c21de06a17e2d139a2409cba46af42c6ff0574eadabb89b4f844ac89d7f2"
    "987fb6bce4f76b750504586779c86dc62b06968cfb86068bbfd6e8e187496b36"
    "c71802795325727213cc040b90240cdcdb031ad52e04855c7e8ebd02262dbd06"
    "1b5034991ba22404f28835c889ead5fb1224bbb53b29ca14a604cea9a85802b9"
    "aae397a3a62257bbada0225feb058611c3edb13f39c236d14a43c8644db06e3a"
    "7c516df78ec6dff38ea41e749db222054d073f967f97f963b9c42b9875f6d684"
    "56dc15d3528b60f3d60ea9ad0707e9028658c2329c90bcc919bfb0547af8cca8"
    "27638229eefb9811bf352962919395fcf4f008e4b23ab45eb3b02e3e20c1d743"
    "597dc6295f69747fb277e10efa85a1c9777383b3cb1c60dbe95369fcb3185915"
    "0f978a7ac883f549dc1b3e86c1954546e216677f1235a0bb27fbccf8307e4fc8"
    "6dab18b20d01cc7920807bfa37aa149e85e825e9d42d354e8fd3deb0068d1515"
    "5265e8390328090267993d13baf3685c4c89b0e36bae165c8825f8330319025b"
    "297b2a412d7549489bb3b6b3bfaadf8c95fe0f13b87b02bb52e11c34c39b8759"
    "e246cc22774bd7c42c31aa847c445188151accae409d1f4497299845607447a1"
    "0da573f053ff01f9f49af13607d02da0792d812325ad4b9cc8bc12554dd4bb95"
    "b1b9be7da6e6a053ba838cdd7ee94bedba2842d8ff986935ca4e9c9d57d6cfa0"
    "895ca2e754d2af4cfb54c4b44fc3baf8a2586919790ea80e3dc804fd2632c8e1"
    "028ba71cc39125e5d849dbdf195f16f5a78b182304d4bffb44c4617c796ec890"
    "15b5eb5087ca7a69472fafa8b5a28a84c44179e8de0cacd0d56f34c6cba776f9"
    "00244205267e7b1486597bdb1c62d5b73ef71744274bd2c66fffc84955ad6552"
    "2d43c2339b63ab3d545428e20265039a034b8f641a9252de32d62bf0bebe1d54"
    "b17c70419b9055da715521b9b66890195fbcaab4550ee6814ca3bebc64d75900"
    "59bd0f6a571aa6a0d51a0a80d30906735a51e2dd2966aca08629212b7a6d9e3a"
    "68d0a3dca72b85a04cd4f0c5c443e4cf0c198130b6f6be71f5ac25aacf429006"
    "641b4529fd3aa3b60b9d299ffa31b86dd8ec43f5927e3522e0c3d309066171da"
    "e8360a19f62381cb89e0676efeb1e64772635c2518e0b46585efb51b26239089"
    "cceee301779563dfc4acbfe637149915498a960291aa1d9821575e8796c7b587"
    "083f58065258178faba84ea17a60b1695e9cbee2d0c51259df31ebd2195496e2"
    "10118e68b41a2dd32fab12f7fef3a7f761fcf77ccbfc878c6a1040297b30d60d"
    "134c71cd5eab36a2f14c05ed5388e5ff8e71795db5afd3676dc4446babc1a7aa"
    "38d8701e08e6d2367b881196dbd268d9ffd8502b3aa9cc451acacdd205c6fca0"
    "350cee982b5cb2396a27128f97eccb7bb6c027f6a74875098298ca3a5de3960c"
    "a5d2b36ca4d11fae9967b03dd69a7a3e008bfd4532f79f287c9403db64aa4480"
    "d227afb373875731eb08d9ba734d2c7703bff50f473c22da3fb9f19a1b228316"
    "eef418fc08e83b301c0450aa4ce32853abdef85f32d9e1787bf1c5a8ca85b69f"
    "891f40b82c88d7c1663445d646fd7bf372a3325523cfb5b079aba0f1005cdbee"
    "3f51aaaec0898e47a5304e4bddd6aed86d401c4e8efb0c608d541e2f17b73aed"
    "dedc81f57285b7a639316f47508443c511f36a268eba7f819831fd136b83c911"
    "614864fae3f5392c1211c16d4d0313a6c2e0dff5328e5b35a77f08f785270d71"
    "9db8ce9c1eba773af6a1a7269429c0201065756eefaa320c66913a4e0e74e28a"
    "feb6f817c7a7e4d835672ef083a89fa6281340a396dc498355e185abbd4ded88"
    "fa3669a977595a9cd0a0b13deb3116dc3e297b39015bd4ff5ce59edaf755d53f"
    "e33b5176838e40aee12ee83ef808b7b0242691ad824c2e2f377a34a105bd8c9a"
    "75525ccd5980cb92f8b1f8a5f22c9f4a59bfef76a3744fe1c97c7f91d90d1205"
    "b28ed0e0bb46d45c442f656d7a1c0286fb7e7db62a57b9db80cd02bfe79e3521"
    "fbbe2813829ff074f79255def27bf2f27df5a0140f994d25f4dc11177a776577"
    "ccbeef9088e8fdb24e8ef526fe535d65a974470bcbe9e8719595876cfd8694a7"
    "e5fc20001e0a0ae3851724d4d0738a111e1eef83e3d7e1bfcc98076d70373a8f"
    "3117554e60a8c8ab4f082d3776e62b58dd810fd16e9aa6553d8082999e2d169a"
    "df4ecb3b5ddaa85308c7ff54ddc611311ab6eba303084afbb445ecc07c0dc6cf"
    "cb1b7846888ff46a15622f1712e6416476589678db29b56aaede63416fbe9b37"
    "6cc9d0ec1bf679179efe790eb18228f20615c2be969ce08180d700db95874bc0"
    "0d91555b1f86226474ea1b8985d2ddf79ff1d9090664fa6d5972efce66a703d1"
    "99e8dfaed7635f605fab6ec522c83a946a3b0072f8db90e705dca2890f83aa03"
    "fe42141c8ae61c9edbd8d0ca97216caded0ae0a29eecc1ffd1b48a9aadab340b"
    "133fb5188d859e0df9fbac212edd7adebf9f7ebdbf84dff5fd1ebee11f0ff818"
    "9d73090229b75b267e4475044db1aa2f3adb463812d14135912906dfc9986992"
    "02f24812a971d2ae3b236d1ce26b8b75874a13a71f814d2965530a3a34ce6de6"
    "318d7e4edd256e7644823c47364cb9c49bf44f84431156c294537eb02e36daeb"
    "775fc164e2ca9fbe29d8063653d06f8219dabc8c5f4d45e721379e90a6d433a8"
    "644decbc905efe8e8bca177cffac96bb21cf3d24713bc2a1746885cf328e7f63"
    "39c5e78ea5e0cd3af59ab8fd43d44339088e45765fdfe917545912edd0e93d6f"
    "3f02148a0a479ad1e7fa4ea1410050ef609d4dc1ca879840e7b20f76c09d71ef"
    "d74693c12b9f11b8f905aceda7726bf5119b3e0a04217d06d746767badae9d95"
    "a6476805adf5387cc7a55acab2cb4818c1f262559836390880c528b106e4fb46"
    "113c38a14f1cfea181b7fcdb94b07afeb574f1bb92aaffb0fe1e318bc6bcf04f"
    "1afe91c57a9c73094a329051018b12c020ca3ccb1483d3c77c5a1279ee561a36"
    "c409e23edce8cef1c1a19e99da644fcf1ed62b7027863ecfbe751c399bf95363"
    "c16b58cc71d2074188bb147096f168ce1375fef4a0c885a2671849560d07941d"
    "7461890c32499d0d94734aab1ae90fe0bab64a34f9331db371c2b864d70bcb19"
    "f7bde0693e2496b1c428095f58ae8ac0839919644d443755a69ba1425084b818"
    "29b52191582388eb8f134a2409ec0f6d7daf3efcf7f39f343915c48403bb7e67"
    "395f2a2c6794f4a6b5023f4556790c2a9b257767c23bccf2713b4f832a8d8c53"
    "0d184954ca580ebe8b3a5374fc6f4728078ec1f553d3344b0805ffe91429401b"
    "57ad77ece8dada3555a77803564c7cb2ed3bb5616591df41b45dc9b79b138241"
    "15d7b36e1cc815b4f0f33f914ba1c8907891395a2155da6ae12cbac93869f6ae"
    "a82b8cb714c1358235a0784756c09aa77f74146485f1b748bc558c6aa4951ccb"
    "f352f9546115275643d02795e335aa39dc2338daef1f27653aabf7ccbb25db00"
    "363496d1f7c4ec4437427e171867c89c9a5b39085c3cf492f1163188fa12449e"
    "79271cc20b46accd1f39b89f9a56340a8586c2b1b19b31ce4757053ea7ae3f3e"
    "012dc5b9c1cbbaab0a2ad271e4ecf80a7185cca1ca6eef9d8722385d8081f71a"
    "6c317b8286bd7f109d89b6f7afe4410d4f97288034063e193a2160ed5418020f"
    "2fd5d53ba5870121381ba6993228e98d6f02356085bd64c4b0267e68d1e697b5"
    "326eb24feb064c4dc2978e6b3022c0b43d47937867ac2742dd5c3c27ed0a6ce4"
    "4a0d0fdf5263a6707609f02e58f605b2dfeec91fcb1d110ca18b1926b8102c81"
    "48ff98ef30360c01c54ad9ac057289c73fd64de017babab3d3e81b0c8cc8df6b"
    "fe7eba91fdf6a0cb5919b0012fd70ba0620f5fce74b8eb4289b5becac9efda9a"
    "bbc6661be065eed43aced9cc0ebb8550414501ba1b29116f34115503dd0cb599"
    "563a934d4d956dcec351e015543eff2fa3da59ec3d592d62fc6439d67bc88078"
    "1dd7fde80b5d8aed1a9d98cbc2ee784730ad8f64a5821223dab33eca4c857a80"
    "d59f4620d6eed1f933fa1fc59c8ef91e6651a54668dcb77fa85adee618d78c2b"
    "5deaa8ec6b8b48c1925ac1b16a5e3782224b6ab6f040168916a581f8d41b2026"
    "8635e5adc1016ec9b5d069c50b3108515d35fc74f513047af45710535ba4cc8b"
    "218282154b8c3d6bda9185cbd6cf0580d0f0cf0ddf7ab499c7f8d54c765630e9"
    "65b65860c1c0398a4254bc4a488ba1d95c32057a1cbb50515b7fc7752d6855e6"
    "837bc398fde6d5b8daa8310178f5608b1ad2fd513447faaf23aee2de15a70766"
    "69359a406155259823542a50c97da6ce74f8190c8e63e5492ff91705fd391555"
    "f4b091bf60b7b2402e7ad36886c0fc3888abb9038a04051a9f61aef2d3b8a429"
    "f85143cf84264a906e1327af7b52dbf900e8aec0b56f64035720597cf5e165a8"
    "47c3bdee722a85e2708dea9d98d42ad570a2e976a2dae67cb0f714d923b688c0"
    "b36f4212f4690c1581d6f70bb71bdf15e675631353b32043799034e3344880d6"
    "86bb45a285ddf823643bd568ab995334c6250a877317375639ba8c0e39244bcc"
    "aa98840c2f27e6e2ac86345d1e25aefd1eff3c27ad26184a1ae509615d835f2c"
    "dc41a7c607555bb50b71fe86e730a1bc27af5f24511add20f6329e3d646fdc43"
    "652a80cb95c4b6f0e1f3cf6cf2c29cea81880c2dd2da7482c6a51e98d3bc71ed"
    "e20b05dabb0efa350a2cd5c862e7b1af95146c837df1ce9f136bd868c9a5f587"
    "2ea58fd75cb2c69937315aa4d0e243dfc8bebd10c0d8226395461ee78ca861e4"
    "74026cb430f3061511e62a3a0d3b2fb93bb383401879fb3938b7ce4dbaf69eaa"
    "e18f321cb168dd5c2c376561733dc63456cdeabc776aa17d6af1f978af0fd9c2"
    "aad3d7a82da86ebc198396b5a33eb3b25c54ad77ce1de5d5aab30d367a327d5c"
    "a360668d84a0bd4f0fa90989b8ec148a2b2b748e75775a8eb251d026d6068c9a"
    "ca31d69417f014d7431c820c0083e675055c52ab0c388fa3357752e83e3bcb48"
    "81e325b1a94012764f16f1ce3dd7238944d73f247eb74666c1167a17b22a99f1"
    "ac3cc99dc5fe89bebf2c68bc2ca7f1c52f261eccd1af7daa7dc5944a4dc48797"
    "2d2b6a5e5ebf398218ab8cb9dc8083a1d180d265fe2ecc6af10284b236603724"
    "4e5e57ada5c5501a5ea45c31b6936057acebed653fbfeac708ca130093e5e679"
    "f63720cab46e399e834f158b15cde78c9093b085919bae21ef03d0a4b62ab4c6"
    "d307049254728eec2eb3476cce42067fe05b96f2488bfa8f83e24710a5b730f8"
    "68b0fd02746f4871d7f12edfa15261769947be0a2ff8f2699dad03fae684a7cf"
    "357d8f5fc5a69b216635bc58d589b5e09f11f0a88a1fc83c24b2b7f16c8adb3b"
    "397acad0ef15612272fdfc023dbd76359ee1c6d72cb259e103e0ff7a8703799f"
    "61abcc4998c241cf6e9baa529bd008b59e23f6c1398277165dd4e1b3ada00c58"
    "f8e267006a0b4bd26ce1c56b9dba3f4082c528b8c1607585eec4fa04ed6264b6"
    "2910674b9bd66c0e06626483caf02f2db8f60ad7d76a1c5814be1860802902cd"
    "f6b195a56d2e279c08e31fc5c2077f637fdb82c6c685aca6d24cf17fdb1dcf86"
    "205660c024e0c0420b4e005f8b7860feeaec6d31934970eb2a454f929b6c1728"
    "bb89fcc00784ccad1b85f285185c3d5a6054af039d9ee426d386aa0b7ca3329c"
    "c20f3ad43e1f5243a831e970fc0cb47cf5e3c76f11ed224c0c1b82cb72a49528"
    "1ad41be5c46ed7f1ecbf252cb89287a8d215793439c0be0dc8682df2d38e0109"
    "3c4894326989d5c05de82ce6a697594b9ac661b09edb81dcd3f947348400ca87"
    "be5d6d56f301023bffffffff00000000"
)

_KEYS = bytes.fromhex(_KEY_TABLE_HEX)

# 密钥表按用途切成四张 256×u32 的查表（各 1024 字节）与 16 个轮常量 + 2 个初始常量。
_TABLE_2 = np.frombuffer(_KEYS, dtype="<u4", count=256, offset=0x448)  # 取 num 的第 2 字节
_TABLE_3 = np.frombuffer(_KEYS, dtype="<u4", count=256, offset=0x048)  # 取 num 的第 3 字节
_TABLE_1 = np.frombuffer(_KEYS, dtype="<u4", count=256, offset=0x848)  # 取 num 的第 1 字节
_TABLE_0 = np.frombuffer(_KEYS, dtype="<u4", count=256, offset=0xC48)  # 取 num 的第 0 字节
#: 轮常量按偏移 0x40 → 0x04 **递减**使用，故此处反转。
_ROUND_KEYS = np.frombuffer(_KEYS, dtype="<u4", count=16, offset=0x04)[::-1].copy()
_CONST_FIRST = np.frombuffer(_KEYS, dtype="<u4", count=1, offset=0x00)[0]
_CONST_INIT = np.frombuffer(_KEYS, dtype="<u4", count=1, offset=0x44)[0]


@dataclass(frozen=True)
class GbbqRecord:
    """一条原始权息记录。

    四个浮点数的含义**随类别而变**（类别 1 是分红/配股/送转，类别 2–10 是股本数量），
    故这里保留 ``f1..f4`` 的中性命名，不冒充语义。类别 1 的语义解释见
    :meth:`GbbqDataSource.events`。
    """

    symbol: str
    date: dt.date
    category: int
    f1: float
    f2: float
    f3: float
    f4: float


def _decrypt_records(raw: bytes, count: int) -> np.ndarray:
    """把整表密文解密成 ``(count, RECORD_SIZE)`` 的明文字节矩阵。

    记录之间没有链接模式，故对**记录维度**整体向量化：16 轮迭代各自作用在长度为
    ``count`` 的数组上。分组解密本身是串行的 Feistel 结构，无法再并行。
    """
    records = np.frombuffer(
        raw, dtype=np.uint8, count=count * RECORD_SIZE, offset=HEADER_SIZE
    ).reshape(count, RECORD_SIZE)
    # 前 24 字节按 6 个 u32 取出。切片在 29 字节步长上不连续，故先拷成连续内存再 view。
    words = np.ascontiguousarray(records[:, :ENCRYPTED_SIZE]).view("<u4").reshape(count, 6)

    plaintext = np.empty_like(words)
    for block in range(3):
        num = _CONST_INIT ^ words[:, 2 * block]
        numold = words[:, 2 * block + 1]
        for round_key in _ROUND_KEYS:
            value = _TABLE_2[(num >> 16) & 0xFF]
            value = value + _TABLE_3[num >> 24]
            value = value ^ _TABLE_1[(num >> 8) & 0xFF]
            value = value + _TABLE_0[num & 0xFF]
            value = value ^ round_key
            num, numold = numold ^ value, num
        plaintext[:, 2 * block] = numold ^ _CONST_FIRST
        plaintext[:, 2 * block + 1] = num

    out = np.empty((count, RECORD_SIZE), dtype=np.uint8)
    out[:, :ENCRYPTED_SIZE] = plaintext.view(np.uint8).reshape(count, ENCRYPTED_SIZE)
    out[:, ENCRYPTED_SIZE:] = records[:, ENCRYPTED_SIZE:]
    return out


def _validate(plaintext: np.ndarray) -> None:
    """校验解密结果是否满足文件结构约束，首个违规即报错。

    这是**不依赖 oracle 的自证**：密钥表若用错，解出来的 market / code / date /
    category 会几乎全部越界。缺口纪律（ADR-0005）要求异常报错而非放过。
    """
    market = plaintext[:, 0]
    code = plaintext[:, 1:7]
    terminator = plaintext[:, 7]
    dates = np.ascontiguousarray(plaintext[:, 8:12]).view("<u4").reshape(-1)
    category = plaintext[:, 12]

    year, month, day = dates // 10_000, (dates // 100) % 100, dates % 100
    bad = (
        (market > 2)
        | ~((code >= 0x30) & (code <= 0x39)).all(axis=1)
        | (terminator != 0)
        | (year < 1990)
        | (year > 2100)
        | (month < 1)
        | (month > 12)
        | (day < 1)
        | (day > 31)
        | (category < 1)
        | (category > MAX_CATEGORY)
    )
    if bad.any():
        i = int(np.flatnonzero(bad)[0])
        raise GbbqError(
            "解密结果不满足 gbbq 结构约束，密钥表或文件格式与预期不符："
            f"记录 {i} 解出 market={int(market[i])} category={int(category[i])} "
            f"date={int(dates[i])}"
        )


class _Table:
    """已解密并校验过的整表，按标的定位记录。"""

    def __init__(self, plaintext: np.ndarray):
        self.plaintext = plaintext
        market = plaintext[:, 0].astype(np.uint64)
        code = np.zeros(len(plaintext), dtype=np.uint64)
        for column in range(1, 7):
            code = code * 10 + (plaintext[:, column] - 0x30)
        # 把 (market, 代码) 合成一个整数键，定位即一次向量化比较。
        self._keys = market * 1_000_000 + code

    @property
    def count(self) -> int:
        return len(self.plaintext)

    def positions(self, market: int, code: int) -> list[int]:
        return [int(i) for i in np.flatnonzero(self._keys == market * 1_000_000 + code)]


#: 整表解析结果按「路径 + mtime + 大小」在进程内缓存。全表 199,800 条的向量化解密
#: 约 0.3 秒，远贵于一次查表，故缓存是必要的；只保留最近几份，避免长期进程堆积。
_CACHE: dict[tuple, _Table] = {}
_CACHE_LIMIT = 4


def _load(path: Path) -> _Table:
    try:
        stat = path.stat()
    except OSError as exc:
        raise GbbqError(f"权息事件文件不可读：{path}（{exc}）") from exc

    key = (str(path.resolve()), stat.st_mtime_ns, stat.st_size)
    cached = _CACHE.get(key)
    if cached is not None:
        return cached

    raw = path.read_bytes()
    if len(raw) < HEADER_SIZE:
        raise GbbqError(f"权息事件文件长度 {len(raw)} 不足一个计数头：{path}")

    (count,) = struct.unpack_from("<I", raw, 0)
    expected = HEADER_SIZE + count * RECORD_SIZE
    if len(raw) != expected:
        raise GbbqError(
            f"权息事件文件长度与计数头不符：计数 {count} 条应为 {expected} 字节，"
            f"实际 {len(raw)} 字节（{path}）"
        )

    plaintext = _decrypt_records(raw, count)
    _validate(plaintext)

    if len(_CACHE) >= _CACHE_LIMIT:
        _CACHE.clear()
    _CACHE[key] = table = _Table(plaintext)
    return table


def _parse_symbol(symbol: str) -> tuple[int, int]:
    prefix, code = symbol[:2].lower(), symbol[2:]
    well_formed = (
        prefix in _MARKET_BY_PREFIX and len(code) == 6 and code.isascii() and code.isdigit()
    )
    if not well_formed:
        raise ValueError(f"标的代码应形如 'sh600000'（市场前缀 + 6 位数字），收到 {symbol!r}")
    return _MARKET_BY_PREFIX[prefix], int(code)


def _to_record(plaintext: np.ndarray, position: int) -> GbbqRecord:
    market, code, date, category, f1, f2, f3, f4 = struct.unpack(
        "<B7sIBffff", plaintext[position].tobytes()
    )
    digits = code.rstrip(b"\x00").decode("ascii")
    return GbbqRecord(
        symbol=f"{_PREFIX_BY_MARKET[int(market)]}{digits}",
        date=dt.date(date // 10_000, (date // 100) % 100, date % 100),
        category=int(category),
        f1=f1,
        f2=f2,
        f3=f3,
        f4=f4,
    )


class GbbqDataSource:
    """本地 ``gbbq`` 权息事件文件。

    文件位置由调用方注入，测试因此可指向仓库内 fixture，而不依赖本机通达信安装路径。
    注意该文件位于 ``<安装目录>/T0002/hq_cache/gbbq``，**不在** ``vipdoc`` 之内——
    与 :class:`mbt.data.TdxDataSource` 的根目录不是同一个，故这里直接收文件路径。
    """

    def __init__(self, path):
        self._path = Path(path)

    @property
    def total_records(self) -> int:
        """整表记录数（含所有类别与所有标的）。"""
        return _load(self._path).count

    def records(self, symbol: str) -> tuple[GbbqRecord, ...]:
        """标的的**全部类别**记录，按日期升序。

        只有类别 1 参与价格复权（见模块文档）；其余类别的取用交给将来的估值因子，
        此处原样给出，不替调用方解释语义。
        """
        market, code = _parse_symbol(symbol)
        table = _load(self._path)
        rows = tuple(_to_record(table.plaintext, i) for i in table.positions(market, code))
        return tuple(sorted(rows, key=lambda row: row.date))

    def events(self, symbol: str) -> tuple[AdjustmentEvent, ...]:
        """标的的**除权除息事件**（类别 1），按除权除息日升序。

        类别 2–15 一律不进来——它们的四个浮点数不是价格数据（ADR-0003）。
        """
        return tuple(
            AdjustmentEvent(
                symbol=row.symbol,
                ex_date=row.date,
                cash_per_10=row.f1,
                rights_price=row.f2,
                bonus_per_10=row.f3,
                rights_per_10=row.f4,
            )
            for row in self.records(symbol)
            if row.category == CATEGORY_DIVIDEND
        )
