# 通达信 (TDX / Tongdaxin) `gbbq` 除权除息 file format — fact report

**Scope:** format of `T0002/hq_cache/gbbq` and `gbbq.map`, the 复权 algorithm, and library support.
**Inspected machine:** Windows, TDX installed at `D:\Tools\tdx`.
**Files inspected (this machine, 2026-09-11):**

| file | size (bytes) | note |
|---|---:|---|
| `D:\Tools\tdx\T0002\hq_cache\gbbq` | 5,794,204 | encrypted event table |
| `D:\Tools\tdx\T0002\hq_cache\gbbq.map` | 96,465 | plaintext index |
| `D:\Tools\tdx\vipdoc\sh\lday\sh600000.day` | 69,856 (2,183 records at re-read) | daily bars — TDX was running and updated this file mid-session |

**Labels used below:** **[VERIFIED]** = I read the real bytes on this machine and confirmed it. **[SECONDARY]** = comes from external sources only. Where they disagree, the real files win.

---

## TL;DR

- **[VERIFIED]** `gbbq` = a 4-byte little-endian record count (**199,800**) followed by **199,800 fixed 29-byte records**. `(5,794,204 − 4) / 29 = 199,800` exactly.
- **[VERIFIED]** Each record's **first 24 bytes are encrypted; the last 5 bytes are plaintext**. Decrypted record layout is: `market u8` + `code` 7 bytes (6 ASCII digits + NUL) + `date u32 YYYYMMDD` + `category u8` + **four little-endian `float32`**. Total 29 bytes.
- **[VERIFIED]** The cipher is **not** a simple XOR (all 256 single-byte XOR keys tested → no readable ASCII). It is a 16-round DES-like block cipher with an embedded **4,176-byte key table**, matching the disassembly account in the secondary sources.
- **[VERIFIED]** `gbbq.map` is **plaintext ASCII**, **6,431 lines × 15 bytes** (`6-digit code` + `4 decimal digits` + 2 spaces + `0D 0D 0A`). The 4-digit field is a small batch/version number (range 3,421–9,749) — **it is NOT a file offset**, so the common "`code → file offset`" belief is wrong.
- **[VERIFIED]** Category `1` (除权除息) carries the price-relevant data. The four floats are: **f1 = 每10股分红 (cash)**, **f2 = 配股价**, **f3 = 每10股送转股**, **f4 = 每10股配股**.
- **[VERIFIED]** `.day` files store **raw / unadjusted (不复权) prices** (corr 0.994 between observed ex-date gaps and gaps implied by category-1 records).
- **Libraries:** `pytdx` can read and decrypt `gbbq` via `pytdx.reader.gbbq_reader.GbbqReader().get_df(path)` — **[VERIFIED] I ran it on the real file**. pytdx is **archived/unmaintained** (last release 2019, last push 2020) but works. Stock **`mootdx` cannot read `gbbq`**; its adjustment path uses online/Sina data. **`mootdx2` (2026)** ships a working offline `read_gbbq()`, **but** its advertised `adjust='qfq'/'hfq'` path uses the online TDX xdxr feed, not the local file.
- **Single most important fact:** 前复权 is anchored to the **latest** 除权 event, so **every historical 前复权 price must be recomputed whenever a new dividend/split occurs**; incremental appending of 前复权 bars corrupts the series.

---

## 1. `gbbq` byte layout

### 1.1 Solving the frame **[VERIFIED]**

```
file size            = 5,794,204
header               = 4 bytes  -> uint32 LE = 199,800   (0x00030C78)
(5,794,204 - 4) / 29 = 199,800.0                          (exact)
```

The only clean `(size − header) % record_size == 0` solution with a sensible header is **header = 4, record size = 29**. Divisibility alone is not proof, but the decoded content below confirms it.

### 1.2 Raw bytes of the first records **[VERIFIED]**

```
header          : 78 0c 03 00
record 0 (enc)  : 63 12 24 b0 f3 11 c9 a9 53 c0 bb 18 06 86 6a 39 dc bc 5c 55 14 c3 e1 d2 00 00 00 80 3f
record 1 (enc)  : 63 12 24 b0 f3 11 c9 a9 b1 e7 86 c0 97 f3 e8 1c a0 ae 10 1b b3 66 41 e1 45 23 90 97 45
```

Note the first 8 bytes repeat across consecutive records — those are all records for one stock (`000001`). The high entropy (≈7.7 bits/byte) and total absence of ASCII codes/dates is what proves encryption. **A byte-wise XOR is ruled out**: sweeping all 256 single-byte XOR keys produces no `600000` / `000001` ASCII sequence.

### 1.3 Decrypted record layout **[VERIFIED]**

Decrypting the first 24 bytes (the same 3×(8-byte) DES-like transform used by pytdx / mootdx2, with the 4,176-byte key table) and appending the last 5 clear bytes gives:

```
record 0 plaintext: 00 30 30 30 30 30 31 00 8d a7 2f 01 01 00 00 00 00 0a d7 63 40 00 00 00 00 00 00 80 3f
record 1 plaintext: 00 30 30 30 30 30 31 00 03 cf 2f 01 05 00 00 00 00 00 00 00 00 00 a0 25 45 23 90 97 45
sh600000 plaintext:01 36 30 30 30 30 30 00 c2 2f 31 01 01 00 00 c0 3f 00 00 00 00 00 00 00 00 00 00 00 00
```

Decoded:

| offset | size | type | field | record 0 value |
|---:|---:|---|---|---|
| 0 | 1 | u8 | **market** (0=SZ, 1=SH, 2=BJ/NEEQ) | 0 |
| 1–6 | 6 | ASCII | **code** (6 digits) | `000001` |
| 7 | 1 | u8 | NUL terminator | `00` |
| 8–11 | 4 | u32 LE | **date** `YYYYMMDD` | `0x012FA78D` = 1990-03-01 |
| 12 | 1 | u8 | **category** | 1 |
| 13–16 | 4 | float32 LE | **f1** | 0.0 |
| 17–20 | 4 | float32 LE | **f2** | 3.56 |
| 21–24 | 4 | float32 LE | **f3** | 0.0 |
| 25–28 | 4 | float32 LE | **f4** | 1.0 |

Total: 1 + 7 + 4 + 1 + 16 = **29 bytes** ✔

Byte 24 is the last byte of f3 and is stored **in clear** (the encrypted span is exactly bytes 0–23). This matches the secondary account of "前24字节加密，后5字节未加密". **[SECONDARY corroboration:** [CSDN: 通达信股本变迁gbbq权息文件解密](https://blog.csdn.net/yeyiqun/article/details/116247057); [Baidu Word doc](http://word.baidu.com/view/c8178b2515cc700abb68a98271fe910ef12dae04.html)**]**

### 1.4 Sanity check on the decoded table **[VERIFIED]**

The full file decodes to **199,800 records across 6,608 distinct `(market, code)`** symbols, dates `19900301`–`20260918`, and 6,608 contiguous same-symbol runs (mean ≈30 records/symbol — exactly what a 30-year 权息 history looks like). Selected `sh600000` 浦发银行 records:

| date | cat | f1 | f2 | f3 | f4 | interpretation |
|---|---:|---:|---:|---:|---:|---|
| 2000-07-06 | 1 | 1.50 | 0 | 0 | 0 | 10派1.50元 |
| 2002-08-22 | 1 | 2.00 | 0 | 5.00 | 0 | 10派2元送5股 |
| 2022-07-21 | 1 | 4.10 | 0 | 0 | 0 | 10派4.10元 |
| 2024-07-18 | 1 | 3.21 | 0 | 0 | 0 | 10派3.21元 |
| 2025-07-16 | 1 | 4.10 | 0 | 0 | 0 | 10派4.10元 |
| 2026-07-16 | 1 | 4.20 | 0 | 0 | 0 | 10派4.20元 |

And a known 配股 on `sz000001` 平安银行:

| date | cat | f1 | f2 | f3 | f4 | interpretation |
|---|---:|---:|---:|---:|---:|---|
| 2000-11-06 | 1 | 0 | 8.00 | 0 | 3.00 | 配股价 8.00 元，10配3 |
| 2024-06-14 | 1 | 7.19 | 0 | 0 | 0 | 10派7.19元 |

These are economically sane (dividends match the announced 每10股派息; the 2000 rights issue is 8元/10配3). **[VERIFIED against the real file; interpretation follows [SECONDARY] [mkmerich notes](https://mkmerich.com/%E9%80%9A%E8%BE%BE%E4%BF%A1%E5%92%8C%E5%90%8C%E8%8A%B1%E9%A1%BA%E6%9C%AC%E5%9C%B0%E6%95%B0%E6%8D%AE%E8%A7%A3%E6%9E%90%E7%AC%94%E8%AE%B0.html) / [Gitee readTDX_cw.py](https://gitee.com/mayfan11/stock-analysis/blob/master/readTDX_cw.py).]**

---

## 2. `gbbq.map` — what it actually is **[VERIFIED]**

```
size = 96,465 bytes
= 6,431 lines × 15 bytes     (exact; 0 remainder)

each 15-byte line:
  bytes 0..5   : 6 ASCII digits  (stock code)
  bytes 6..9   : 4 ASCII digits  (small decimal number)
  bytes 10..11 : two spaces
  bytes 12..14 : 0D 0D 0A        (CR CR LF)
```

Real first lines (hex + text):

```
30 30 30 30 30 31 39 37 32 33 20 20 0d 0d 0a   "0000019723  \r\r\n"
30 30 30 30 30 32 39 37 33 36 20 20 0d 0d 0a   "0000029736  \r\r\n"
30 30 30 30 30 36 39 37 33 34 20 20 0d 0d 0a   "0000069734  \r\r\n"
...
60 30 30 30 30 30 39 36 38 37 20 20 0d 0d 0a   "6000009687  \r\r\n"
```

**The "`code → file offset`" belief is wrong.** The 4-digit field ranges 3,421–9,749, is non-monotonic, and is far too small to be an offset into a 5.79 MB file. Secondary sources describe it as an update/batch id ("更新批次", "表示 gbbq 里面该股票的数据最近一次是更新是哪一次"). **[SECONDARY: [CSDN](https://blog.csdn.net/yeyiqun/article/details/116247057)]**

Cross-check against the decoded `gbbq`: the map has **6,431** codes while `gbbq` has **6,608** distinct `(market, code)` symbols — 289 `gbbq` symbols are absent from the map (largely `market=2` 北交所/新三板 `43/83/87` codes), and 112 map codes have no `gbbq` records. So the map is an index/roster, but **not** an offset table. **[VERIFIED]**

---

## 3. Categories **[VERIFIED counts; SECONDARY meanings]**

Category histogram in the real file (all 15 codes exist):

| cat | count | documented meaning | fields that are meaningful |
|---:|---:|---|---|
| 1 | 65,422 | **除权除息** | f1 每10股分红(元), f2 配股价(元), f3 每10股送转股, f4 每10股配股 — **all four are price-relevant** |
| 2 | 4,261 | 送配股上市 | share counts |
| 3 | 5,367 | 非流通股上市 | share counts |
| 4 | 2 | 未知股本变动 / 国家股配售 (sources differ) | not reliably interpretable |
| 5 | 114,905 | 股本变化 | share counts (used for 流通/总股本) |
| 6 | 69 | 增发新股 | share counts |
| 7 | 18 | 股份回购 | share counts |
| 8 | 1,103 | 增发新股上市 | share counts |
| 9 | 7,847 | 转配股上市 | share counts |
| 10 | 274 | 可转债上市 | share counts |
| 11 | 409 | 扩缩股 (ETF 份额折算) | float ratio |
| 12 | 8 | 非流通股缩股 | float ratio |
| 13 | 9 | 送认购权证 | warrant fields |
| 14 | 15 | 送认沽权证 | warrant fields |
| 15 | 91 | 重整调整 *(eltdx only)* | float ratio |

Sources for meanings: [Gitee readTDX_cw.py](https://gitee.com/mayfan11/stock-analysis/blob/master/readTDX_cw.py) (cats 1–14), [eltdx GBBQ doc](https://github.com/electkismet/eltdx/blob/main/docs/methods/7709-%E8%82%A1%E6%9C%AC%E5%8F%98%E8%BF%81GBBQ.md) (adds cat 15 and a different cat-4 label), [stockso TDX v6 format](https://www.stockso.com/blog/blogpost/5c013eb83f3ca92684f90659).

**For categories 2/3/5/6/7/8/9/10 the four floats are share counts, in units of 10,000 shares (万股).** This is confirmed numerically on the real file: `sz000001` on 2014-06-12 (cat 9) has f1=557,590.2, f2=952,074.6, f3=669,105.9, f4=1,142,490.0 — i.e. 流通前 55.76亿 → 66.91亿, 总股本前 95.21亿 → 114.25亿, which matches 平安银行's 2014 capital change (10转2). The pytdx column names encode this as `(前流通, 前总, 后流通, 后总)`. **[VERIFIED magnitudes]**

**Only category 1 should drive price 复权.** Categories 2–10 are share-structure only; 11/12/15 are ratio events that some libraries also apply (notably ETF 份额折算 via cat 11). **[SECONDARY: [mootdx2 PyPI](https://pypi.org/project/mootdx2/1.4.0/), [eltdx](https://github.com/electkismet/eltdx)]**

---

## 4. 复权 algorithm

### 4.1 Single-event 除权因子

For a category-1 event with ex-date `t`, using the **previous trading day's close** `C(t−1)`:

```
D = f1 / 10        # cash dividend per share
P = f2             # rights-issue price
B = f3 / 10        # bonus + transfer shares per share
R = f4 / 10        # rights-issue shares per share

除权除息参考价  ref(t) = ( C(t-1) - D + P*R ) / ( 1 + B + R )

除权因子        r(t)   = ref(t) / C(t-1)
```

This is the standard formula. **[SECONDARY: [dsx_base_algorithm](https://github.com/dsxkline/dsx_base_algorithm) (Chinese derivation), [tdxrs ADJUSTER_ALGORITHM.md](https://github.com/jiangtaovan/tdxrs/blob/main/docs/ADJUSTER_ALGORITHM.md)]**

### 4.2 后复权 / 前复权 series

Let events be `t_1 < t_2 < … < t_n` with factors `r_i = r(t_i)` (usually < 1).

```
后复权 (hfq):  hfq(d) = raw(d) * Π_{ t_i ≤ d } ( 1 / r_i )
               -> 首日/第一个除权日前价格不变，之后价格被放大

前复权 (qfq):  qfq(d) = raw(d) * Π_{ t_i >  d }  r_i
               -> 最新价不变（因子=1），历史价格被下调
```

Equivalently: maintain a cumulative factor `F(d) = Π_{t_i ≤ d} (1/r_i)`; then `hfq = raw·F` and `qfq = raw·F / F(last)`.

### 4.3 The critical dependency **[MOST IMPORTANT]**

`qfq(d) = raw(d) · Π_{t_i > d} r_i` depends on **every event after `d`, including the newest one**. Therefore:

- **A new 除权/除息 event multiplies a new `r_{n+1}` into every historical 前复权 price → the entire 前复权 series changes and must be fully recomputed.**
- This is why you cannot append new 前复权 bars onto a stored 前复权 series without re-basing (`前复权数据不能增量拼接`). Backward (后复权) values for a fixed past date are not disturbed by later events, but its scale still depends on the events up to that date.
- Practical consequence: store **raw prices + the event table**, and compute factors at query time (or store 后复权, or a fixed-anchor "定点复权" series). Same warning appears in [mootdx2 docs](https://pypi.org/project/mootdx2/1.4.0/) and the [tdxrs algorithm note](https://github.com/jiangtaovan/tdxrs/blob/main/docs/ADJUSTER_ALGORITHM.md).

### 4.4 Numerically validated on the real files **[VERIFIED]**

Using decoded `gbbq` + raw `sh600000.day`:

| ex-date | f1 | C(t−1) raw | ref | r |
|---|---:|---:|---:|---:|
| 2024-07-18 | 3.21 | 9.04 | 8.719 | 0.964491 |
| 2025-07-16 | 4.10 | 13.93 | 13.520 | 0.970567 |
| 2026-07-16 | 4.20 | 9.31 | 8.890 | 0.954887 |

Resulting series around the last event:

```
20260715  raw 9.31  qfq 8.89  hfq 17.3056
20260716  raw 8.85  qfq 8.85  hfq 17.2277
20260717  raw 8.87  qfq 8.87  hfq 17.2667
... latest 20260910  raw 9.35  qfq 9.35  hfq 18.2011
```

`qfq` is continuous across the ex-date and the latest close is unchanged; `hfq` preserves the early scale. Formula confirmed.

---

## 5. Existing libraries

| library | reads local `gbbq`? | exact API | status in 2026 |
|---|---|---|---|
| **`pytdx`** | **Yes** ✔ | `from pytdx.reader.gbbq_reader import GbbqReader; df = GbbqReader().get_df(path)` | PyPI **1.72** (2019-08-26); GitHub `rainx/pytdx` **archived**, last push 2020-04-15 → **unmaintained**, but the code works |
| **`mootdx`** | **No** ✘ | `mootdx.reader.Reader.factory('std', tdxdir=...).daily(symbol)` reads only `.day` (raw). 复权 goes through `mootdx.utils.adjust.to_adjust` → `get_xdxr` (online `Quotes.xdxr`) / `mootdx.utils.factor.fq_factor` (Sina) | PyPI **0.11.7** (2024-05-04); last push 2024-07-16 → not updated in 2026 (not archived) |
| **`mootdx2`** | Partly: offline reader **yes**, adjust path **no** | `mootdx2.tdx.offline.gbbq.read_gbbq(path)` → `list[GbbqRecord]` (**works**, verified). But `reader.daily(symbol, adjust='qfq'/'hfq')` → `utils.adjust.get_xdxr` = online TDX xdxr + 90-day pickle cache (not the local file) | PyPI **1.4.0** (2026-09-07) → **actively maintained in 2026** |
| `eltdx` | **No** (network protocol client) | `client.corporate.capital_changes(code)` / `client.get_gbbq(...)` (proto cmd `0x000f`); `client.helpers.factors()` computes 复权 factors locally from fetched events | PyPI **3.1.7** (2026-09-06), GitHub pushed 2026-09-06 → **active** |
| `tdxpy` | No | dependency of `mootdx`; no `gbbq` reader found | PyPI 0.2.7 (2024-03-10) |
| `easy-tdx` | — | search results claimed `easy_tdx.offline.read_gbbq(...)`, **but both the PyPI project and GitHub repo 404 on 2026-09-11** → treat as **unverified / nonexistent** | n/a |

**Verified library behaviour:** I downloaded pytdx's `gbbq_reader.py`, extracted mootdx2's `mootdx2/tdx/offline/gbbq.py` from the 1.4.0 wheel, and ran both on `D:\Tools\tdx\T0002\hq_cache\gbbq`. Both returned **199,800 records** with identical first rows (`000001, 19900301, cat 1`) and the same sh600000 events. So **`pytdx.GbbqReader` is a working pure-Python `gbbq` parser**, and mootdx2 ships an equivalent one — even though it labels the cipher "XOR" in its docstring.

**Other maintained pure-Python `gbbq` parsers:** none beyond the above were found that read the *local* file. `dsxkline/dsx_base_algorithm` is pure-Python but is a 复权-algorithm reference, not a `gbbq` parser; `zzshare`/`tdxrs` are other data sources. The most actively maintained *correct* options as of 2026-09 are **pytdx's reader (frozen but working)** and **mootdx2's `read_gbbq`**.

**Notable inconsistency in mootdx2 1.4.0:** its `Reader.xdxr()` docstring states the local `gbbq` is encrypted and "公开 Python 解密实现不存在" (no public Python decryption exists) and therefore falls back to the online feed — yet the same distribution contains a working `read_gbbq()`. So its front-page "原生支持...本地复权" claim should be read as *算法* support, not *local `gbbq` file* support.

---

## 6. Do `.day` files store raw or adjusted prices? **[VERIFIED: raw / 不复权]**

Method: for every category-1 event whose ex-date exists in the corresponding `.day` file, compare the **observed** overnight open ratio `open(ex)/close(ex−1)` against the **theoretical raw** ratio `ref/close(ex−1)`.

```
events matched to .day                              : 39,083
events with >5% theoretical ex-date gap             :  6,209
corr(theory_ratio, observed_open_ratio)             :  0.9938
mean |observed_open - theory|  (raw hypothesis)     :  0.0086
mean |observed_open - 1.0|     (adjusted hypothesis) :  0.2980
```

If `.day` were adjusted, the observed ratio would be ≈1 and the second error would dominate. It does not. Examples of large, unmistakable raw gaps:

| market | code | ex-date | prev close | ex open | raw theory ratio | observed ratio |
|---|---|---|---:|---:|---:|---:|
| SH | 600654 | 1992-12-10 | 106.50 | 10.50 | 0.1000 | 0.0986 |
| SZ | 002695 | 2016-09-22 | 74.66 | 18.33 | 0.2496 | 0.2455 |
| SZ | 002256 | 2016-09-27 | 28.72 | 7.00 | 0.2500 | 0.2437 |
| SZ | 300006 | 2016-05-24 | 35.98 | 9.99 | 0.2770 | 0.2777 |

`sh600000` also shows ordinary dividend gaps, e.g. 2026-07-16: prev close 9.31 (2026-07-15) → open 8.92 → close 8.85, matching `9.31 − 0.42 = 8.89`.

Conclusion: **`.day` = 不复权 (raw) OHLCV.** Any 前/后复权 series must be derived by applying `gbbq` factors.

**Caveat on this run:** `tdxw.exe` was running and rewrote some `.day` files during the session (e.g. `sh600000.day` grew from 1,240 to 2,183 records and its first date moved 2021-08-02 → 2015-01-05). Numbers above were re-computed after the update; the raw conclusion is unchanged.

---

## 7. Itemised source list

Verified locally (primary evidence, read-only):
- `D:\Tools\tdx\T0002\hq_cache\gbbq` (5,794,204 B), `gbbq.map` (96,465 B), `D:\Tools\tdx\vipdoc\{sh,sz}\lday\*.day`

Secondary sources:
1. CSDN — 通da信股本变迁gbbq权息文件解密: <https://blog.csdn.net/yeyiqun/article/details/116247057>
2. 通达信股本变迁文件gbbq解密方法 (3DES-style, 4176-byte table, 24+5 split): <http://word.baidu.com/view/c8178b2515cc700abb68a98271fe910ef12dae04.html>
3. mkmerich — 通达信和同花顺本地数据解析笔记: <https://mkmerich.com/%E9%80%9A%E8%BE%BE%E4%BF%A1%E5%92%8C%E5%90%8C%E8%8A%B1%E9%A1%BA%E6%9C%AC%E5%9C%B0%E6%95%B0%E6%8D%AE%E8%A7%A3%E6%9E%90%E7%AC%94%E8%AE%B0.html>
4. Gitee `mayfan11/stock-analysis` — `readTDX_cw.py` (category table): <https://gitee.com/mayfan11/stock-analysis/blob/master/readTDX_cw.py>
5. stockso — 通达信v6数据格式 (gbbq): <https://www.stockso.com/blog/blogpost/5c013eb83f3ca92684f90659>
6. pytdx `GbbqReader` source: <https://github.com/rainx/pytdx/blob/master/pytdx/reader/gbbq_reader.py> (raw: <https://raw.githubusercontent.com/rainx/pytdx/master/pytdx/reader/gbbq_reader.py>)
7. pytdx on PyPI (1.72): <https://pypi.org/project/pytdx/> · repo (archived): <https://github.com/rainx/pytdx>
8. mootdx `reader.py`: <https://github.com/mootdx/mootdx/blob/master/mootdx/reader.py>
9. mootdx `utils/factor.py` (Sina factors): <https://github.com/mootdx/mootdx/blob/master/mootdx/utils/factor.py>
10. mootdx `utils/adjust.py` (online xdxr): <https://github.com/mootdx/mootdx/blob/master/mootdx/utils/adjust.py>
11. mootdx on PyPI (0.11.7): <https://pypi.org/project/mootdx/> · repo: <https://github.com/mootdx/mootdx> · issue #67 (前/后复权): <https://github.com/mootdx/mootdx/issues/67>
12. mootdx2 on PyPI (1.4.0): <https://pypi.org/project/mootdx2/1.4.0/>
13. eltdx: <https://github.com/electkismet/eltdx> · PyPI: <https://pypi.org/project/eltdx/> · GBBQ doc: <https://github.com/electkismet/eltdx/blob/main/docs/methods/7709-%E8%82%A1%E6%9C%AC%E5%8F%98%E8%BF%81GBBQ.md>
14. tdxpy on PyPI (0.2.7): <https://pypi.org/project/tdxpy/>
15. kbxu/mootdx fork (2026-06-02): <https://github.com/kbxu/mootdx>
16. dsx_base_algorithm — 复权 factor derivation: <https://github.com/dsxkline/dsx_base_algorithm>
17. tdxrs — `docs/ADJUSTER_ALGORITHM.md`: <https://github.com/jiangtaovan/tdxrs/blob/main/docs/ADJUSTER_ALGORITHM.md>
18. zzshare: <https://github.com/zzquant/zzshare>

**Not verified / could not confirm:** the "`easy-tdx`" package (`easy_tdx.offline.read_gbbq`) surfaced by web search — both `https://pypi.org/pypi/easy-tdx/json` and the GitHub repo returned 404 on 2026-09-11. I do **not** rely on it. tdxpy's GitHub tree was also 404 (only its PyPI sdist metadata was available), so its lack of a `gbbq` reader is based on the mootdx dependency graph and the absence of any `gbbq` reference in mootdx's readers, not a direct source read.
