# 通达信 (TDX) 专业财务数据 format — `vipdoc/cw`

**Scope.** Empirical (byte-level) investigation of the local TDX installation's `vipdoc/cw` directory
(Windows, Python 3.10.6). Every **[VERIFIED]** claim below was produced by reading actual bytes with an
independent script; **[SECONDARY]** means it comes only from an external document/repository.

> **路径已变更（2026-09-12 更正）**：本报告写于另一台机器状态上，当时记录的是 `D:\Tools\tdx\vipdoc\cw`，
> 而**本机实际安装在 `D:\Tools\new_tdx\vipdoc\cw`**（`D:\Tools\tdx` 不存在）。文末提到的
> `G:\MyProject\MyBacktrader\.scratch\cw_research\`（`G:` 盘现亦不存在）是**研究期间的临时脚本**，
> 未提交、现已失。复现请用 [§Reproduce](#reproduce) 里描述的方法，不要去找那两个路径。
>
> **头部读法已更正**：下文 §2A 曾把头部写成一个 struct 串 `'<1hI1H3L'`，而该串只产出 **6** 个值、
> 与它自己的偏移表（7 个字段：0/2/6/8/10/12/16）**对不上**。**偏移表是对的**——实现按显式偏移读，
> 并在全部 147 个文件上验证了帧恒等式（见 §Reproduce）。以偏移表为准，不要用那个 struct 串。

---

## TL;DR

* `vipdoc/cw` holds **two different file families**, both plain (uncompressed) little-endian binary:
  1. **Per-report-period financial statements** — `gpcwYYYYMMDD.dat` (+ an identical `.zip`).
     147 periods, 1988-12-31 → 2026-09-30. Layout = **20-byte header + N×11-byte index + N×2336-byte data block**.
     Each block is **584 little-endian `float32`** (financial fields). **[VERIFIED]**
  2. **Per-stock "股票数据包"** — `gp{sh,sz,bj}NNNNNN.dat`. 8,969 files.
     Layout = **records of exactly 13 bytes**: `[u8 tag][u32 date][f32 a][f32 b]`, no header. **[VERIFIED]**
* **There IS a per-record 公告日 (announcement date).** In the `gpcw` block, **float index 313 (byte offset 1252)**
  is `财报公告日期` (annual/quarterly report disclosure date), stored as a `float32` holding a `YYMMDD`/`YYYYMMDD`
  integer. It is real and plausible for modern periods (e.g. `20240331` reports → median 27 days later, all in
  2024-04; `000001` 2023 annual = `2024-03-15`, `600519` = `2024-04-03`, `600000` = `2024-04-30`). **[VERIFIED]**
  *Caveat:* for very old periods the field degenerates to the report period itself (placeholder) — unreliable before ≈2005.
* `gpcw` files mix **累计 (YTD)** and **单季 (single-quarter)** fields at different offsets; e.g. `其中：营业收入`
  (float 73) is cumulative while `营业收入` (float 229) is single-quarter. **[VERIFIED]**
* Historical `gpcw` files are **survivor-filtered**: every code in old files still exists in the newest file, and
  long-delisted codes (600001, 600806, 002450, 600485, 000939 …) are **absent from every period**. **[VERIFIED]**
* Parsers: **mootdx 0.11.7** (MIT) and **mootdx2 1.4.0** (MIT, released 2026-09-07) read `gpcw`;
  **pytdx 1.72** also reads it but its repo is **archived** and it has **no license**. **No open-source parser
  could be found for the per-stock `gpXX*.dat` files.** **[VERIFIED]** (code inspection) / **[SECONDARY]** (repo metadata)

---

## 1. What is in the directory

**[VERIFIED]** (full `os.listdir` + `os.stat`; nothing loaded into RAM)

`D:\Tools\tdx\vipdoc\cw` contains **9,267 files, 2,621,401,516 bytes (2.50 GB)**:

| Family | Pattern | Count | Total size | Size range | `LastWriteTime` range |
|---|---|---|---|---|---|
| Period financials (uncompressed) | `gpcwYYYYMMDD.dat` | 147 | 712,296,317 B (679.3 MB) | 20 B … 13,065,769 B | 2026-01-09 … 2026-09-11 |
| Period financials (zipped copy) | `gpcwYYYYMMDD.zip` | 147 | 275,929,884 B (263.1 MB) | 164 B … 5,804,680 B | 2026-04-07 … 2026-09-11 |
| Per-stock packages | `gp{sh,sz,bj}NNNNNN.dat` | 8,969 | 1,632,228,507 B (1,556.6 MB) | 13 B … 1,158,690 B | 2026-04-07 … 2026-09-11 |
| Manifests / misc | `gpcw.txt`, `gpszsh.txt`, `gpszsh.local`, `gpsh881020.dat~~~` | 4 | 946,808 B | — | 2026-09-11 |

Size histograms **[VERIFIED]**:

* `gpcw*.dat`: <1 KB × 26 (empty periods), 1 KB–1 MB × 10, 1–5 MB × 51, 5–10 MB × 34, 10–20 MB × 26.
* `gpcw*.zip`: <1 KB × 27, 1 KB–1 MB × 37, 1–5 MB × 71, 5–10 MB × 12.
* per-stock: <1 KB × 67, 1 KB–1 MB × 8,901, 1–5 MB × 1.

### Distinct report periods

**[VERIFIED]** 147 distinct `YYYYMMDD` values, from `19881231` to `20260930`. Every year from 1991 onward has 4
quarter-ends (0331/0630/0930/1231) except 2026 (3); 1988/1989/1990 are partial.
26 `.dat` files are **empty** (exactly 20 bytes, `nstk=0`) — e.g. `19881231`, `19900630`, `19910331`,
`19920331` … (pre-2002 Q1/Q3 were not mandatorily disclosed) and `20260930` (period not yet filed).

Periods per calendar year (count of `gpcw*.dat`): 1988:1, 1989:1, 1990:2, **1991 … 2025: 4 each**, 2026:3.
Full period list (147 values):

```
19881231 19891231 19900630 19901231 19910331 19910630 19910930 19911231 19920331 19920630 19920930 19921231
19930331 19930630 19930930 19931231 19940331 19940630 19940930 19941231 19950331 19950630 19950930 19951231
19960331 19960630 19960930 19961231 19970331 19970630 19970930 19971231 19980331 19980630 19980930 19981231
19990331 19990630 19990930 19991231 20000331 20000630 20000930 20001231 20010331 20010630 20010930 20011231
20020331 20020630 20020930 20021231 20030331 20030630 20030930 20031231 20040331 20040630 20040930 20041231
20050331 20050630 20050930 20051231 20060331 20060630 20060930 20061231 20070331 20070630 20070930 20071231
20080331 20080630 20080930 20081231 20090331 20090630 20090930 20091231 20100331 20100630 20100930 20101231
20110331 20110630 20110930 20111231 20120331 20120630 20120930 20121231 20130331 20130630 20130930 20131231
20140331 20140630 20140930 20141231 20150331 20150630 20150930 20151231 20160331 20160630 20160930 20161231
20170331 20170630 20170930 20171231 20180331 20180630 20180930 20181231 20190331 20190630 20190930 20191231
20200331 20200630 20200930 20201231 20210331 20210630 20210930 20211231 20220331 20220630 20220930 20221231
20230331 20230630 20230930 20231231 20240331 20240630 20240930 20241231 20250331 20250630 20250930 20251231
20260331 20260630 20260930
```

Per-period record counts (`nstk`, from the header) are saved in `.scratch/cw_research/out_curve.txt`;
highlights: 1995 → 673, 2005 → 1,426, 2015 → 3,374, 2018 → 5,012, 2020 → 5,446, 2025 → 5,564.

### First 30 filenames **[VERIFIED]**

```
gpbj920000.dat  gpbj920001.dat  gpbj920002.dat  gpbj920003.dat  gpbj920005.dat
gpbj920006.dat  gpbj920007.dat  gpbj920008.dat  gpbj920009.dat  gpbj920010.dat
gpbj920011.dat  gpbj920012.dat  gpbj920014.dat  gpbj920015.dat  gpbj920016.dat
gpbj920017.dat  gpbj920018.dat  gpbj920019.dat  gpbj920020.dat  gpbj920021.dat
gpbj920022.dat  gpbj920023.dat  gpbj920026.dat  gpbj920027.dat  gpbj920028.dat
gpbj920029.dat  gpbj920030.dat  gpbj920033.dat  gpbj920035.dat  gpbj920036.dat
```

### Last 10 filenames **[VERIFIED]**

```
gpsz301688.dat  gpsz301689.dat  gpsz301696.dat  gpsz301697.dat  gpsz301699.dat
gpsz301707.dat  gpsz301717.dat  gpsz302132.dat  gpszsh.local   gpszsh.txt
```

### Manifests **[VERIFIED]**

* `gpcw.txt` (8,518 B) — one line per period package: `filename,md5,filesize`, e.g.
  `gpcw20260930.zip,00cfd49dd9b9fd920484cb0a6c3f5279,164`.
* `gpszsh.txt` (498,755 B) — same format for the per-stock packages, e.g.
  `gpsz302132.dat,a4c646562b124e3cf83bc445d02534c3,264316`.
* `gpszsh.local` (439,535 B) — MD5 cache of already-downloaded per-stock files.
* `gpsh881020.dat~~~` — 0-byte temp file.

Official download endpoints (from the shipped TDX downloader) **[VERIFIED, by source] / [SECONDARY] (URLs)**:
`https://data.tdx.com.cn/tdxfin/gpcw.txt`, `https://data.tdx.com.cn/tdxfin/{file}` (period financials),
`https://data.tdx.com.cn/tdxgp/gpszsh.txt`, `https://data.tdx.com.cn/tdxgp/{file}` (per-stock packages).
The vendor page confirms the two families are officially called **财务数据包** and **股票数据包**
(<https://www.tdx.com.cn/article/stockfin.html>).

---

## 2. Binary layout

### 2A. Period financials — `gpcwYYYYMMDD.dat`

**Frame.** Example `gpcw20251231.dat`, size **13,058,728 bytes** **[VERIFIED]**:

```
20 (header) + 5564 × 11 (index) + 5564 × 2336 (blocks) = 13,058,728   exactly
```

The frame is not ambiguous in practice: the header self-describes `nstk`, index-entry size and block size, and
**all 147 `gpcw*.dat` files** satisfy `20 + nstk*index + nstk*block == filesize` exactly (checked by scanning every file).
All non-empty files use `index_entry = 11` and `block = 2336`; the 26 empty files are a bare 20-byte header with the
block-size field set to `0xFFFFFFFC` (`nstk=0`, so the identity still holds).

**Header — 20 bytes. 按偏移读**（**不要**用 §2A 早先写的 struct 串 `'<1hI1H3L'`——它与本表对不上）**[VERIFIED]**:

| Offset | Type | Value (20251231 / 19911231) | Meaning |
|---|---|---|---|
| 0 | `u16` | `1` / `1` | format version (always 1 in this install) |
| 2 | `u32` | `20251231` / `19911231` | **report period = YYYYMMDD** |
| 6 | `u16` | `5564` / `88` | **`nstk`** — number of stock records |
| 8 | `u16` | `0` | unused |
| 10 | `u16` | `11` | index-entry size (`6s + 1c + 1L`) |
| 12 | `u32` | `2336` | **data-block size in bytes** (= 584 × 4) |
| 16 | `u32` | `0` | unused |

Hex dump of the header of `gpcw20251231.dat` **[VERIFIED]**:

```
00000000  01 00 5f 02 35 01 bc 15 00 00 0b 00 20 09 00 00   .._.5....... ...
00000010  00 00 00 00                                        ....
```

**Index — `nstk` × 11 bytes** (`struct '<6s1c1L'`) **[VERIFIED]**:

| Offset (in entry) | Type | Meaning |
|---|---|---|
| 0 | `6s` | stock code, ASCII, 6 digits (not NUL-terminated; padded with `0x00` where shorter) |
| 6 | `u8` | always `0x00` — verified over **all 303,491 index entries in the directory** |
| 7 | `u32` | absolute byte offset of this stock's 2336-byte block |

```
00000014  30 30 30 30 30 31 00 28 ef 00 00   "000001" . <0x0000ef28>
0000001f  30 30 30 30 30 32 00 48 f8 00 00   "000002" . <0x0000f848>
```

**Data block — 2336 bytes = 584 × `float32` (little-endian)** **[VERIFIED]**.
No per-block header; field *k* (0-based) is at `block_offset + 4*k`. The first block starts exactly at
`20 + nstk*11` (61224 for 20251231). First block of `000001` in `gpcw20251231.dat`:

```
0000ef28  e1 7a 04 40 e1 7a 04 40 c4 42 5f 41 00 00 ba 41   .z.@.z.@.B_A...A
0000ef38  db f9 84 40 1f 85 f7 40 71 3d 82 41 f4 4c b9 52   ...@...@q=.A.L.R
```

`e1 7a 04 40` = `2.07` → 平安银行 2025 基本每股收益 = 2.07 ✓ (matches the independent annual figure).

> Note: block size is **584 floats (2336 B)** for *every* period in this install, including 1991.
> Older third-party code that hard-codes **264 floats** (1056 B) is stale — see §6.

### 2B. Per-stock packages — `gp{sh,sz,bj}NNNNNN.dat`

**Frame.** No header, fixed **13-byte records**: `size % 13 == 0` for **all 8,969 files** **[VERIFIED]**.

```
Record (13 bytes, little-endian):
  offset 0   u8    tag
  offset 1   u32   date (YYYYMMDD)
  offset 5   f32   a
  offset 9   f32   b
```

Hex dump, `gpsz000001.dat` (first 39 bytes = 3 records) **[VERIFIED]**:

```
00000000  01 9f bc 30 01 f0 7b 4b 49 00 00 00 00 01 56 e1   ...0..{KI.....V.
00000010  30 01 b0 33 60 49 00 00 00 00 01 af e3 30 01 20   0..3`I.......0.
00000020  49 5f 49 00 00 00 00                              I_I....
```

Decoded: `tag=1 date=19971231 a=833471.0 b=0.0`; `tag=1 date=19980630 a=918331.0 b=0.0`;
`tag=1 date=19981231 a=914578.0 b=0.0`.

The stream is **not** a time series — it is a concatenation of independent tagged series. For `gpsz000001.dat`
(27,899 records, dates 1990…2026-09-11) the tag histogram is **[VERIFIED]**:

```
tag:  1:117   2:1    3:2000  4:211   5:30   6:1758  7:540   8:23   9:25   10:2000
     11:2000 12:2000 13:2000 14:3   15:5   16:2000 17:28  18:2   19:434  20:432
     21:2000 22:3    24:4    25:2000 27:2000 29:37  30:29  31:1999 32:794  33:1
     34:1    35:30   36:18   37:1    38:471  39:471  40:4   41:5    42:13   43:5
     44:1    45:242  47:128  48:4    49:29   50:2000
```

Many tags (3, 10, 11, 12, 13, 16, 21, 25, 27, 31, 50 …) are capped at exactly **2,000 records** and carry daily
dates (≈2018-06 → today) → they look like **daily series truncated to the most recent 2,000 points**. Tag 1 carries
report-period dates (0331/0630/0930/1231) → a per-report-period series. Other tags are event-driven
(e.g. tag 4 has 211 sparse records spanning 2012-08 → 2026-07). **The meaning of each tag is NOT confirmed here**
— see §6 for why (no parser exists). The maintainer of `mootdx` suggests these files feed TDX's
`GPJYVALUE` / `SCJYVALUE` functions **[SECONDARY]** (<https://github.com/mootdx/mootdx/issues/142>).

There is **no obvious announcement-date field** in this family (only one date per record, which is the series' own
observation date) — but since tag semantics are unconfirmed, treat that as an open question, not a negative proof.

---

## 3. Which financial fields are present

**[VERIFIED]** — field indices below were validated against real reported figures for `000001` and `600000`
(see §5). Offset = `4 × float_index` inside the 2336-byte block. Names are the Chinese labels shipped with
`mootdx.financial.columns` (which resolves `columns[k] ↔ float[k-1]`; the file has 581 names for 584 floats, so the
last three floats are unnamed). Scaling: all values are already in the stated unit — **no division/multiplication
is required** except that **dates are stored as integers inside `float32`**.

| float idx | byte off | Field (mootdx name) | Unit / notes |
|---|---|---|---|
| 0 | 0 | 基本每股收益 | 元 — **累计 (YTD)** |
| 1 | 4 | 扣除非经常性损益每股收益 | 元 |
| 2 | 8 | 每股未分配利润 | 元 — balance-sheet, point-in-time |
| 3 | 12 | 每股净资产 | 元 — point-in-time |
| 4 | 16 | 每股资本公积金 | 元 — point-in-time |
| 5 | 20 | 净资产收益率 | % — cumulative |
| 6 | 24 | 每股经营现金流量 | 元 |
| 73 | 292 | 其中：营业收入 | 元 — **累计 (YTD)** |
| 94 | 376 | 五、净利润 | 元 — **累计** |
| 95 | 380 | 归属于母公司所有者的净利润 | 元 — **累计** |
| 106 | 424 | 经营活动产生的现金流量净额 | 元 — **累计** |
| 229 | 916 | 营业收入 | 元 — **单季** |
| 231 | 924 | 归属于母公司所有者的净利润 | 元 — **单季** |
| 233 | 932 | 经营活动产生的现金流量净额 | 元 — **单季** |
| 237 | 948 | 总股本 | 股 — point-in-time |
| 238 | 952 | 已上市流通A股 | 股 |
| 270 | 1080 | 归属于母公司股东权益(资产负债表) | 元 — point-in-time |
| 310 | 1240 | 基本每股收益（单季度） | 元 — **单季** |
| 311 | 1244 | 营业总收入(单季度)(万元) | 万元 — **单季** |
| **312** | **1248** | **业绩预告公告日期** | **`YYMMDD` in float32; mostly 0** |
| **313** | **1252** | **财报公告日期** | **`YYMMDD` in float32 — the announcement date** |
| **314** | **1256** | **业绩快报公告日期** | **`YYMMDD` in float32; mostly 0** |

Additional confirmed structure **[VERIFIED]**:

* Records per file = `nstk` from the header (0 … 5,567 here); distinct codes per file = `nstk` minus a handful of
  duplicates (≤3 in the files checked). `gpcw20251231.dat`: `nstk=5564`, 5,561 distinct codes.
* Duplicate codes exist and are byte-identical duplicates, e.g. `gpcw20251231.dat` contains **`300750` ×2,
  `301192` ×2, `301321` ×2**, each pair pointing at two separate but identical 2336-byte blocks. De-duplicate
  on code (or on block bytes) before use.
* 303,491 index entries scanned across all `gpcw*.dat`: **all** codes match `^\d{6}$`, the 7th byte is always `0x00`.

---

## 4. CRITICAL — the 公告日 / announcement date

**Answer: YES — there is a real per-record announcement date.** **[VERIFIED]**

* Field **`财报公告日期`**, **float index 313, byte offset 1252** within each stock's 2336-byte block.
* Encoding: a `float32` whose (rounded) integer value is `YYMMDD` (6 digits, leading zero dropped) or `YYYYMMDD`
  (8 digits). `240315` → `2024-03-15`.
* The other two dates (`业绩预告公告日期` = float 312 @1248, `业绩快报公告日期` = float 314 @1256) are present but
  **frequently zero** (sparse); they are *not* substitutes for field 313.

### Plausibility statistics by period **[VERIFIED]**

`lag = announcement_date − report_period_end`, in days (only records with a non-zero, decodable date):

| Period | records | zero | date < period (bad) | date == period | real (> period) | lag p25 / median / p75 | dominant month |
|---|---|---|---|---|---|---|---|
| 19911231 | 88 | 0 | 0 | 88 | 0 | — | (placeholder = report period) |
| 19951231 | 673 | 0 | 0 | 670 | 3 | — | (placeholder) |
| 20011231 | 1160 | 0 | 0 | 140 | 1020 | 700 / 700 / 700 | 2003-12 (844) — implausible, not usable |
| 20051231 | 1425 | 0 | 0 | 0 | 1425 | 80 / **101** / 118 | 2006-04 (568), 2006-03 (439) |
| 20101231 | 2312 | 0 | 0 | 19 | 2293 | 78 / **90** / 116 | 2011-03 (967), 2011-04 (783) |
| 20151231 | 3374 | 0 | 0 | 0 | 3374 | 89 / **110** / 120 | 2016-04 (1446), 2016-03 (1024) |
| 20181231 | 5011 | 0 | 0 | 0 | 5011 | 99 / **115** / 582 | 2019-04 (2284), 2019-03 (1028) |
| 20231231 | 5562 | 0 | 0 | 0 | 5562 | 101 / **114** / 118 | 2024-04 (4155), 2024-03 (1098) |
| **20240331** | 5289 | 0 | 0 | 0 | 5289 | 25 / **27** / 29 | 2024-04 (5264) |
| 20240630 | 5423 | 0 | 0 | 0 | 5423 | 54 / **59** / 61 | 2024-08 (5218) |
| 20240930 | 5349 | 0 | 0 | 0 | 5349 | 26 / **29** / 30 | 2024-10 (5309) |
| 20241231 | 5562 | 0 | 0 | 0 | 5562 | 102 / **113** / 118 | 2025-04 (4376), 2025-03 (971) |
| 20250331 | 5401 | 0 | 0 | 0 | 5401 | 25 / **28** / 29 | 2025-04 (5363) |
| 20251231 | 5561 | 0 | 0 | 0 | 5561 | 100 / **113** / 118 | 2026-04 (4371), 2026-03 (1116) |
| 20260630 | 5560 | 0 | 0 | 0 | 5560 | 53 / **57** / 59 | 2026-08 (5486) |

This is exactly the expected shape (annual reports = 3.5–4 months after year-end; Q1 = late April; H1 = late
August; Q3 = late October), so the field is a genuine disclosure date — **not** the report period.

### Spot checks vs. real-world dates **[VERIFIED]**

| period | code | decoded 财报公告日期 | real-world disclosure |
|---|---|---|---|
| 20231231 | 000001 | 2024-03-15 | 平安银行 2023 年报 = 2024-03-15 |
| 20231231 | 600519 | 2024-04-03 | 贵州茅台 2023 年报 = 2024-04-03 |
| 20231231 | 600000 | 2024-04-30 | 浦发银行 2023 年报 = 2024-04-30 |
| 20231231 | 300750 | 2024-03-16 | 宁德时代 2023 年报 |
| 20240331 | 000001 | 2024-04-20 | 平安银行 2024 一季报 |
| 20240331 | 600519 | 2024-04-27 | 贵州茅台 2024 一季报 |
| 20240630 | 000001 | 2024-08-16 | 平安银行 2024 中报 |
| 20241231 | 600519 | 2025-04-03 | 贵州茅台 2024 年报 |
| 20260630 | 000001 | 2026-08-16 | 平安银行 2026 中报 |
| 19911231 | 000001 | 1991-12-31 | placeholder (= period, no separate date) |

### How to use it (point-in-time discipline)

```python
ann_int = int(round(block[313]))          # e.g. 240420
s = str(ann_int).zfill(6)                 # '240420'
yy = int(s[:2]); year = 2000 + yy if yy < 70 else 1900 + yy
ann = datetime.date(year, int(s[2:4]), int(s[4:6]))   # 2024-04-20
usable = ann > period_end and ann.year >= 2005        # guard against placeholders/garbage
```

**Practical reliability:** usable from ≈2005 onward (and certainly 2012+). For periods ≤1995 the field equals the
report period; the 2001 file is internally inconsistent (median lag ≈700 days) — do not trust pre-2005 values
without a sanity filter.

**已由实现固化（2026-09-12）**：上述两条已成为代码行为，而不再只是研究结论——见
`src/mbt/data/fundamental.py`（`MIN_ANNOUNCEMENT_YEAR = 2005`、`FinancialRecord.usable`）与
`tests/test_fundamental.py`。取数一律走「评估日 → 该日**已公告**的最近一期」，不可用者视为
「无财务数据」，相关过滤返回假（排除）并留痕。

---

## 5. Per-report-period vs per-report-type; 累计 vs 单季

**One file per report period**, not per statement type. **[VERIFIED]** — `gpcwYYYYMMDD.dat` exists for each
quarter-end; each record is one stock's *combined* financial snapshot (income statement + balance sheet + cash
flow + derived ratios), i.e. an entire "report period" row, matching the `report_date` column of the parsers.

**Both cumulative and single-quarter values are stored, at different offsets.** Verified against 2024 annuals
(units: 元, except EPS in 元/股) **[VERIFIED]**:

`000001` (平安银行, 2024 annual 营业收入 146.6 bn, 归母净利 44.5 bn, EPS 2.15):

| float idx | field | 2024Q1 | 2024H1 | 2024Q3 | 2024FY | interpretation |
|---|---|---|---|---|---|---|
| 73 | 其中：营业收入 | 38,769,999,872 | 77,131,997,184 | 111,582,003,200 | **146,694,995,968** | **累计 (YTD)** — FY = annual ✓ |
| 95 | 归母净利润 | 14,931,999,744 | 25,878,999,040 | 39,729,000,448 | **44,508,000,256** | **累计** ✓ |
| 0 | 基本每股收益 | 0.66 | 1.23 | 1.94 | **2.15** | **累计** ✓ |
| 229 | 营业收入 | 38,769,999,872 | 38,362,001,408 | 34,450,001,920 | 35,113,000,960 | **单季** — sum = 146.70 bn ✓ |
| 231 | 归母净利润 | 14,931,999,744 | 10,947,000,320 | 13,850,000,384 | 4,778,999,808 | **单季** — sum = 44.51 bn ✓ |
| 310 | 基本每股收益（单季度） | 0.66 | 0.57 | 0.71 | 0.21 | **单季** — sum = 2.15 ✓ |
| 3 | 每股净资产 | 21.42 | 21.23 | 21.67 | 21.89 | point-in-time |

`600000` (浦发银行, 2024 annual 营业收入 170.5 bn, 归母净利 45.4 bn, EPS 1.36) behaves identically:
float 73 → 45.33 / 88.25 / 129.84 / **170.75 bn** (累计 ✓); float 229 → sum 170.75 bn (单季 ✓);
float 95 → 17.42 / 26.99 / 35.22 / **45.26 bn** (累计 ✓); float 231 → sum 45.26 bn (单季 ✓).

**Rule of thumb:** fields 73/94/95/106/0/5 are cumulative YTD; 229/231/233/281/310/311 are single-quarter;
balance-sheet fields (2/3/4/237/270) are point-in-time. Field names alone (`营业收入` vs `其中：营业收入`) do not
reveal which is which — use the index.

---

## 6. Existing open-source parsers

| project | version / date | can read `gpcw`? | exact API | license | 2026 status |
|---|---|---|---|---|---|
| **mootdx** | 0.11.7, **2024-05-04** (PyPI) | **Yes** | `mootdx.affair.Affair.files() / .fetch(downdir, filename) / .parse(downdir, filename)`; low level `mootdx.financial.FinancialReader().to_data(path)` (accepts `.dat` or `.zip`, writes `tdxfin/` URL, `to_df(header='zh')` gives Chinese columns) | **MIT** (`LICENSE` present; PyPI `License: MIT`) | not archived; GitHub API records last push 2024-07-16 |
| **mootdx2** | 1.4.0, **2026-09-07** (PyPI) | **Yes** (same code) | same module names under `mootdx2.*`: `mootdx2.financial.FinancialReader().to_data(path)`; offline readers in `mootdx2.tdx.offline` cover daily/min bars + `gbbq` only, **not** `cw` | **MIT** (wheel ships `licenses/LICENSE`) | **new PyPI release 4 days before this report**; its metadata still points at `github.com/mootdx/mootdx` (whose last recorded push is 2024-07-16), so treat "actively maintained" as PyPI-release-level, not repo-activity-level |
| **mootdx2** | 1.4.0, **2026-09-07** (PyPI) | **Yes** (same code) | same module names under `mootdx2.*`: `mootdx2.financial.FinancialReader().to_data(path)`; offline readers in `mootdx2.tdx.offline` cover daily/min bars + `gbbq` only, **not** `cw` | **MIT** (wheel ships `licenses/LICENSE`) | **actively maintained** — release 4 days before this report; repo `github.com/mootdx/mootdx` |
| **pytdx** | 1.72, **2019-08-26** (PyPI) | **Yes** | `from pytdx.reader import HistoryFinancialReader; HistoryFinancialReader().get_df('gpcw20251231.dat')` (`.dat` or `.zip`); crawler `pytdx.crawler.history_financial_crawler.HistoryFinancialCrawler`; CLI `hqreader -d hf ...` | **NONE** — PyPI `license` is empty, no `LICENSE` file in the sdist, `setup.py` declares none | **GitHub repo `rainx/pytdx` is ARCHIVED** (last push 2020-04-15); unusable for current data (downloads from a third-party mirror `data.yutiansut.com`, list from `gitee.com/yutiansut/QADATA`) |
| **tdxpy** | 0.2.7, 2024-03-10 | n/a (transport) | dependency of mootdx/mootdx2 (`TdxHq_API`) | **MIT** | `github.com/mootdx/tdxpy` → 404 at time of writing |

Notes **[VERIFIED, by source inspection]**:

* `mootdx` / `mootdx2` parse the header as `'<1hI1H3L'` and each index entry as `'<6s1c1L'`, then read
  `report_size/4` floats **dynamically** — this matches the empirical layout in §2 exactly (584 floats here).
* `pytdx`'s `HistoryFinancialCrawler.parse()` uses the same structs and the same dynamic float count
  (`report_fields_count = int(report_size / 4)`), so it parses the layout correctly; only its *download* path is dead.
* The Chinese field names come from `mootdx/financial/columns.py` (581 entries). Alignment is
  `columns[k] ↔ float[k-1]` (i.e. `columns[314] = '财报公告日期' ↔ float 313`), consistent with the empirically
  verified announcement-date index. `pytdx` does **not** ship these names — it returns generic `col1 … col584`.
* **Do not use `mootdx.utils.gpcw()`** — it is stale: it hard-codes `'<264f'` and `return`s inside the loop
  (returns only the first stock).
* `mootdx.tools.DownloadTDXCaiWu.DownloadTDXCaiWu` downloads **both** families from
  `data.tdx.com.cn/tdxfin/…` and `data.tdx.com.cn/tdxgp/…`, verifying MD5 against `gpcw.txt` / `gpszsh.txt`.
* **No open-source parser for `gp{sh,sz,bj}NNNNNN.dat` was found.** The `mootdx` reader only implements
  `gpcw`; `mootdx` issue #142 (2025-10-28, still open) asks exactly this and has no reader as an answer
  **[SECONDARY]** (<https://github.com/mootdx/mootdx/issues/142>).
* pytdx does expose the 历史专业财务数据 parser in its docs and CLI (`hf`) **[SECONDARY]**
  (<https://pytdx-docs.readthedocs.io/zh-cn/latest/pytdx_crawler/>).

---

## 7. Coverage quirks

**[VERIFIED]** (header scans + code-set comparisons across files)

* **北交所 (BJ).** In this installation BJ securities use **`920xxx`** codes: 344 of the 8,969 per-stock files are
  `gpbj920xxx.dat`, and `vipdoc\bj\lday` holds 345 `bj92xxxx.day` files. There are **no 43xxxx / 83xxxx / 87xxxx**
  codes anywhere. `gpcw` files carry 65 (2018) → 184 (2019) → 343 (2025) `920xxx` records — i.e. TDX has
  **back-filled pre-BJ (新三板) financial history** onto the new codes.
* **科创板 (688xxx).** Present: e.g. 519 in `gpcw20181231.dat` (pre-listing history), 582 in `20191231`, 616 in
  `20251231` — again back-filled for companies that listed later. `689xxx` appears once (1 code).
* **Distinct codes per file.** Grows over time: 88 (1991), 673 (1995), 1,426 (2005), 3,374 (2015), 5,012 (2018),
  5,305 (2019), 5,466 (2020), 5,564 (2025 year-end), 5,560 (2026H1). Quarterly files in a year hold fewer rows
  than the annual file. 26 files are empty (pre-2002 Q1/Q3, and the not-yet-filed `20260930`).
* **退市 (delisted) codes are absent.** Known delisted/merged tickers — `600001` (merged 2009), `600002`,
  `600005`, `600806` (2018), `002450`/`600485`/`000939`/`000511` (2019–2021), `000003`, `000405`, `000047`,
  `900901` — do **not** appear in *any* file, including periods when they were listed. Quantitatively, **100.0 %**
  of the codes in `gpcw19951231/20051231/20101231` still exist in `gpcw20251231`; across the periods scanned the only
  code that exists in old files but not the newest is `002731` (present up to `20250331`, absent from `20251231`
  onward). This is **survivorship bias** built into the data: the vendor regenerates each period file against its
  current security master, so delisted names vanish from history (file mtimes are recent: `gpcw20051231.dat` =
  2026-09-04, `gpcw20151231.dat` = 2026-09-11).
* Duplicate code rows exist within a file (see §3) and should be de-duplicated.
* `.zip` files are byte-for-byte equivalent content: `gpcw20251231.zip` contains exactly one member,
  `gpcw20251231.dat` (13,058,728 B uncompressed). Empty periods ship a 164-byte zip containing a 20-byte `.dat`.

---

## 8. Sources

Every source consulted, with what it was used for:

* Vendor — TDX 专业财务数据 download page (defines 财务数据包 / 股票数据包, install target `vipdoc/cw`):
  <https://www.tdx.com.cn/article/stockfin.html> **[SECONDARY]**
* `pytdx` issue #87 — early community reverse-engineering of `gpcw` (header `<3h1H3L`, index `<6s1c1L`, 264 floats at the time): <https://github.com/rainx/pytdx/issues/87> **[SECONDARY]**
* `pytdx` issue #133 — the field-mapping discussion referenced by both readers: <https://github.com/rainx/pytdx/issues/133> **[SECONDARY]**
* `pytdx` docs — crawler + `HistoryFinancialReader` usage (`hf` CLI): <https://pytdx-docs.readthedocs.io/zh-cn/latest/pytdx_crawler/> **[SECONDARY]**
* `mootdx` issue #142 (2025-10-28, open) — how to read the per-stock `gpsh*/gpsz*` packages; answer points at `GPJYVALUE`/`SCJYVALUE`: <https://github.com/mootdx/mootdx/issues/142> **[SECONDARY]**
* PyPI metadata (versions, release dates, license fields): <https://pypi.org/pypi/mootdx/json>, <https://pypi.org/pypi/mootdx2/json>, <https://pypi.org/pypi/pytdx/json>, <https://pypi.org/pypi/tdxpy/json> **[SECONDARY]**
* GitHub API metadata (archived flags, last push, SPDX license): `rainx/pytdx` (archived, no license), `mootdx/mootdx` (MIT) **[SECONDARY]**
* Source code read for this report: `mootdx-0.11.7` (`mootdx/financial/financial.py`, `columns.py`, `base.py`, `mootdx/affair.py`, `mootdx/utils/__init__.py`, `mootdx/tools/DownloadTDXCaiWu.py`), `mootdx2-1.4.0` (`mootdx2/financial/*`, `mootdx2/tdx/offline/__init__.py`), `pytdx-1.72` (`pytdx/reader/history_financial_reader.py`, `pytdx/crawler/history_financial_crawler.py`, `pytdx/bin/hqreader.py`, `pytdx/reader/__init__.py`). **[VERIFIED]**

### Reproduce

复现方式（2026-09-12 更正）：本报告的方法**已由实现固化**——`src/mbt/data/fundamental.py` 按显式偏移
读头部、按帧恒等式校验，`tests/test_fundamental.py` 用真实切片（`tests/fixtures/cw/gpcw20251231.dat`，
7,061 字节）锁住字段索引与公告日。要复核本报告的数字，直接读该模块的文档串与测试即可，
**不必**去找文末提到的 `.scratch` 脚本（未提交，且 `G:` 盘已不存在）。

原始研究脚本（已失，仅存档其名）：
`final_evidence.py` → `out_final.txt` (inventory, header/layout, hex dumps, field table, coverage);
`final_ann2.py` → `out_ann2.txt` (announcement-date stats); `cumq.py` (累计 vs 单季);
`curve.py` → `out_curve.txt` (per-period record counts); `sizes.py` (histograms, zip check);
`frame_scan.py` (frame constants across all `gpcw*.dat`); `gp_validate2.py` (per-stock framing);
`coverage2.py`/`coverage3.py`/`coverage4.py`/`coverage5.py` (BJ/STAR/delisted/duplicates/92xxxx investigation).

### Not determined

* The semantic meaning of each `tag` in the per-stock `gpXX*.dat` files (only the 13-byte framing and the
  tag/date histograms are confirmed).
* Whether `业绩预告公告日期` (float 312) / `业绩快报公告日期` (float 314) are period-correct when non-zero
  (they are sparse — e.g. 2,773 / 1,146 non-zero in `gpcw20240930.dat` — but some samples look like they carry a
  *later* period's date; they were not used).
* Field names for float indices 581–583 (no names shipped in `mootdx`).
