# `backtrader` + pandas 2.x / numpy 2.x compatibility — fact report

Date of research: **2026-09-11**
Target environment under evaluation: **Windows 10, Python 3.10.6, pandas 2.3.3, numpy 2.2.6** (nothing installed; fresh virtualenv).
Scope: facts and evidence only. Options and their trade-offs are listed; no recommendation is made.

Evidence classes are marked inline:

- **[V]** = verified from a primary source (PyPI JSON API, GitHub REST API, raw source file, or an actual run in a local virtualenv).
- **[S]** = secondary source (third-party page/documentation, not the upstream maintainer).
- **[I]** = my inference (explicitly not verified).

---

## TL;DR

- The latest PyPI release of `backtrader` is **1.9.78.123**, published **2023-04-19**. Upstream (`github.com/mementum/backtrader`) is **effectively unmaintained**: the last commit on `master` is `b853d7c9` dated **2023-04-19**, and there are no GitHub Releases. A repo push event occurred **2024-08-19**, but the `master`/`development` branches have not advanced past the 2023 release commit. **[V]**
- A full `Cerebro` backtest using `bt.feeds.PandasData`, `bt.feeds.PandasDirectData`, weekly `resampledata`, and indicators **ran successfully** on the exact target stack (Python 3.10.6 + pandas 2.3.3 + numpy 2.2.6 + backtrader 1.9.78.123). **[V]**
- The pandas 2.0 removals in the task premise (`DataFrame.iteritems`, `DataFrame.append`, `Int64Index`) **do exist in pandas 2.3.3** but are **not called by `backtrader`**. `backtrader/feeds/pandafeed.py` only uses `columns.values`, `.iloc`, `.index[...]`, `.itertuples()` and `.to_pydatetime()`, all present in pandas 2.3.3. **[V]**
- The real, reproducible compatibility defect is a **Python 3.10** issue, not pandas/numpy: `backtrader/lineiterator.py` uses `collections.Iterable` (removed in 3.10) inside `bindlines()`/`bind2lines()`, raising `AttributeError: module 'collections' has no attribute 'Iterable'`. Normal backtests do not hit this path. A 2026 PR (#520) fixes it but is **open/unmerged**. **[V]**
- No issue or PR in the upstream repo mentions "pandas 2" or "numpy 2". The compat PRs that do exist are all about `collections.abc`. **[V]**
- `vectorbt` (≥1.1.0) and `lib-pybroker` (≥2.0.0) **cannot be installed** on Python 3.10: they require Python ≥3.11. **[V]**
- The actively developed `cloudQuant/backtrader` fork **was verified to run** on the exact target stack (`bt.__version__` = 1.3.0, `bindlines()` OK, `PandasData` Cerebro run OK). **[V]**

---

## 1. Latest release, publish date, upstream maintenance

### 1.1 PyPI

| Version | Uploaded (UTC) | Wheel |
|---|---|---|
| 1.9.74.123 | 2019-05-30 | `backtrader-1.9.74.123-py2.py3-none-any.whl` |
| 1.9.75.123 | 2020-05-28 | `backtrader-1.9.75.123-py2.py3-none-any.whl` |
| 1.9.76.123 | 2020-07-03 | `backtrader-1.9.76.123-py2.py3-none-any.whl` |
| 1.9.77.123 | 2023-04-16 | `backtrader-1.9.77.123-py2.py3-none-any.whl` |
| **1.9.78.123** | **2023-04-19T14:13:18Z** | `backtrader-1.9.78.123-py2.py3-none-any.whl` |

- **Latest = `backtrader` 1.9.78.123, published 2023-04-19.** **[V]** — PyPI JSON API: <https://pypi.org/pypi/backtrader/json>; project page: <https://pypi.org/project/backtrader/>; release page: <https://pypi.org/project/backtrader/1.9.78.123/>.
- Declared runtime dependency: none. Only optional extra: `matplotlib ; extra == 'plotting'`. `requires_python` is unset (empty string). **[V]**
- Declared classifiers stop at Python 3.7 (`Programming Language :: Python :: 3.7`), i.e. the metadata predates modern-Python support claims. **[V]**
- No release has been published for the last ~3.4 years (as of 2026-09-11). **[V]**

### 1.2 GitHub upstream (`mementum/backtrader`)

| Fact | Value | Source |
|---|---|---|
| `pushed_at` | 2024-08-19T17:47:36Z | GitHub REST API |
| Last commit on `master` | `b853d7c9`, **2023-04-19T14:13:08Z**, "Version 1.9.78.123" | GitHub REST API |
| Last commit on `development` | same `b853d7c9` (2023-04-19) | GitHub REST API |
| GitHub Releases | **none** (empty list) | GitHub REST API |
| Tags | `1.94.15.104`, `1.9.74.123`, `1.9.72.122`, … (no tag for 1.9.75/76/77/78) | GitHub REST API |
| Stars / open issues / archived | 23,226 / 63 / `archived=false` | GitHub REST API |
| `backtrader/backtrader` | resolves to the same repo/data as `mementum/backtrader` (org alias) | GitHub REST API |

- The most recent *tag* visible via the tags API is `1.94.15.104`, but the most recent *commit* is the 1.9.78.123 version bump on 2023-04-19. Other branches are stale: `numpylines` last commit 2017-02-19, `fix-compression` 2018-10-10, `merge_memento_backtrader` 2020-07-06. **[V]**
- The `pushed_at` of 2024-08-19 is later than the last `master` commit; this indicates some push to *some* ref (tag/branch) in Aug 2024, but **not** to `master` or `development`. I could not identify which ref without a local clone (git clone was blocked by a proxy in this environment). **[V for the dates; I for the interpretation of which ref]**
- **Conclusion [V/I]:** upstream is not actively maintained. Last release 2023-04-19; last `master` commit 2023-04-19; zero GitHub Releases; 63 open issues.

---

## 2. Compatibility with pandas 2.3.3 / numpy 2.2.6 on Python 3.10

### 2.1 Method (verification)

I created a fresh virtualenv with **Python 3.10.6** (the exact target minor) and installed:

```
pip install "backtrader==1.9.78.123" "pandas==2.3.3" "numpy==2.2.6"
# -> Successfully installed backtrader-1.9.78.123 numpy-2.2.6 pandas-2.3.3 python-dateutil-2.9.0.post0 pytz-2026.3.post1 six-1.17.0 tzdata-2026.3
```

I then ran a `Cerebro` backtest with an SMA/CrossOver strategy over a synthetic 300-bar DataFrame. Results: **[V]** (local run)

| Test | Result |
|---|---|
| `bt.feeds.PandasData` (DatetimeIndex) | **OK** — final value 99992.15 |
| `bt.feeds.PandasDirectData` (itertuples, numeric indices) | **OK** — final value 99992.15 |
| `bt.feeds.PandasData` + `cerebro.resampledata(timeframe=Weekly)` | **OK** — final value 99992.15 |
| `bt.feeds.PandasData` with datetime as a `set_index('date')` | **OK** — 120 bars |
| `bt.feeds.PandasData(dataname=df, datetime='date')` | **OK** — 120 bars |
| `lineiterator.bindlines()` / `bind2lines()` | **FAIL** — `AttributeError: module 'collections' has no attribute 'Iterable'` |

Python/pandas/numpy/backtrader versions printed at runtime: `3.10.6 | pandas 2.3.3 | numpy 2.2.6 | backtrader 1.9.78.123`. **[V]**

### 2.2 The pandas 2.0 removals exist — but backtrader does not call them

On pandas 2.3.3, the three APIs named in the task are confirmed removed: **[V]** (local run)

```
[REMOVED] DataFrame.iteritems: AttributeError: 'DataFrame' object has no attribute 'iteritems'
[REMOVED] DataFrame.append:    AttributeError: 'DataFrame' object has no attribute 'append'
[REMOVED] pd.Int64Index:       AttributeError: module 'pandas' has no attribute 'Int64Index'
```

Official removal record (primary, pandas docs): pandas 2.0.0 "What's new" — <https://pandas.pydata.org/docs/whatsnew/v2.0.0.html> **[V for the removals themselves; the doc URL is the upstream record]**

**Actual pandas APIs referenced by `backtrader/feeds/pandafeed.py` (the file that matters):** **[V]**

Source inspected: raw upstream file
<https://raw.githubusercontent.com/mementum/backtrader/master/backtrader/feeds/pandafeed.py> and the identical file shipped in the installed 1.9.78.123 wheel (`site-packages/backtrader/feeds/pandafeed.py`).

`PandasData`:
- `list(self.p.dataname.columns.values)` — still valid in pandas 2.3.3
- `[x.lower() for x in self.p.dataname.columns.values]` — valid
- `len(self.p.dataname)` — valid
- `self.p.dataname.iloc[self._idx, colindex]` — valid (uses `.iloc`, not the removed `.ix`)
- `self.p.dataname.index[self._idx]` — valid
- `tstamp.to_pydatetime()` — valid on `pandas.Timestamp`

`PandasDirectData`:
- `self.p.dataname.itertuples()` — valid
- `row[colidx]` (namedtuple indexing) — valid
- `tstamp.to_pydatetime()` — valid

There is **no call to `iteritems`, no call to `DataFrame.append`, and no `Int64Index` reference** in `pandafeed.py`. The only mention of `.ix` is a comment: `# Transform names (valid for .ix) into indices (good for .iloc)`. **[V]**

A package-wide grep of the upstream source (378 files, downloaded from
<https://github.com/mementum/backtrader/archive/refs/heads/master.zip>) found: **[V]**

- `Int64Index` / `Float64Index`: **0 matches**.
- `DataFrame.append`: **0 matches** (every `.append(` in the codebase is on a Python `list` or `collections.deque`).
- `iteritems`: only in `backtrader/utils/py3.py`, where it is **backtrader's own Python-2/3 shim** operating on plain dicts:
  - `def iteritems(d): return d.iteritems()` (on Python 2 branch)
  - `def iteritems(d): return iter(d.items())` (on Python 3 branch)
  - Callers operate on dicts, e.g. `order.py`: `for key, val in iteritems(kwargs)`; `strategy.py`: `for name, data in iteritems(self.env.datasbyname)`. `analyzers/pyfolio.py` imports `iteritems` from `backtrader.utils.py3` (not from pandas). **[V]**

### 2.3 Python 3.10 `collections` breakage (real, reproducible)

`collections.Iterable` (and `Mapping`, `Callable`, etc.) were removed from the `collections` top-level namespace in **Python 3.10**. **[V]** — confirmed locally: on Python 3.10.6, `hasattr(collections, 'Iterable') == False`.

Package-wide grep results for `collections`: **[V]**

- `backtrader/cerebro.py` — already patched: `collectionsAbc = collections.abc` (with fallback).
- `backtrader/writer.py` — already patched the same way.
- `backtrader/lineiterator.py` — **NOT patched**:
  - line ~229: `elif not isinstance(owner, collections.Iterable):`
  - line ~237: `elif not isinstance(own, collections.Iterable):`
  - Source: <https://raw.githubusercontent.com/mementum/backtrader/master/backtrader/lineiterator.py>

Reproduction: creating an indicator and calling `sma.bindlines()` on Python 3.10.6 raises
`AttributeError: module 'collections' has no attribute 'Iterable'`. **[V]**
Normal backtests (Cerebro → strategy → `next()`) do **not** call `bindlines`, so they are unaffected — which is why my full backtest passed. **[V/I]**

Other `collections` usages (`deque`, `defaultdict`, `OrderedDict`, `collections.abc`) are all still valid on 3.10. **[V]**

### 2.4 numpy 2.x

The **core backtesting engine imports numpy nowhere.** A grep for `import numpy` / `np.` across the package matches only: **[V]**

- `backtrader/plot/plot.py` (`np.array`, `np.isfinite`, `np.full_like`)
- `backtrader/plot/locator.py` (`np.floor`)
- `backtrader/plot/multicursor.py` (`np.arange`, `np.sin`, `np.pi`)
- `backtrader/talib.py` (`np.array`)
- plus a couple of `pandas`/`numpy` mentions inside `try:` imports in `indicators/hurst.py` / `indicators/ols.py`.

None of these use numpy aliases removed in numpy 2.0. A targeted grep for `np.float_`, `np.complex_`, `np.unicode_`, `np.string_`, `np.bool8`, `np.int0`, `np.NaN`, `np.asscalar`, etc. returned **0 matches** in the whole package. **[V]**
Empirically, a full run with numpy 2.2.6 succeeded (see 2.1). **[V]**

### 2.5 Other pandas-touching modules (for completeness)

- `backtrader/tradingcal.py` — `pd.DatetimeIndex([0.0])`, `pd.DataFrame(index=...)`: valid in pandas 2.3.3. **[V]**
- `backtrader/analyzers/pyfolio.py` — `pandas.to_datetime`, `DataFrame.from_records`, own `iteritems`: valid in pandas 2.3.3. Also lazily imported. **[V]**
- `backtrader/indicators/ols.py` — `OLS_BetaN.next()` calls `pd.ols(...)`. **`pandas.ols` was removed in pandas 0.20** (well before pandas 2), so this one indicator is broken on all modern pandas, independent of the pandas-2 question. **[V for the source line; the pandas 0.20 removal is well-established history — I did not re-verify the exact removal release from a primary pandas changelog]**

### 2.6 Caveats on what I did *not* verify

- I did not test pandas **3.x** with backtrader. Note pandas 3.0.5 (2026-07-22) requires Python ≥3.11, so it **cannot be installed** on Python 3.10.6 anyway. **[V]**
- I did not test the full indicator library, live brokers (IB/Oanda), `TA-Lib`, or the `btrun` CLI.
- I did not test `bt.feeds.PandasData` with every dtype/shape (e.g. `MultiIndex`, nullable dtypes, timezone-aware index). **[V for the cases in 2.1 only]**

---

## 3. GitHub issues / discussions about pandas 2.x and numpy 2.x

### 3.1 What exists

I fetched **all 256 issues + PRs** (state=all) from the upstream repo via the GitHub REST API and filtered titles and bodies. **[V]**

**No issue or PR in `mementum/backtrader` mentions "pandas 2".** **[V]**
**No issue or PR body mentions "numpy 2" or "numpy" at all.** **[V]**
Bodies matching `pandas` (5 PRs, none about pandas 2):

| # | Type | State | Created | Title | URL |
|---|---|---|---|---|---|
| 500 | PR | open | 2024-08-24 | Create vectar | https://github.com/mementum/backtrader/pull/500 |
| 447 | PR | open (unmerged) | 2021-05-06 | fix pandas TypeError when using calendar | https://github.com/mementum/backtrader/pull/447 |
| 385 | PR | closed | 2019-05-20 | Fix `voloverlay=False` printing whole DataFrame as legend label | https://github.com/mementum/backtrader/pull/385 |
| 175 | PR | closed | 2016-10-20 | Fix deprecation warning to_datetime() -> to_pydatetime() | https://github.com/mementum/backtrader/pull/175 |
| 173 | PR | closed | 2016-10-19 | Avoid raising exception in analyzers.SQN if no trade made | https://github.com/mementum/backtrader/pull/173 |

> Note on #447 ("fix pandas TypeError when using calendar"): this is about `pandas_market_calendars` integration in `tradingcal.py`, **not** about the pandas-2 feed API. It is open and unmerged (19 files). **[V]**

Searches for `iteritems`, `Int64Index`, `append in:title` scoped to the repo returned **0 relevant matches**. **[V]**

### 3.2 The compat PRs that *do* exist (all about `collections.abc`, i.e. Python 3.10)

| # | Type | State | Created | Title | Notes |
|---|---|---|---|---|---|
| **520** | PR | **open (unmerged)** | **2026-05-16** | **Fix bindlines on Python 3.10+ by using collections.abc.Iterable** | 2 files, +59/−2; patches exactly the two `collections.Iterable` lines in `lineiterator.py` and adds a test. Author: `williamhsiao0403`. Base `master`. |
| 466 | PR | open (unmerged) | 2022-01-21 | Update to collections.abc | 1 file, +6/−1 |
| 465 | PR | open (unmerged) | 2022-01-21 | Update to collections.abc | same author |
| 461 | PR | closed | 2021-12-08 | Error: AttributeError: module 'collections' has no attribute 'Iterable' when running QuickStartGuide sample | 11 files |
| 477 | PR | closed | 2022-12-11 (closed 2023-01-21) | Fix import of collections.Iterable | — |
| 476 | PR | closed | 2022-11-20 | collections iterable fix | — |
| 523 | PR | **closed unmerged** | 2026-06-02 | Feature/compatibility | 468 files, +52,687/−14,915; body: "Includes tests to validate if the code is still compatible with existing samples from Backtrader." Closed **22 seconds** after opening (created 09:25:01Z, closed 09:25:23Z). |
| 472 | PR | closed (merged 2023-04-16) | 2022-09-18 | "Here you can find a link for new backtrader with 3 commits !!" | 6 files; merged just before the 1.9.78.123 release |

URLs: <https://github.com/mementum/backtrader/pull/520>, `/pull/523`, `/pull/466`, `/pull/465`, `/pull/461`, `/pull/477`, `/pull/476`, `/pull/472`, `/pull/447`. **[V]**

**No linked fix branch for pandas 2 or numpy 2 exists in the upstream repo.** The only compat fix branch I found is `williamhsiao0403:fix-bindlines-collections-abc` (PR #520), which is about Python 3.10, not pandas/numpy. **[V]**

### 3.3 Important correction to the premise

The task states: *"Find GitHub issues/discussions about pandas 2.x and numpy 2.x incompatibility."* The verified finding is that **upstream has no such issues**, because `pandafeed.py` was already migrated to `.iloc`/`.itertuples` before the 1.9.78.123 release (2023-04), and the core engine has no numpy dependency at all. The genuine compatibility gap that is documented in the issue tracker is **Python 3.10 `collections.Iterable`** (PRs #461/#465/#466/#476/#477/#520). **[V]**

---

## 4. Latest pandas / numpy known to work with unmodified backtrader

- **Verified by direct execution:** `backtrader==1.9.78.123` works with **pandas 2.3.3** and **numpy 2.2.6** on **Python 3.10.6** for `PandasData`, `PandasDirectData`, resampling and indicators. This is the highest combination I tested, and it matches the target environment exactly. **[V]**
- **No upstream maximum-version statement exists.** `backtrader`'s PyPI metadata declares no runtime deps and no `requires_python`, so there is no official upper bound on pandas or numpy. **[V]**
- **Platform ceilings relevant to the target (as of 2026-09-11):**
  - pandas latest = **3.0.5** (2026-07-22), `requires_python >=3.11` → not installable on 3.10.6; pandas **2.3.x** is the relevant series for Python 3.10. **[V]**
  - numpy latest = **2.5.3** (2026-09-06), `requires_python >=3.12` → not installable on 3.10.6; numpy **2.2.6** is a py3.10-consistent choice. **[V]**
- **Not verified:** whether pandas 3.x works with backtrader at all (cannot be installed on the target interpreter; I did not test on 3.11+). **[V that it was untested]**

---

## 5. Maintained alternatives and forks (2026)

### 5.1 `backtrader2` (PyPI + repo)

- PyPI `backtrader2` latest = **1.9.76.123**, uploaded **2020-07-30**. Author metadata still reads `Daniel Rodriguez`; homepage points at `github.com/mementum/backtrader`; artifact `backtrader2-1.9.76.123-py3-none-any.whl`. **[V]** — <https://pypi.org/project/backtrader2/>
- GitHub `backtrader2/backtrader`: 268 stars, **last commit 2021-11-02**, `pushed_at` 2024-03-24, **no GitHub Releases**. **[V]** — <https://github.com/backtrader2/backtrader>
- It is a *2020-era* snapshot (older than upstream 1.9.78.123 from 2023), and it does not appear to carry pandas-2 or Python-3.10 fixes. **Not an actively maintained fork.** **[V for dates; I for the "not carrying fixes" judgement based on version/date]**

### 5.2 `cloudQuant/backtrader` (active 2026 community fork)

- GitHub `cloudQuant/backtrader`: 174 stars, **default branch `development`**, `pushed_at` **2026-09-08**, latest commits **2026-08-22**, branches include `dev`, `development`, `feature/plot-backend-unification-bokeh`. **[V]** — <https://github.com/cloudQuant/backtrader>
- GitHub Releases: **v1.3.0 (2026-07-26)**, **v1.2.0 (2026-06-01)**. **[V]**
- README claims: "full compatibility with the original backtrader… improved stability across Python 3.8–3.13", "Fixed Python 3.12 and 3.13 compatibility issues", "Made `backtrader.analyzers.pyfolio` lazy-load `empyrical` so `import backtrader` works without it on pandas 3". **[V for the claims; they are maintainer claims, not independently verified by me]**
- Source inspection of the **`development`** branch `lineiterator.py` shows **`collections.abc.Iterable`** — i.e. it fixes the Python 3.10 `bindlines` bug that upstream still has. (The `master` branch is stale and still uses `collections.Iterable`.) **[V]**
- `pyproject.toml` on `development` declares `name = "backtrader"`, `version = "0.2.0"`, `requires-python = ">=3.8"`, license MIT, author `cloudQuant <yunjinqi@qq.com>`, dependencies `matplotlib`, `pandas`, `numpy` (unpinned). Note the declared project name is literally `backtrader`, and the version in `pyproject.toml` differs from the release tag `v1.3.0`. **[V]**
- Install method per README: `pip install -U git+https://github.com/cloudQuant/backtrader.git` (or a Gitee mirror `git+https://gitee.com/yunjinqi/backtrader.git`). **No PyPI package found** under `backtrader-cloudquant`, `backtrader3`, etc. **[V]**
- **Empirically verified on the target stack.** I downloaded the `development` branch archive and ran it with Python 3.10.6 + pandas 2.3.3 + numpy 2.2.6. Results: **[V]** (local run; the archive download took ~6 minutes, which is why an earlier attempt appeared to hang — it had in fact completed)
  ```
  fork backtrader file: ...\backtrader-development\backtrader\__init__.py
  version attr: 1.3.0
  bindlines OK
  FINAL: 10000.0
  ```
  - `bt.__version__` reports **1.3.0** (note: the `pyproject.toml` `version` field still reads `0.2.0`, so the runtime `__version__` and the declared version disagree). **[V]**
  - `bt.ind.SMA(...).bindlines()` **succeeds** here, unlike upstream 1.9.78.123 where it raises `AttributeError` on Python 3.10 (§2.3). **[V]**
  - A `Cerebro` run with `PandasData` + SMA/CrossOver completed without error. `FINAL: 10000.0` equals the starting cash because that particular random series produced no crossover entries, so **no trade was executed**; this verifies the code path runs, not the strategy's trading behavior. **[V]**
  - Tested by putting the unpacked source tree on `sys.path`/`PYTHONPATH`, **not** via `pip install`; the README's `pip install -U git+https://github.com/cloudQuant/backtrader.git` path was not exercised. **[V]**

### 5.3 Other community forks

- `WISEPLAT/backtrader` (45 stars): last commit `b853d7c9` **2023-04-19**, i.e. an unmodified mirror of upstream. **[V]**
- Of forks named `backtrader` sorted by recent update (GitHub repo search), most 2026-pushed entries are downstream *applications* (strategies, dashboards, API bridges) rather than engine forks; the only notable actively-developed engine fork with releases in 2026 is `cloudQuant/backtrader`. **[V for the search results; I for the summary judgement]**

### 5.4 Non-fork alternatives

| Project | PyPI latest | Uploaded | Python | Key pandas/numpy constraints | Repo last push | Active? |
|---|---|---|---|---|---|---|
| **vectorbt** | **1.1.0** | 2026-07-05 | `>=3.11,<3.15` | `numpy>=2.4.6`, `pandas>=3.0.3,<4.0` | 2026-08-02 | yes |
| vectorbt (compatible line) | 1.0.0 / 0.28.5 | 2026-04-22 / 2026-03-26 | `>=3.10` | `numpy>=1.23`, `pandas>=2.0,<3.0` | — | yes |
| **backtesting.py** (`backtesting`) | **0.6.6** | 2026-07-22 | `>=3.9` | `numpy>=1.17.0`, `pandas>=0.25.0` | 2026-08-05 | yes |
| **zipline-reloaded** | **3.1.1** | 2025-07-19 | `>=3.10` | `pandas>=1.3.0,<3.0`; `numpy>=1.23.5` (py<3.12) | 2026-01-06 | yes (slower) |
| **bt** | **1.2.0** | 2026-04-25 | `>=3.9` | `ffn>=1.1.2`, `pyprind`, `tqdm` (pandas via ffn) | 2026-09-10 | yes |
| **pybroker** (`lib-pybroker`) | **2.0.1** | 2026-08-28 | **`>=3.11`** | `numpy>=1.26.4,<3`, `pandas>=2.2.0,<4`, plus `akshare` | 2026-09-07 | yes |
| **qlib** (`pyqlib`) | **0.9.7** | 2025-08-15 | `>=3.8` | `pandas>=0.24`, `numpy` (unpinned in core) | 2026-09-02 | yes |

Sources (PyPI JSON + GitHub REST API): **[V]**
<https://pypi.org/pypi/vectorbt/json>, <https://pypi.org/pypi/backtesting/json>, <https://pypi.org/pypi/zipline-reloaded/json>, <https://pypi.org/pypi/bt/json>, <https://pypi.org/pypi/lib-pybroker/json>, <https://pypi.org/pypi/pyqlib/json>;
<https://github.com/polakowo/vectorbt>, <https://github.com/kernc/backtesting.py>, <https://github.com/stefan-jansen/zipline-reloaded>, <https://github.com/pmorissette/bt>, <https://github.com/edtechre/pybroker>, <https://github.com/microsoft/qlib>.

Additional notes:

- **vectorbt**: the 1.1.0 release is a Rust-backed line that **requires Python ≥3.11 and pandas ≥3.0.3 / numpy ≥2.4.6**, so it is **incompatible with the target Python 3.10.6 / pandas 2.3.3 / numpy 2.2.6**. The last releases compatible with py3.10 + pandas 2.x are **1.0.0** (2026-04-22) and **0.28.5** (2026-03-26), both declaring `pandas>=2.0,<3.0` and `numpy>=1.23`. **[V]**
- **backtesting.py**: no GitHub Releases; latest distribution is PyPI 0.6.6. Repo `kernc/backtesting.py` last pushed 2026-08-05. **[V]**
- **zipline-reloaded**: last PyPI release 2025-07-19; repo pushed 2026-01-06. Maintained but with a slower release cadence. It is a full platform (bundles `alembic`, `sqlalchemy`, `bcolz-zipline`, `exchange-calendars`, etc.). **[V]**
- **bt**: PyPI latest is 1.2.0 (2026-04-25), while GitHub Releases show v1.2.2 published 2026-09-07; I did not find a corresponding PyPI upload for 1.2.1/1.2.2 in the PyPI JSON (latest PyPI = 1.2.0). **[V]**
- **pybroker**: published on PyPI as **`lib-pybroker`** (not `pybroker`, which 404s). `requires_python = ">=3.11"`, so **not installable on Python 3.10.6**. It also lists `akshare>=1.17.50` as a test extra (A-share data source). **[V]**
- **qlib**: published as **`pyqlib`**. Core deps include `pandas>=0.24` and unpinned `numpy`; the `rl` extra pins `numpy<2.0.0`. Repo actively developed (pushed 2026-09-02). Whether its test suite passes on pandas 2.3.3/numpy 2.2.6 was **not verified by me**. **[V for metadata; V that compatibility untested]**

---

## 6. Concrete dependency pins each option would need (facts, not recommendations)

These are the pins implied by the metadata plus the target interpreter (Python 3.10.6). "Fine on target" means the package's own declared constraints permit py3.10.6 + pandas 2.3.3 + numpy 2.2.6; only backtrader was actually executed. **[V for metadata; where marked, V for execution]**

| Option | Install spec | Declared constraints | On Python 3.10.6 / pandas 2.3.3 / numpy 2.2.6 |
|---|---|---|---|
| backtrader (upstream) | `backtrader==1.9.78.123` | no runtime deps; optional `matplotlib` | **Runs (verified).** `bindlines`/`bind2lines` need the PR #520 patch on py3.10 |
| backtrader2 | `backtrader2==1.9.76.123` | no runtime deps | older 2020 snapshot; not verified |
| cloudQuant fork | `pip install -U git+https://github.com/cloudQuant/backtrader.git` (branch `development`) | `matplotlib`, `pandas`, `numpy` (unpinned); `requires-python>=3.8` | **Runs (verified)** — `bt.__version__` = 1.3.0; `bindlines()` OK; `PandasData` Cerebro run OK |
| vectorbt (py3.10 line) | `vectorbt==1.0.0` (or `0.28.5`) | `pandas>=2.0,<3.0`, `numpy>=1.23`, `numba>=0.60`, py`>=3.10` | metadata permits; not executed |
| vectorbt (latest) | `vectorbt==1.1.0` | `python>=3.11`, `pandas>=3.0.3`, `numpy>=2.4.6` | **Cannot install on 3.10.6** |
| backtesting.py | `backtesting==0.6.6` | `numpy>=1.17.0`, `pandas>=0.25.0`, py`>=3.9` | metadata permits; not executed |
| zipline-reloaded | `zipline-reloaded==3.1.1` | `pandas>=1.3.0,<3.0`, `numpy>=1.23.5` (py<3.12), py`>=3.10` | metadata permits; not executed |
| bt | `bt==1.2.0` (PyPI) | `ffn>=1.1.2`, `pyprind>=2.11`, `tqdm>=4`, py`>=3.9` | metadata permits; not executed |
| pybroker | `lib-pybroker==2.0.1` | `python>=3.11`, `numpy>=1.26.4,<3`, `pandas>=2.2.0,<4` | **Cannot install on 3.10.6** |
| qlib | `pyqlib==0.9.7` | `pandas>=0.24`, `numpy` unpinned, py`>=3.8` | metadata permits; not executed |

Environment-ceiling pins for the target interpreter (facts): **[V]**
- `pandas==2.3.3` is consistent with Python 3.10 (pandas 3.0.5 requires ≥3.11).
- `numpy==2.2.6` is consistent with Python 3.10 (numpy 2.5.3 requires ≥3.12).
- A pinned-file form of the verified-working stack is exactly: `backtrader==1.9.78.123`, `pandas==2.3.3`, `numpy==2.2.6`, Python 3.10.6.

---

## 7. Source index

**Primary — PyPI**
- `backtrader` JSON: <https://pypi.org/pypi/backtrader/json> · project: <https://pypi.org/project/backtrader/> · release: <https://pypi.org/project/backtrader/1.9.78.123/>
- `backtrader2`: <https://pypi.org/pypi/backtrader2/json> · <https://pypi.org/project/backtrader2/>
- `vectorbt`: <https://pypi.org/pypi/vectorbt/json> · <https://pypi.org/project/vectorbt/>
- `backtesting`: <https://pypi.org/pypi/backtesting/json>
- `zipline-reloaded`: <https://pypi.org/pypi/zipline-reloaded/json>
- `bt`: <https://pypi.org/pypi/bt/json>
- `lib-pybroker`: <https://pypi.org/pypi/lib-pybroker/json>
- `pyqlib`: <https://pypi.org/pypi/pyqlib/json>
- `pandas`: <https://pypi.org/pypi/pandas/json> · `numpy`: <https://pypi.org/pypi/numpy/json>

**Primary — GitHub API / repository**
- Repo: <https://github.com/mementum/backtrader>
- Commits API: <https://api.github.com/repos/mementum/backtrader/commits?sha=master>
- Repo API: <https://api.github.com/repos/mementum/backtrader>
- Source archive used for grepping: <https://github.com/mementum/backtrader/archive/refs/heads/master.zip>
- `pandafeed.py` raw: <https://raw.githubusercontent.com/mementum/backtrader/master/backtrader/feeds/pandafeed.py>
- `lineiterator.py` raw: <https://raw.githubusercontent.com/mementum/backtrader/master/backtrader/lineiterator.py>
- `utils/py3.py` raw: <https://raw.githubusercontent.com/mementum/backtrader/master/backtrader/utils/py3.py>
- PRs: [#520](https://github.com/mementum/backtrader/pull/520), [#523](https://github.com/mementum/backtrader/pull/523), [#472](https://github.com/mementum/backtrader/pull/472), [#466](https://github.com/mementum/backtrader/pull/466), [#465](https://github.com/mementum/backtrader/pull/465), [#461](https://github.com/mementum/backtrader/pull/461), [#477](https://github.com/mementum/backtrader/pull/477), [#476](https://github.com/mementum/backtrader/pull/476), [#447](https://github.com/mementum/backtrader/pull/447), [#500](https://github.com/mementum/backtrader/pull/500)
- Fork: <https://github.com/cloudQuant/backtrader> · releases: <https://github.com/cloudQuant/backtrader/releases> · development `lineiterator.py`: <https://raw.githubusercontent.com/cloudQuant/backtrader/development/backtrader/lineiterator.py> · `pyproject.toml`: <https://raw.githubusercontent.com/cloudQuant/backtrader/development/pyproject.toml>
- `backtrader2/backtrader`: <https://github.com/backtrader2/backtrader>
- Alternatives: <https://github.com/polakowo/vectorbt>, <https://github.com/kernc/backtesting.py>, <https://github.com/stefan-jansen/zipline-reloaded>, <https://github.com/pmorissette/bt>, <https://github.com/edtechre/pybroker>, <https://github.com/microsoft/qlib>

**Primary — pandas documentation**
- pandas 2.0.0 "What's new" (API removals): <https://pandas.pydata.org/docs/whatsnew/v2.0.0.html>

**Primary — local empirical runs**
- Fresh venv, Python 3.10.6, `backtrader==1.9.78.123`, `pandas==2.3.3`, `numpy==2.2.6`; results recorded in §2.1 and §2.3 of this report. Scripts: `compat_test.py`, `compat_test2.py` (temporary, under the system temp dir).

**Secondary**
- *(none used for any load-bearing claim; all load-bearing claims above trace to PyPI/GitHub/pandas docs or to local execution)*

---

## 8. Explicitly unverified / open items

1. Which ref was updated by the 2024-08-19 `pushed_at` event on upstream (no local clone possible; `git` was blocked by a proxy in the sandbox). **[V that it's unknown]**
2. ~~Whether the `cloudQuant` fork actually runs on the target stack~~ — **RESOLVED (verified):** it runs; `bindlines()` succeeds; `bt.__version__` = 1.3.0 (see §5.2). Not verified: the README's `pip install git+...` flow, and the fork's wider indicator/analyzer surface. **[V]**
3. Whether `vectorbt` 1.0.0 / 0.28.5, `backtesting.py` 0.6.6, `zipline-reloaded` 3.1.1, `bt` 1.2.0, or `pyqlib` 0.9.7 actually execute on Python 3.10.6 + pandas 2.3.3 + numpy 2.2.6 (only their declared metadata was checked). **[V that it's untested]**
4. Whether pandas 3.x works with backtrader (cannot be tested on Python 3.10; not tested on 3.11+). **[V that it's untested]**
5. The exact pandas release that removed `pandas.ols` (I verified the call site exists in `indicators/ols.py`; I did not re-verify the removal version from a pandas changelog). **[V for the call site; I for "removed in 0.20"]**
