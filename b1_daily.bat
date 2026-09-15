@echo off
rem ===========================================================================
rem  B1 daily screen: pick the day's B1 candidates and write a TDX watchlist.
rem
rem  Use it after the day's data is on disk:
rem    1) in the TDX client, download the day's data (the after-close download)
rem    2) double-click this file, or let Task Scheduler run it
rem
rem  Everything goes to a log; nothing needs typing. Paths come from the
rem  environment ONLY -- there is deliberately no default hard-coded here.
rem  Reason (learned the hard way, see tests/test_real_data_smoke.py): a wrong
rem  default path is worse than none, because it makes "missing data" look like
rem  "nothing there", and then every run quietly screens the wrong thing.
rem
rem  Why this file has no Chinese in it: cmd.exe reads a .bat in the OEM code
rem  page (936 on a zh-CN Windows), so a UTF-8 .bat can mis-render or even
rem  mis-parse its own lines. The one Chinese name this tool needs lives in
rem  Python instead: mbt.report.WATCHLIST_NAME. Grep the README for b1_daily.bat.
rem ===========================================================================
setlocal

rem Pin the whole run to UTF-8 -- the log, and the console that prints it when
rem something fails. Left alone, Python encodes a redirected stdout in the ANSI
rem code page (936 here), and this library prints characters cp936 has no code
rem for at all (a warning sign in mbt.cli, a minus sign in mbt.metrics). The day
rem one of those reaches the log, the run dies with an encode error -- after
rem ten minutes of work. chcp 65001 is here so `type` below reads that log back
rem correctly; it is local to this script's own console.
chcp 65001 >nul
set "PYTHONUTF8=1"

rem --- knobs -----------------------------------------------------------------
rem Keep every candidate (that is the B1 rule's own default). Set a number to
rem truncate to the top N -- the list is ranked best-first either way.
set "TOP_N=all"

rem Where the run artifacts and the watchlist file go. Point this at the TDX
rem blocknew directory (T0002\blocknew) to let the client read it directly.
set "OUT_DIR=%~dp0tmp"

rem Extra arguments, empty on purpose. To switch the recent-listing rule to REAL
rem dates, set e.g.:
rem     set "EXTRA=--master D:\Tools\new_tdx\T0002\hq_cache\base.dbf"
rem It is empty because the research arms behind ADR-0012 ran with a bare
rem UniverseRules() (no listing_dates) -- matching them keeps this tool's
rem universe the same as the one those conclusions were drawn from.
set "EXTRA="
rem ---------------------------------------------------------------------------

set "ROOT=%~dp0"
set "PY=%ROOT%.venv\Scripts\python.exe"
set "LOG=%OUT_DIR%\b1_daily.log"
set "HISTORY=%OUT_DIR%\b1_daily.history.log"

if not exist "%PY%" (
    echo ERROR: no interpreter at "%PY%"
    echo        create it first:  python -m venv .venv  ^&^&  pip install -e .
    exit /b 2
)

if not defined MBT_TDX_ROOT (
    echo ERROR: MBT_TDX_ROOT is not set ^(the vipdoc root^). See the README.
    exit /b 2
)
if not exist "%MBT_TDX_ROOT%\sh" (
    echo ERROR: MBT_TDX_ROOT does not look like a vipdoc root: "%MBT_TDX_ROOT%"
    exit /b 2
)
if not exist "%MBT_TDX_ROOT%\cw" (
    echo ERROR: no financial data under "%MBT_TDX_ROOT%\cw"
    echo        B1's last two filters read PE, so vipdoc\cw is required.
    exit /b 2
)

if not defined MBT_TDX_GBBQ (
    echo ERROR: MBT_TDX_GBBQ is not set ^(the gbbq adjustment-events file^). See the README.
    exit /b 2
)
if not exist "%MBT_TDX_GBBQ%" (
    echo ERROR: MBT_TDX_GBBQ points at something that is not there:
    echo        "%MBT_TDX_GBBQ%"
    exit /b 2
)

if not exist "%OUT_DIR%" mkdir "%OUT_DIR%"

rem The evaluation day is NOT typed anywhere: mbt screen picks the most recent
rem day whose data is complete, and prints which day it used. That matters --
rem the last bar in a TDX file is often only half written (measured on this
rem machine: 5 of 4,718 symbols), and screening that day would look like
rem "quiet day, few candidates" rather than a data problem.
rem
rem What overwrites and what does not: today's watchlist overwrites yesterday's
rem (TDX reads it by name, so it has to), and the timestamped directory under
rem OUT_DIR is never touched. The log rotates instead of accumulating: the
rem previous run is pushed into the history file, so the log always holds
rem exactly this run. That is what lets the failure path at the bottom dump it
rem -- against an append-only log, a failure in month two would spill months of
rem history onto the console, burying the error it is trying to show.
if exist "%LOG%" (
    type "%LOG%" >> "%HISTORY%"
    del "%LOG%"
)
echo ===== %DATE% %TIME% ===== > "%LOG%"
"%PY%" -m mbt screen ^
    --screen b1 ^
    --top-n %TOP_N% ^
    --tdx-root "%MBT_TDX_ROOT%" ^
    --gbbq "%MBT_TDX_GBBQ%" ^
    --cw-root "%MBT_TDX_ROOT%\cw" ^
    --watchlist-dir "%OUT_DIR%" ^
    --output-dir "%OUT_DIR%\screens" ^
    --progress %EXTRA% >> "%LOG%" 2>&1

set "CODE=%ERRORLEVEL%"
if not "%CODE%"=="0" (
    echo FAILED ^(exit %CODE%^). Log follows; the file is "%LOG%".
    type "%LOG%"
    exit /b %CODE%
)

echo OK. Log: "%LOG%"  --  the watchlist file sits in "%OUT_DIR%".
exit /b 0
