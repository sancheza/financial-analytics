# Quote Server

Serves stock and Treasury bond prices to spreadsheets over HTTP on `127.0.0.1`. Every request is answered from memory. A background thread refreshes each symbol that has been requested.

| Item | Value |
|---|---|
| Stock source | yfinance (Yahoo Finance) |
| Bond source | Webull, through `core/webull_bond_fetcher.py` |
| Address | `http://127.0.0.1:8765` |
| Refresh interval | 300 seconds. Set by `REFRESH_SECONDS` in `scripts/run_quote_server.py`; override with `--refresh-seconds` |
| Symbol list (persistent) | `data/quote_server_cache.json`. Deleting it forgets every symbol; prices are refetched when requested again |
| Source blocks (persistent) | `data/quote_server_blocks.json`, next to the cache file. Records which price source refused requests and until when. Both files are ignored by git; copy them to back them up |
| Log | The terminal for a foreground run; `logs/quote_server.log` for the LaunchAgent |

## 1. Set up

**Starting Host & Directory:** this Mac, any directory

- Python 3.10 or later as `python3` (`python3 --version`)
- Run once per checkout. Safe to re-run.

```bash
~/dev/financial-analytics/quote_server/scripts/setup.sh
```

```
Creating /Users/<user>/dev/financial-analytics/.venv
Installing dependencies
Ready. Start the server with: /Users/<user>/dev/financial-analytics/quote_server/scripts/start_quote_server.sh
```

The script creates the repo's virtualenv at `~/dev/financial-analytics/.venv`, installs `requirements.txt`, and installs the repo's packages (`core`, `quote_server`). The start script and the LaunchAgent use that virtualenv automatically.

> [!IMPORTANT]
> Run any other Python command in this repo with `~/dev/financial-analytics/.venv/bin/python`. Another Python on this Mac can hold a different repo's `core` package and fail with an `ImportError` (section 9).

## 2. Start the server

**Starting Host & Directory:** this Mac, any directory

```bash
~/dev/financial-analytics/quote_server/scripts/start_quote_server.sh
```

```
2026-10-26T21:04:35Z INFO:     quote_server 1.0.0 starting on 127.0.0.1:8765, refresh every 300s, 0 symbols loaded
```

Stop with Ctrl+C.

> **Alternate: start at login (LaunchAgent)**
>
> ```bash
> ~/dev/financial-analytics/.venv/bin/python ~/dev/financial-analytics/quote_server/scripts/install_launch_agent.py
> ```
>
> ```
> Loaded com.<user>.quoteserver. Log: /Users/<user>/dev/financial-analytics/logs/quote_server.log
> Server answering on port 8765: refresh every 300.0 s, 50 symbols
> ```
>
> The agent starts at login and restarts the server if it exits. Run the installer again to update it; it waits for the running server to stop (about 5 seconds), loads the new one, and prints `Server answering` only when the server responds on its port.

### Options

| Option | Applies to | Default | Effect |
|---|---|---|---|
| `--refresh-seconds N` | start script, agent installer | 300 | Seconds between full refreshes. Minimum 5 |
| `--port N` | start script, agent installer | 8765 | Port on `127.0.0.1` |
| `--cache-file PATH` | start script | `data/quote_server_cache.json` | Where the symbol list is kept. The source-block file is kept in the same folder |
| `--request-delay N` | start script | 0.25 | Seconds between the starts of two requests to one price source |
| `--restart-on-hang` | start script; the agent installer always sets it | off | Exit when the refresh stops or hangs, so the LaunchAgent starts a new server (section 8.2) |
| `--dry-run` | agent installer | | Print the plist; change nothing |
| `--uninstall` | agent installer | | Unload and delete the LaunchAgent |
| `-h`, `-v` | all scripts | | Help, version |

### 2.1 Change the refresh interval

The interval is the time between full refreshes of every symbol. Default 300 seconds; minimum 5.

**Starting Host & Directory:** this Mac, any directory

```bash
~/dev/financial-analytics/.venv/bin/python ~/dev/financial-analytics/quote_server/scripts/install_launch_agent.py --refresh-seconds 60
```

```
Loaded com.<user>.quoteserver. Log: /Users/<user>/dev/financial-analytics/logs/quote_server.log
Server answering on port 8765: refresh every 60.0 s, 50 symbols
```

The installer rewrites the LaunchAgent and restarts the server; the restart takes about 5 seconds. Run it without `--refresh-seconds` to return to 300.

> **Alternate: foreground run**
>
> ```bash
> ~/dev/financial-analytics/quote_server/scripts/start_quote_server.sh --refresh-seconds 60
> ```
>
> ```
> 2026-10-26T21:04:35Z INFO:     quote_server 1.0.0 starting on 127.0.0.1:8765, refresh every 60s, 0 symbols loaded
> ```

**Check the interval in use:**

```bash
curl -s http://127.0.0.1:8765/health
```

```
{"status":"ok",...,"refresh_seconds":60.0,...}
```

| Item | Where |
|---|---|
| Interval in use | `refresh_seconds` in `/health` |
| Time of the last full refresh | `last_cycle_at` in `/health` (epoch seconds; `date -r <value>` converts it) |
| Each refresh and its duration | `grep refreshed ~/dev/financial-analytics/logs/quote_server.log \| tail` |
| Interval the LaunchAgent starts with | `grep -A8 ProgramArguments ~/Library/LaunchAgents/com.$(whoami).quoteserver.plist` (no `--refresh-seconds` line means 300) |

The interval also sets when a price is stale (1.5 intervals) and expired (3 intervals), and how long the watchdog waits (3 intervals plus 60 seconds). A short interval sends more requests. The next interval starts when a refresh finishes, and 50 symbols take about 12 seconds at the default request spacing, so an interval shorter than that runs refreshes back to back.

## 3. Test the server

**Starting Host & Directory:** this Mac, any directory. The server from section 2 is running in another terminal.

| Step | Command | Expected |
|---|---|---|
| 1. Server is up | `curl -s http://127.0.0.1:8765/health` | `{"status":"ok","version":"1.0.0",...}` |
| 2. New symbol | `curl -s -w "\n[%{http_code}]\n" "http://127.0.0.1:8765/quote?symbol=AAPL"` | `PENDING`, HTTP 202 |
| 3. Wait 3 seconds, repeat step 2 | same command | A price, HTTP 200 |
| 4. Bond field | `curl -s "http://127.0.0.1:8765/quote?symbol=912810UF3&field=ytw"` (first call is `PENDING`; repeat after 3 seconds) | `"value":5.364` (the live value differs) |
| 5. Bad symbol | `curl -s -w "\n[%{http_code}]\n" "http://127.0.0.1:8765/quote?symbol=912810UF4"` | `BAD_SYMBOL`, HTTP 400 |
| 6. Speed | `curl -s -o /dev/null -w "%{time_total}s\n" "http://127.0.0.1:8765/quote?symbol=AAPL"` | About `0.001s` |

Step 3 expected output:

```json
{"symbol":"AAPL","type":"stock","field":"price","value":343.91,"fields":{"price":343.91},"stale":false,"age_seconds":2.8}
[200]
```

Step 5 expected output:

```json
{"error":{"code":"BAD_SYMBOL","message":"'912810UF4' is not a ticker and contains no valid CUSIP"}}
[400]
```

All six pass: the server fetches, caches, and answers in milliseconds.

### 3.1 Automated tests

The tests use fake price sources, so they need no network and the server need not be running.

**Starting Host & Directory:** this Mac, any directory

```bash
~/dev/financial-analytics/.venv/bin/python -m pytest ~/dev/financial-analytics/tests/test_quote_server.py ~/dev/financial-analytics/tests/test_quote_server_resilience.py ~/dev/financial-analytics/tests/test_quote_server_watchdog.py ~/dev/financial-analytics/tests/test_quote_server_yield.py ~/dev/financial-analytics/tests/test_install_launch_agent.py ~/dev/financial-analytics/tests/test_http_client.py ~/dev/financial-analytics/tests/test_calc_quote_client.py -q
```

```
235 passed
```

## 4. Request a price

**Starting Host & Directory:** any directory on this Mac

```bash
curl -s "http://127.0.0.1:8765/quote?symbol=AAPL"
```

```json
{"symbol":"AAPL","type":"stock","field":"price","value":343.91,"fields":{"price":343.91},"stale":false,"age_seconds":3.8}
```

The first request for a new symbol returns `PENDING` (section 8). The fetch starts immediately; a retry a few seconds later returns the price.

### Endpoints

| Request | Result |
|---|---|
| `GET /quote?symbol=AAPL` | Stock price |
| `GET /quote?symbol=912810UF3&field=ytw` | One bond field |
| `POST /batch-refresh` | Registers symbols, queues an immediate refresh, returns each symbol's cached result |
| `POST /refresh-all?wait=5` | Refreshes every known symbol now; returns when finished or after `wait` seconds (`complete` is `false` on timeout) |
| `GET /health` | Status (`ok`, or `degraded` when the refresh thread has stopped), version, symbol count, refresh interval, time of the last full refresh |

### Fields

| Symbol type | Field | Unit |
|---|---|---|
| Stock (ticker, index such as `^GSPC`) | `price` | Last price |
| Bond (9-character CUSIP) | `price` | Percent of par |
| Bond | `ytw` | Yield to worst, percent |
| Bond | `term` | Years to maturity |
| Stock | `dividend_yield` | Dividend yield, percent, from Yahoo Finance. `0` when Yahoo confirms the stock pays no dividend |
| Stock or bond | `yield` | The yield that fits the symbol: `ytw` for a bond, `dividend_yield` for a stock |

`yield` lets one formula cover a column that mixes bonds and stocks: `=GETQUOTE(A2; "yield")`. The response `field` shows the field used (`ytw` or `dividend_yield`). `ytw` on a stock and `dividend_yield` on a bond return `BAD_FIELD`.

**Dividend yield.** The value is Yahoo Finance's `dividendYield`, the forward yield, in percent (6.48 means 6.48%). It is fetched with one extra request, only for stocks a sheet has asked `yield` or `dividend_yield` for; the first request returns `PENDING`.

| Yahoo Finance reports | Result |
|---|---|
| A yield above 0 | The yield |
| A yield of 0 **and** 0 dividends over the last 12 months | `0` |
| A yield of 0 but dividends over the last 12 months, or no 12-month total | Error `NO_DIVIDEND_DATA` |
| No yield (a money-market fund, an index, a renamed ticker) | Error `NO_DIVIDEND_DATA` |
| A yield more than 3 times away from `dividendRate / price` | Error `SOURCE_CHANGED` (the units may have changed) |

The price of a stock is returned whatever happens to its yield.

A 9-character symbol is treated as a CUSIP and must pass the CUSIP check digit. A ticker may carry an exchange suffix (`0700.HK`, `7203.T`). A bare number is rejected with `NEED_TICKER_OR_CUSIP`.

**Text around a CUSIP.** A symbol containing exactly one valid CUSIP resolves to that CUSIP, so a label such as `5Y UST 91282CJF9` works. The response `symbol` field shows the CUSIP used.

| Submitted | Used |
|---|---|
| `5Y UST 91282CJF9` | `91282CJF9` |
| `5y ust 91282cjf9` | `91282CJF9` |
| `UST-91282CJF9`, `(91282CJF9) 5Y` | `91282CJF9` |
| `912810UF3 and 91282CJF9` | Rejected: `AMBIGUOUS_SYMBOL` (two CUSIPs) |
| `5Y UST 91282CJF8` | Rejected: `BAD_SYMBOL` (check digit wrong) |
| `Apple AAPL` | Rejected: `BAD_SYMBOL`; only CUSIPs are picked out of text, never tickers |

### Response fields

| Field | Meaning |
|---|---|
| `value` | The requested field |
| `fields` | Every field available for the symbol |
| `stale` | `true` when the price is older than 1.5 refresh intervals |
| `age_seconds` | Seconds since the price was fetched |

## 5. Request many prices

```bash
curl -s -X POST http://127.0.0.1:8765/batch-refresh \
  -H 'content-type: application/json' \
  -d '{"symbols": ["AAPL", "MSFT", "91282CMM0"], "field": "price"}'
```

```json
{"results":{"AAPL":{"symbol":"AAPL","type":"stock","field":"price","value":343.91,...},"MSFT":{"error":{"code":"PENDING","message":"MSFT is being fetched; retry shortly"}},"91282CMM0":{...}}}
```

- Up to 500 symbols per request.
- `field` is optional and defaults to `price`.
- Each symbol has its own entry: a quote or an `error` object.
- Every valid symbol is queued for a refresh on each call.

## 6. Check the server

```bash
curl -s http://127.0.0.1:8765/health
```

```json
{"status":"ok","version":"1.0.0","symbols":5,"refresher_alive":true,"refresh_seconds":300,"last_cycle_at":1793000000.0,"started_at":1792990000.0,"restart_on_hang":true,"sources":{"Webull":{"status":"ok","requests":7,"failed_calls":0},"Yahoo Finance":{"status":"ok","requests":44,"failed_calls":0}}}
```

| Field | Meaning |
|---|---|
| `status` | `ok`, or `degraded` when `refresher_alive` is `false`. Restart the server when degraded |
| `last_cycle_at` | Time of the last full refresh; `null` until the first one finishes |
| `started_at` | When the server started |
| `restart_on_hang` | `true` when the server exits if its refresh stops (section 8.2) |
| `sources` | One entry per price source: `status`, `requests`, `failed_calls`. `limited` means the source refused requests and none are sent until `cooldown_until`. `unreachable` means calls to it are being skipped for a few seconds after repeated failures |

**LaunchAgent commands**

| Task | Command |
|---|---|
| Status | `launchctl list \| grep quoteserver` |
| Restart | `launchctl kickstart -k gui/$(id -u)/com.$(whoami).quoteserver` |
| Follow the log | `tail -f ~/dev/financial-analytics/logs/quote_server.log` |
| Remove | `~/dev/financial-analytics/.venv/bin/python ~/dev/financial-analytics/quote_server/scripts/install_launch_agent.py --uninstall` |

## 7. Use from LibreOffice Calc

Adds a `GETQUOTE()` cell function, a **Quotes** menu, a **Refresh Prices** toolbar button, and a refresh when a workbook opens.

How the pieces fit:

| Piece | Behavior |
|---|---|
| Quote server | Keeps prices in memory and refreshes them every 5 minutes (section 2) |
| `GETQUOTE()` cell | Asks the server for the cached price when the cell calculates, and keeps that value until it calculates again |
| Refresh Prices | Asks the server to fetch fresh prices, then recalculates every `GETQUOTE()` cell |

> [!IMPORTANT]
> A cell does not change when the server refreshes. Use **Quotes > Refresh Prices** (section 7.4 lists every trigger).

### 7.1 Install

**Starting Host & Directory:** this Mac, any directory. Quit LibreOffice first (Cmd+Q).

```bash
~/dev/financial-analytics/.venv/bin/python ~/dev/financial-analytics/quote_server/scripts/install_calc_integration.py
```

```
Link .../Library/Application Support/LibreOffice/4/user/Scripts/python/calc_quote_client.py -> .../quote_server/libreoffice/calc_quote_client.py
Write .../LibreOffice/4/user/basic/Standard/QuoteServer.xba and list it in script.xlb
Install extension org.quoteserver.calcui with unopkg
Done. Start LibreOffice.
```

| Item installed | Location | Purpose |
|---|---|---|
| `calc_quote_client.py` (link) | `Scripts/python/` in the LibreOffice profile | Looks up prices; runs Refresh Prices |
| `QuoteServer` module | `My Macros > Standard` | Makes `GETQUOTE()` available in cells |
| Extension `org.quoteserver.calcui` | Extension Manager | **Quotes** menu, **Refresh Prices** toolbar button, refresh on open |

Options: `--dry-run` prints the steps and works while LibreOffice is running; `--uninstall` removes everything; `--profile PATH` targets another LibreOffice profile.

### 7.2 Check the install

**Starting Host & Directory:** this Mac, any directory

| Check | How | Expected |
|---|---|---|
| Client link | `ls -l ~/Library/Application\ Support/LibreOffice/4/user/Scripts/python/calc_quote_client.py` | A link to `.../quote_server/libreoffice/calc_quote_client.py` |
| Extension | `/Applications/LibreOffice.app/Contents/MacOS/unopkg list` | `Identifier: org.quoteserver.calcui` and `is registered: yes` |
| Cell function | In Calc: Tools > Macros > Organize Macros > LibreOffice Basic, then My Macros > Standard | A `QuoteServer` module |
| Menu | Start LibreOffice and open a spreadsheet | A **Quotes** menu in the menu bar |

### 7.3 Use in a sheet

**Starting Host & Directory:** a Calc spreadsheet, with the server running (section 2; the LaunchAgent starts it at login)

| Formula | Result |
|---|---|
| `=GETQUOTE(A1)` | Price of the ticker or CUSIP in A1 |
| `=GETQUOTE(A1; "ytw")` | Bond yield to worst, percent |
| `=GETQUOTE(A1; "term")` | Bond years to maturity |
| `=GETQUOTE($A2; B$1)` | Field named in a header cell (`price`, `ytw`, `term`) |

**Steps (first use)**

1. Type `AAPL` in A2 and `=GETQUOTE(A2)` in B2.
   ```
   B2: #ERR: PENDING
   ```
2. Choose Quotes > Refresh Prices.
   ```
   B2: 343.91
   ```
3. Add more rows. Each new formula shows `#ERR: PENDING` until the next Refresh Prices.

| | A | B |
|---|---|---|
| 1 | Symbol | Price |
| 2 | AAPL | `=GETQUOTE(A2)` |
| 3 | 912810UF3 | `=GETQUOTE(A3)` |

```
Symbol      Price
AAPL        343.91
912810UF3   91.82
```

**Entering symbols**

- A symbol cell can hold text or a formula result (`="AAPL"`). Changing the symbol cell recalculates its `GETQUOTE` cell.
- A cell can hold a CUSIP with other text, such as `5Y UST 91282CJF9`. The one valid CUSIP in the text is used. A ticker must be alone in its cell.
- Format a CUSIP column as Text (Format > Cells > Text) before typing. A CUSIP made only of digits, or one starting with `0`, is stored as a number, loses its leading zero, and shows `#ERR: NEED_TICKER_OR_CUSIP`. Typing `'037833100` (apostrophe first) also stores text.
- A blank symbol cell gives a blank result.
- Use the argument separator set in Tools > Options > LibreOffice Calc > Formula (`;` or `,`). A mismatch shows `Err:508`.
- The Calc client talks to port 8765 (`SERVER_URL` in `libreoffice/calc_quote_client.py`).
- `GETQUOTE` is a Calc-only function. Outside this Mac's LibreOffice, saved values display and a recalculation shows `#NAME?`.

### 7.4 When cells update

| Event | Updates `GETQUOTE` cells? |
|---|---|
| Server refreshes in the background (every 5 minutes) | No. Cells keep the value they show |
| You type or change a `GETQUOTE` formula, or the symbol cell it refers to | That cell only; it shows the server's cached price, or `#ERR: PENDING` for a new symbol |
| Quotes > Refresh Prices, or the toolbar button | Yes, every `GETQUOTE` cell, with fresh prices |
| Opening a workbook | Yes, same as Refresh Prices |
| Recalculate, F9 | No |
| Data > Calculate > Recalculate Hard (Ctrl+Shift+F9) | Yes, with the server's cached prices; also recalculates every other formula |

A price shown can be up to 5 minutes old, plus the time since the last Refresh Prices. The server marks a price stale after 1.5 refresh intervals, and Calc shows the number until the server reports `STALE` at 3 intervals. A failed refresh keeps the last price; the cell shows the cause only once the price is `STALE`.

### 7.5 What a cell shows

An error cell reads `#ERR: <CODE> - <cause>`. The cause is in plain words; the table lists the fix for each code.

| Cell shows | Cause | Fix |
|---|---|---|
| A number | Price from the server's cache | |
| `#ERR: PENDING` | Symbol is new; the first fetch is running | Quotes > Refresh Prices |
| `#ERR: NOT_FOUND` | Yahoo Finance or Webull answered that it has no listing. A text label that looks like a ticker (such as a header `Price`) also gives this | Check the symbol; confirm the formula refers to the symbol cell |
| `#ERR: NO_DIVIDEND_DATA` | Yahoo Finance does not confirm whether the stock pays a dividend, so the yield is unknown. The price still works | Look the stock up on finance.yahoo.com; a money-market fund or an index has no value |
| `#ERR: NO_FIELD` | The symbol is priced but the source supplied no value for that field (for example `ytw`) | Use `price`, or check the bond on webull.com |
| `#ERR: NEED_TICKER_OR_CUSIP` | The cell holds a number. A ticker or a CUSIP must be submitted. An 8-digit number is usually a CUSIP that lost its leading zero | Refer to the symbol cell; format a CUSIP column as Text |
| `#ERR: AMBIGUOUS_SYMBOL` | The cell contains more than one CUSIP | Keep one CUSIP per cell |
| `#ERR: BAD_SYMBOL` | Text that is not a ticker and holds no valid CUSIP, or a CUSIP failing its check digit | Correct the symbol; a ticker must be alone in its cell |
| `#ERR: BAD_FIELD` | Field not offered for that symbol type | Stocks: `price`, `dividend_yield`, `yield`. Bonds: `price`, `term`, `ytw`, `yield` |
| `#ERR: NO_NETWORK` | The server cannot reach Yahoo Finance or Webull | Check the internet connection; the server retries on its own (section 8.1) |
| `#ERR: SOURCE_TIMEOUT` | The source did not answer in time | None; the server retries on its own |
| `#ERR: RATE_LIMITED` | The source is limiting requests | None; the server asks again at the next full refresh |
| `#ERR: SOURCE_REFUSED` | The source refused the request (HTTP 401 or 403) | Wait; if it persists, the source blocks this Mac |
| `#ERR: SOURCE_DOWN` | The source answered HTTP 5xx | None; the server retries on its own |
| `#ERR: SOURCE_ERROR` | The source answered another HTTP error | Read the server log |
| `#ERR: SOURCE_CHANGED` | Webull's page or price could not be read | Section 9, Webull row; the server code needs an update |
| `#ERR: NO_DATA` | Yahoo Finance answered with no price rows | None; the server retries on its own |
| `#ERR: LOCAL_CACHE` | yfinance could not open its cache files in `~/Library/Caches/py-yfinance` | Check that folder's permissions and free disk space |
| `#ERR: INTERNAL` | The server hit a bug on this symbol | Read the server log |
| `#ERR: UPSTREAM` | The failure reason is not recorded | Quotes > Refresh Prices |
| `#ERR: STALE` | Last good price is older than 3 refresh intervals. The cell gives the age and why the last refresh failed | Fix the cause named in the cell |
| `#ERR: REFRESH_STOPPED` | The server's refresh thread has stopped | Restart the server (section 6) |
| `#ERR: OFFLINE` | The server is not running, or an earlier lookup failed in the last 10 seconds | Start the server (section 2), then Quotes > Refresh Prices |
| `#ERR: TIMEOUT` | The server did not answer within 0.5 seconds | Quotes > Refresh Prices; if it repeats, restart the server |
| `#ERR: BAD_RESPONSE` | The server's reply was not readable | Confirm port 8765 is the quote server (`curl -s http://127.0.0.1:8765/health`) |
| `#ERR: NO_VALUE` | The server answered without a number for that field | Read the server log |
| `#ERR: CLIENT` | The Python client did not load | Section 7.2, then run the installer again |
| `Err:508` | Argument separator does not match Calc's setting | Section 7.3, Entering symbols |

`SUM()` and similar functions skip `#ERR:` text. A formula such as `=A1*2` on an `#ERR:` cell shows `#VALUE!`; test with `ISNUMBER()` first.

After one failed lookup, every `GETQUOTE` cell shows `#ERR: OFFLINE` for 10 seconds without contacting the server, so an offline server cannot slow Calc down. **Refresh Prices** clears that wait.

### 7.6 Refresh Prices

- **Menu:** Quotes > Refresh Prices. **Toolbar:** Refresh Prices (show it from View > Toolbars if hidden).
- The command recalculates the `GETQUOTE` cells, asks the server to refresh every symbol, waits up to 5 seconds, and recalculates them again. Calc pauses while it waits.
- A workbook runs the same command when it finishes opening. A workbook with no `GETQUOTE` cell is not touched.
- With the server offline, the cells show `#ERR: OFFLINE` and the command returns at once.

### 7.7 Update or remove

**After pulling new code:** run the installer again (section 7.1), then restart the server. The client file updates itself through its link; the cell function and menu need the installer.

**Remove:**

```bash
~/dev/financial-analytics/.venv/bin/python ~/dev/financial-analytics/quote_server/scripts/install_calc_integration.py --uninstall
```

## 8. Error codes

An error response is `{"error": {"code": "...", "message": "..."}}`. The message is the cause in plain words; Calc shows it after the code.

| HTTP | Code | Cause | Fix |
|---|---|---|---|
| 202 | `PENDING` | Symbol is new; the first fetch is running | Retry in a few seconds |
| 400 | `NEED_TICKER_OR_CUSIP` | The value is a number (`5`, `1234.5`, `-3`). The message says to submit a ticker or a CUSIP | Submit a ticker or CUSIP; an 8-digit number is a CUSIP missing its leading zero |
| 400 | `AMBIGUOUS_SYMBOL` | The text contains more than one valid CUSIP; the message lists them | Submit one CUSIP |
| 400 | `BAD_SYMBOL` | Text that is not a ticker and holds no valid CUSIP, or a CUSIP failing its check digit | Correct the symbol |
| 400 | `BAD_FIELD` | Field not offered for that symbol type | Stocks: `price`, `dividend_yield`, `yield`. Bonds: `price`, `term`, `ytw`, `yield` |
| 404 | `NOT_FOUND` | The source answered that it has no listing. Any ticker-shaped text gets this when the source has no match | Check the symbol; it is asked again every cycle |
| 404 | `NO_FIELD` | The symbol is priced but the source supplied no value for that field | Use `price` |
| 404 | `NO_DIVIDEND_DATA` | Yahoo Finance does not confirm whether the stock pays a dividend | Section 4, Fields |
| 503 | `NO_NETWORK`, `SOURCE_TIMEOUT`, `SOURCE_DOWN`, `NO_DATA`, `LOCAL_CACHE` | The last fetch failed for a cause that may pass, and no earlier price exists | Section 8.1 |
| 503 | `RATE_LIMITED`, `SOURCE_REFUSED`, `SOURCE_ERROR`, `SOURCE_CHANGED`, `INTERNAL` | The last fetch failed for a cause that does not pass by retrying at once, and no earlier price exists | Read the message and the log |
| 503 | `STALE` | Last good price is older than 3 refresh intervals; the message gives the age and the last failure | Fix the cause in the message |
| 503 | `REFRESH_STOPPED` | The refresh thread has stopped | Restart the server (section 6) |

### 8.1 How the server handles a failed fetch

All requests to Yahoo Finance and Webull go through one HTTP client (`core/http_client.py`).

| Step | Behavior |
|---|---|
| Pacing | Requests to one source start at least 0.25 seconds apart (varied by +/-25%), with at most 2 in flight. A refresh of 50 symbols takes about 12 seconds. A symbol requested during a refresh is fetched before the rest of it |
| Attempts | Each symbol gets up to 3 attempts, waiting about 1 second, then 3 seconds (longer if the source sends `Retry-After`). Only a cause that may pass is retried: no network, timeout, HTTP 5xx, a local cache error |
| Not retried | A refusal or rate limit (HTTP 401, 403, 429, or an empty 202/204 reply), a Webull page that cannot be read, a bug |
| Short stop | After 5 failed symbols in a row, the server skips that source for 20 seconds, then tries one symbol. Each source is stopped separately |
| Block | After a refusal or rate limit, no request goes to that source for 5 minutes; a repeat after that waits 15 minutes, then 60 minutes. The block is kept in `data/quote_server_blocks.json`, so a restart keeps it. The first successful request after the wait clears it. Cells show `RATE_LIMITED` or `SOURCE_REFUSED` with the time requests resume |
| Quick re-fetch | Symbols that failed for a cause that may pass are fetched again after 15 seconds, then 30, 60, and 120 seconds until they succeed. Other symbols keep their prices |
| Last price kept | A failed fetch keeps the last good price. The price shows as `STALE` after 3 refresh intervals |
| Not found | Only an answer from the source that it has no listing gives `NOT_FOUND`. A network or source problem never does |
| Schedule | The refresh schedule follows the clock, so a refresh that came due while the Mac slept runs within 15 seconds of waking |

**Clear a block by hand:** stop the server, delete `data/quote_server_blocks.json`, start the server. Do this only when the cause is known to be gone; the source may refuse again.

### 8.2 Restart on a stopped or hung refresh

A watchdog thread inside the server checks every 30 seconds that the refresh thread is running and that a full refresh has finished within 3 refresh intervals plus 60 seconds.

| Item | Behavior |
|---|---|
| Confirmation | A problem must last 2 minutes before the watchdog acts. A refresh that resumes in that time (for example after the Mac wakes) cancels it |
| With `--restart-on-hang` | The watchdog shows a macOS notification (`Restarting: <reason>.`), saves the cache, and exits with code 70. The LaunchAgent starts a new server within 10 seconds |
| Without it | The watchdog logs the problem and shows one notification (`Not refreshing: <reason>. Restart it.`). The server keeps running |
| LaunchAgent | `install_launch_agent.py` always sets `--restart-on-hang`. Run the installer again after pulling this change |

### 8.3 Request spacing measurements

Sequential requests, slowest spacing first, 2026-10-30. A run stops at the first refusal.

| Source | Spacing | Requests | Refused | Median reply |
|---|---|---|---|---|
| Yahoo Finance | 1.0 s | 25 | 0 | 0.09 s |
| Yahoo Finance | 0.5 s | 25 | 0 | 0.09 s |
| Yahoo Finance | 0.25 s | 25 | 0 | 0.08 s |
| Yahoo Finance | 0.1 s | 25 | 0 | 0.02 s (some replies may have come from a cache) |
| Webull | 1.0 s | 12 | 0 | 0.19 s |
| Webull | 0.5 s | 12 | 0 | 0.18 s |
| Webull | 0.25 s | 12 | 0 | 0.14 s |
| Webull | 0.1 s | 12 | 0 | 0.20 s |

No limit was reached, so 0.25 seconds is a tested spacing, not a measured maximum rate.

## 9. Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| `ImportError: cannot import name 'webull_bond_fetcher' from 'core' (...other-repo/core...)` | A Python outside `.venv` | Run with `~/dev/financial-analytics/.venv/bin/python` |
| `no such file or directory: .../.venv/bin/python` | The virtualenv is not created | Section 1 |
| `ModuleNotFoundError: No module named 'quote_server'` | Setup not run | Section 1 |
| `ModuleNotFoundError: No module named 'fastapi'` (or another package) | Dependencies not installed in `.venv` | Section 1 |
| `curl: (7) Failed to connect` | Server is not running | Section 2 |
| `LibreOffice is running. Quit it (Cmd+Q)...` | The installer needs LibreOffice closed | Quit LibreOffice, run the installer again |
| `No Standard library at ...` (installer) | The LibreOffice profile has never been created | Start LibreOffice once, quit it, run the installer again |
| `launchctl bootstrap failed (...Input/output error); attempt 1 of 4, retrying in 2 s` (agent installer) | launchd is still removing the old server | None; the installer waits and retries |
| `launchctl could not load com.<user>.quoteserver after 4 attempts` (agent installer) | launchd refused the job 4 times. The server is not running | Run the `launchctl bootstrap` command the message prints; if it fails, read its error and `launchctl print gui/$(id -u)/com.$(whoami).quoteserver` |
| `The server did not answer on port N within 30 s` (agent installer) | The job is loaded but the server did not start | Read `logs/quote_server.log` |
| `unopkg failed: ...` (installer) | LibreOffice is still running in the background, or the profile is locked | Quit LibreOffice; check `pgrep -x soffice` prints nothing; run the installer again |
| `#NAME?` in a cell | The Calc integration is not installed, or LibreOffice was not restarted | Section 7.1, then restart LibreOffice |
| No Quotes menu | The extension is not installed | Section 7.2 |
| Quotes menu present, no toolbar button | The toolbar is hidden | Show it from View > Toolbars; or use the Quotes menu |
| Cell stays `#ERR: PENDING` | Cells do not update by themselves | Quotes > Refresh Prices |
| Cell stays `#ERR: PENDING` after Refresh Prices | The source cannot price the symbol, or the server is slow | `curl -s "http://127.0.0.1:8765/quote?symbol=<symbol>"` and read the server log; bonds: see the Webull row below |
| Cell shows `#ERR: OFFLINE` and the server is running | The server uses a port other than 8765, or the 10-second wait has not ended | `curl -s http://127.0.0.1:8765/health`; then Quotes > Refresh Prices |
| Price in the cell differs from the server | The cell has not recalculated since the server refreshed | Quotes > Refresh Prices |
| CUSIP shows `#ERR: NEED_TICKER_OR_CUSIP` or `#ERR: BAD_SYMBOL` but is valid | Calc stored it as a number and dropped a leading zero | Format the column as Text and retype it |
| Cell shows `#ERR: NOT_FOUND` for something that is not a ticker | The formula refers to a label cell (a header or a name) | Point the formula at the cell holding the ticker or CUSIP |
| Refresh Prices does nothing | The workbook has no `GETQUOTE` cell, or the active document is not a spreadsheet | Enter a `GETQUOTE` formula |
| Opening a workbook is slow | It waits up to 5 seconds for fresh prices while the server fetches | Keep the server running (LaunchAgent, section 2) so prices are cached |
| Calc features stopped after a LibreOffice reset or reinstall | The profile was recreated | Section 7.1 |
| `[Errno 48] address already in use` | Port 8765 is taken, often by an earlier server | `lsof -i :8765`, stop that process, or use `--port` |
| Bond stays `PENDING`, or shows `NO_NETWORK` or `SOURCE_CHANGED` | Webull unreachable or changed its page | Read the log; run `~/dev/financial-analytics/.venv/bin/python -c "from core import webull_bond_fetcher as w; print(w.fetch_treasury_price('912810UF3'))"` |
| Stock returns `NOT_FOUND` for a real ticker | Yahoo Finance answered that it has no listing | Confirm the ticker on finance.yahoo.com; the server asks again every cycle |
| Every stock shows `NO_NETWORK` while the internet works | A proxy variable (`HTTPS_PROXY`) is set for the server, or DNS fails for the server only | `launchctl kickstart -k gui/$(id -u)/com.$(whoami).quoteserver`; read the log |
| `/health` shows `"status":"degraded"`, or cells show `REFRESH_STOPPED` | The refresh thread stopped | Restart the server (section 6) and read the log lines above the traceback. With `--restart-on-hang` the server restarts itself after 2 minutes (section 8.2) |
| A notification `Quote server: Restarting: ...` | The watchdog found the refresh stopped or hung | Read `logs/quote_server.log` around the time shown |
| `/health` shows a source as `limited`, or cells show `RATE_LIMITED` | The source refused requests; none are sent until `cooldown_until` | Wait. To clear it early, section 8.1 |

## 10. Files

| Path | Role |
|---|---|
| `app.py` | HTTP routes and error codes |
| `cache.py` | Thread-safe cache and symbol-list persistence |
| `providers.py` | yfinance and Webull lookups, failure codes and messages |
| `watchdog.py` | Restarts the server when the refresh stops or hangs |
| `../core/http_client.py` | HTTP client: pacing, retries, short stop, persistent blocks |
| `refresher.py` | Background refresh thread |
| `symbols.py` | Ticker and CUSIP detection |
| `scripts/run_quote_server.py` | Server entry point |
| `scripts/setup.sh` | Creates the virtualenv and installs dependencies |
| `scripts/start_quote_server.sh` | Starts the server with the repo venv |
| `scripts/install_launch_agent.py` | Installs or removes the LaunchAgent |
| `scripts/install_calc_integration.py` | Installs or removes the Calc integration |
| `libreoffice/calc_quote_client.py` | Calc client: price lookups, error text, Refresh Prices |
| `libreoffice/QuoteServer.bas` | The `GETQUOTE()` cell function |
| `libreoffice/extension/` | Quotes menu, toolbar button, refresh on open |
| `../tests/test_quote_server.py`, `../tests/test_quote_server_resilience.py`, `../tests/test_quote_server_watchdog.py`, `../tests/test_quote_server_yield.py`, `../tests/test_install_launch_agent.py`, `../tests/test_http_client.py`, `../tests/test_calc_quote_client.py` | Tests |
