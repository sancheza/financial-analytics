#!/usr/bin/env python3
"""bond_yield_high_alert.py

Monitor configured Treasury maturity series and raise a macOS sound and
persistent popup when the live yield exceeds the highest yield already printed
in the trailing window of trading days, intraday prints included.

The window is LOOKBACK_TRADING_DAYS below, so changing the lookback is a
one-line edit.

Usage:
    python3 bond_yield_high_alert.py
    python3 bond_yield_high_alert.py --dry-run
    python3 bond_yield_high_alert.py --sound siren --raise-volume
    python3 bond_yield_high_alert.py --install-schedule 5
    python3 bond_yield_high_alert.py --schedule-status
    python3 bond_yield_high_alert.py --remove-schedule
    python3 bond_yield_high_alert.py --help
    python3 bond_yield_high_alert.py --version

Configuration in this directory's .env file:

    BOND_TYPES=2Y_NOTE,10Y_NOTE,20Y_BOND,30Y_BOND,10Y_TIPS

The configured names map to Webull's intraday yield series: the 2Y/10Y/20Y/30Y
yield indices, and for 10Y TIPS the current on-the-run CUSIP discovered from
TreasuryDirect (Webull publishes no TIPS index). A run reports every configured
series, aggregates qualifying series into one modal popup, and plays a system
sound before showing it. The popup remains open until it is acknowledged. Use
--dry-run to check the yields and conditions without playing a sound, writing a
log, or showing a popup.

Data source notes:
    Webull's chart endpoint serves 30-second bars for the last five trading days
    ("d5") and daily bars ("m1") going back further. The lookback window is the
    union: intraday bars for the five most recent sessions, plus the daily high
    for any earlier session still inside the window, so a window of more than
    five trading days is covered and a shorter one is trimmed. The current value
    is the newest in-session intraday bar, so a run before the open reports the
    prior close rather than a stale official daily observation. That newest bar
    is the live print, and the window high it is compared against is built from
    the rest of the window, so the alert means a new window high rather than a
    comparison a print can never beat.

Scheduled runs:
    --install-schedule MINUTES writes and loads a macOS LaunchAgent that runs
    this monitor on that cadence during the cash session on weekdays, and
    --schedule-status reports whether a run will fire next, the installed
    cadence, the next firing, and the last one, and --remove-schedule
    uninstalls it. Re-running
    --install-schedule regenerates and reloads the plist, so the cadence lives
    in the command rather than in a hand-edited file. See the SCHEDULED RUNS
    section of --help.
"""

from __future__ import annotations

import argparse
import getpass
import math
import os
import plistlib
import re
import subprocess
import sys
import time
import wave
from array import array
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import webull_bond_fetcher


SCRIPT_DIR = Path(__file__).resolve().parent
ENV_FILE = SCRIPT_DIR / ".env"
LOG_DIR = SCRIPT_DIR / "logs"
LOG_FILE = LOG_DIR / "bond_yield_high.log"
CACHE_DIR = SCRIPT_DIR / "cache"
ASSETS_DIR = SCRIPT_DIR / "assets"
BUNDLED_SOUND_FILE = ASSETS_DIR / "bond-high-alert.mp3"
SCHEDULE_LOG_FILE = LOG_DIR / "bond_yield_high_schedule.log"
LAUNCHD_LABEL = f"com.{getpass.getuser()}.bondanalytics.bondhigh"
LAUNCHD_PLIST_PATH = Path(
    os.path.expanduser(f"~/Library/LaunchAgents/{LAUNCHD_LABEL}.plist")
)
SCHEDULE_MINUTES_MIN = 1
SCHEDULE_MINUTES_MAX = 60
SCHEDULE_WEEKDAYS = (1, 2, 3, 4, 5)
WEEKDAY_NAMES = ("Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun")

# The comparison window, in trading days. This is the one place to change it:
# the window, the comparison, the reported label, the popup text, and the
# schedule's help text all read it, and nothing in the file or on disk carries
# the number in its name.
LOOKBACK_TRADING_DAYS = 7

VERSION = "1.2.0"
REQUEST_TIMEOUT = 15
ALERT_TITLE = f"Bond {LOOKBACK_TRADING_DAYS}D high"

MARKET_TIMEZONE = ZoneInfo("America/New_York")
# Webull's index series carry junk overnight bars (timestamps 00:00-06:59 ET
# holding a per-series sentinel instead of a yield, plus occasional bad ticks).
# Genuine prints run 07:00-17:00 ET. The cash close is the 17:00 bar: the index
# series also emits 17:05-17:40 quotes after the close, so the window keeps the
# 17:00 bar and drops the rest.
SESSION_START_HOUR = 7
SESSION_END_HOUR = 17
SESSION_CLOSE_MINUTE = 0

INTRADAY_PERIOD = "d5"
DAILY_PERIOD = "m1"
INTRADAY_BAR_COUNT = 1000
DAILY_BAR_COUNT = 30

TREASURY_SECURITIES_URL = "https://www.treasurydirect.gov/TA_WS/securities/search"
TIPS_TARGET_YEARS = 10
TIPS_TERM_TOLERANCE_YEARS = 0.75
TIPS_CUSIP_COUNT = 2

_USE_COLOR = sys.stdout.isatty()
GREEN = "\033[92m" if _USE_COLOR else ""
YELLOW = "\033[33m" if _USE_COLOR else ""
CYAN = "\033[96m" if _USE_COLOR else ""
RED = "\033[91m" if _USE_COLOR else ""
BOLD = "\033[1m" if _USE_COLOR else ""
RESET = "\033[0m" if _USE_COLOR else ""

BOND_SERIES: dict[str, tuple[str, str]] = {
    "2Y": ("2-Year Treasury Note", "2Y"),
    "2Y_NOTE": ("2-Year Treasury Note", "2Y"),
    "10Y": ("10-Year Treasury Note", "10Y"),
    "10Y_NOTE": ("10-Year Treasury Note", "10Y"),
    "20Y": ("20-Year Treasury Bond", "20Y"),
    "20Y_BOND": ("20-Year Treasury Bond", "20Y"),
    "30Y": ("30-Year Treasury Bond", "30Y"),
    "30Y_BOND": ("30-Year Treasury Bond", "30Y"),
    "10Y_TIPS": ("10-Year TIPS", "10Y_TIPS"),
}

SAMPLE_RATE = 44100
PEAK_AMPLITUDE = 0.89
FADE_SECONDS = 0.005
BEEP_COUNT = 3
SOUND_REPEAT_GAP = 0.25

# Each preset is a sequence of (frequency_hz, seconds) segments; a frequency of
# 0 is silence. Segments are synthesized at full scale, so an alternative alert
# is as loud as the system output volume allows -- use --raise-volume when that
# volume is low. "asset" plays BUNDLED_SOUND_FILE, "beeps" uses the system beep,
# and "none" is silent; none of those three are synthesized.
SOUND_PRESETS: dict[str, tuple[str, tuple[tuple[float, float], ...]]] = {
    "beeps": ("beeps", ((0.0, 0.0),)),
    "tone": ("sine", ((1046.0, 0.45), (0.0, 0.15), (1046.0, 0.45)) * 3),
    "klaxon": ("square", ((659.0, 0.30), (988.0, 0.30)) * 8),
    "siren": (
        "sine",
        (
            (523.0, 0.14), (587.0, 0.14), (659.0, 0.14), (784.0, 0.14),
            (880.0, 0.14), (988.0, 0.14), (1046.0, 0.20), (988.0, 0.14),
            (880.0, 0.14), (784.0, 0.14), (659.0, 0.14), (587.0, 0.14),
            (523.0, 0.20), (0.0, 0.30),
        ) * 2,
    ),
    "chime": (
        "sine",
        (
            (784.0, 0.40), (0.0, 0.12), (1046.0, 0.40), (0.0, 0.12),
            (1318.0, 0.55), (0.0, 0.25),
        ) * 2,
    ),
    "buzz": ("square", ((196.0, 0.55), (0.0, 0.20)) * 6),
    "none": ("sine", ((0.0, 0.0),)),
}
SOUND_CHOICES = ["asset", *sorted(SOUND_PRESETS)]
DEFAULT_SOUND = "asset"


@dataclass(frozen=True)
class SoundOptions:
    """How the alert sound should be produced."""

    preset: str = DEFAULT_SOUND
    repeats: int = 1
    sound_file: str | None = None
    raise_volume: bool = False


@dataclass(frozen=True)
class SeriesResult:
    """The live yield and the settled window high for one configured series."""

    bond_type: str
    label: str
    current_time: datetime
    current_value: float
    high_value: float
    high_date: date
    window: tuple[date, date]

    @property
    def qualifies(self) -> bool:
        """Return whether the live yield is above the settled window high."""
        return self.current_value > self.high_value

    @property
    def window_label(self) -> str:
        """Return the lookback window as an inclusive ISO date range."""
        return f"{self.window[0].isoformat()}..{self.window[1].isoformat()}"


def colorize(value: str, color: str) -> str:
    """Color terminal text while keeping redirected output plain."""
    if not _USE_COLOR:
        return value
    return f"{color}{value}{RESET}"


def normalize_bond_type(value: str) -> str:
    """Normalize a configured bond type for lookup in BOND_SERIES."""
    return re.sub(r"[\s-]+", "_", value.strip().upper())


def parse_bond_types(raw_value: str) -> list[str]:
    """Parse and validate a comma-separated BOND_TYPES value."""
    bond_types: list[str] = []
    unknown: list[str] = []

    for value in raw_value.split(","):
        bond_type = normalize_bond_type(value)
        if not bond_type:
            continue
        if bond_type not in BOND_SERIES:
            unknown.append(bond_type)
            continue
        if bond_type not in bond_types:
            bond_types.append(bond_type)

    if unknown:
        supported = ", ".join(sorted(BOND_SERIES))
        raise ValueError(
            f"Unknown BOND_TYPES value(s): {', '.join(unknown)}. "
            f"Supported values: {supported}"
        )
    if not bond_types:
        raise ValueError(
            f"No bond types configured; set BOND_TYPES in {ENV_FILE}"
        )
    return bond_types


def candle_market_time(candle: webull_bond_fetcher.YieldBar) -> datetime:
    """Return a candle's timestamp in the market timezone."""
    return datetime.fromtimestamp(candle.timestamp, MARKET_TIMEZONE)


def in_session(candle: webull_bond_fetcher.YieldBar) -> bool:
    """Return whether a candle was printed inside regular cash-market hours."""
    moment = candle_market_time(candle)
    if SESSION_START_HOUR <= moment.hour < SESSION_END_HOUR:
        return True
    return moment.hour == SESSION_END_HOUR and moment.minute == SESSION_CLOSE_MINUTE


def session_bars(
    bars: list[webull_bond_fetcher.YieldBar],
) -> list[webull_bond_fetcher.YieldBar]:
    """Keep only bars printed during regular market hours, close print included."""
    return [candle for candle in bars if in_session(candle)]


def build_daily_highs(
    intraday_bars: list[webull_bond_fetcher.YieldBar],
    daily_bars: list[webull_bond_fetcher.YieldBar],
) -> dict[date, float]:
    """Map each trading day to its highest yield across both bar sources.

    Intraday bars win for a day they cover, since a 30-second bar maximum is a
    truer high than the daily bar's summary fields.
    """
    highs: dict[date, float] = {}
    for candle in daily_bars:
        highs[candle_market_time(candle).date()] = candle.high

    intraday_highs: dict[date, float] = {}
    for candle in intraday_bars:
        day = candle_market_time(candle).date()
        if day not in intraday_highs or candle.close > intraday_highs[day]:
            intraday_highs[day] = candle.close
    highs.update(intraday_highs)
    return highs


def discover_tips_cusips(requests_module: Any, count: int = TIPS_CUSIP_COUNT) -> list[str]:
    """Return the most recent on-the-run 10-Year TIPS CUSIPs, newest first.

    Webull publishes no TIPS yield index, so the TIPS series is read off the
    current on-the-run bond. The previous issue is returned as well: a bond
    rolls roughly every three months, and a longer window can still reach
    back past the new issue's first trade.
    """
    response = requests_module.get(
        TREASURY_SECURITIES_URL,
        params={"format": "json", "type": "TIPS"},
        timeout=REQUEST_TIMEOUT,
    )
    response.raise_for_status()

    try:
        payload = response.json()
    except (TypeError, ValueError) as error:
        raise ValueError("TreasuryDirect returned invalid JSON") from error

    if not isinstance(payload, list):
        raise ValueError("TreasuryDirect returned an unexpected response")

    issue_dates: dict[str, date] = {}
    for item in payload:
        if not isinstance(item, dict):
            continue
        cusip = item.get("cusip")
        try:
            issue_date = date.fromisoformat(str(item.get("issueDate"))[:10])
            maturity_date = date.fromisoformat(str(item.get("maturityDate"))[:10])
        except ValueError:
            continue
        if not isinstance(cusip, str) or not cusip:
            continue
        years_to_maturity = (maturity_date - issue_date).days / 365.25
        if abs(years_to_maturity - TIPS_TARGET_YEARS) > TIPS_TERM_TOLERANCE_YEARS:
            continue
        if cusip not in issue_dates or issue_date > issue_dates[cusip]:
            issue_dates[cusip] = issue_date

    ranked = sorted(issue_dates.items(), key=lambda item: item[1], reverse=True)
    if not ranked:
        raise ValueError("TreasuryDirect returned no 10-Year TIPS securities")
    return [cusip for cusip, _ in ranked[:count]]


def collect_tips_series(cusips: list[str]) -> tuple[list[Any], list[Any]]:
    """Fetch intraday and daily bars for on-the-run TIPS CUSIPs, newest first.

    Days the newest issue covered take precedence, so a rollover mid-window does
    not mix two bonds' prints into the same trading day.
    """
    intraday: list[Any] = []
    daily: list[Any] = []
    claimed: set[date] = set()

    for cusip in cusips:
        ticker_id = webull_bond_fetcher.fetch_ticker_id(cusip, REQUEST_TIMEOUT)
        if not ticker_id:
            print(colorize(f"Warning: Webull does not list TIPS CUSIP {cusip}", RED))
            continue
        cusip_intraday = session_bars(
            webull_bond_fetcher.fetch_yield_bars(
                ticker_id, INTRADAY_PERIOD, INTRADAY_BAR_COUNT, REQUEST_TIMEOUT
            )
        )
        cusip_daily = webull_bond_fetcher.fetch_yield_bars(
            ticker_id, DAILY_PERIOD, DAILY_BAR_COUNT, REQUEST_TIMEOUT
        )
        for candle in cusip_intraday:
            day = candle_market_time(candle).date()
            if day not in claimed:
                claimed.add(day)
                intraday.append(candle)
        for candle in cusip_daily:
            day = candle_market_time(candle).date()
            if day not in claimed:
                claimed.add(day)
                daily.append(candle)

    return intraday, daily


def fetch_source_series(source: str, requests_module: Any) -> tuple[list[Any], list[Any]]:
    """Fetch intraday and daily bars for one configured series source."""
    if source == "10Y_TIPS":
        return collect_tips_series(discover_tips_cusips(requests_module))

    ticker_id = webull_bond_fetcher.TREASURY_YIELD_INDEXES.get(source)
    if not ticker_id:
        raise ValueError(f"No Webull yield index is configured for {source}")
    intraday = webull_bond_fetcher.fetch_yield_bars(
        ticker_id, INTRADAY_PERIOD, INTRADAY_BAR_COUNT, REQUEST_TIMEOUT
    )
    daily = webull_bond_fetcher.fetch_yield_bars(
        ticker_id, DAILY_PERIOD, DAILY_BAR_COUNT, REQUEST_TIMEOUT
    )
    return intraday, daily


def evaluate_series(
    bond_type: str,
    label: str,
    source: str,
    intraday_bars: list[Any],
    daily_bars: list[Any],
) -> SeriesResult:
    """Compare the live yield with the high of the lookback window.

    The comparison basis is the LOOKBACK_TRADING_DAYS sessions ending today,
    measured before the live print. Folding the newest bar into its own high
    would make the strict comparison unsatisfiable, because that bar is then
    always at least as large as every other value in the basis. Today's earlier
    prints stay in the basis, so re-touching a level already printed today does
    not alert, while a genuine new window high does.
    """
    session = session_bars(intraday_bars)
    if not session:
        raise ValueError(
            f"{source} returned no in-session bars between "
            f"{SESSION_START_HOUR:02d}:00 and the "
            f"{SESSION_END_HOUR:02d}:{SESSION_CLOSE_MINUTE:02d} ET close"
        )

    newest = max(session, key=lambda candle: candle.timestamp)
    current_time = candle_market_time(newest)
    today = current_time.date()

    settled = [candle for candle in session if candle.timestamp < newest.timestamp]
    daily_highs = build_daily_highs(settled, daily_bars)

    window = sorted({*daily_highs, today})[-LOOKBACK_TRADING_DAYS:]
    if len(window) < LOOKBACK_TRADING_DAYS:
        raise ValueError(
            f"{source} has {len(window)} trading days of history; "
            f"{LOOKBACK_TRADING_DAYS} are required"
        )
    basis = [(day, daily_highs[day]) for day in window if day in daily_highs]
    if not basis:
        raise ValueError(f"{source} has no settled high in the comparison window")
    high_date, high_value = max(basis, key=lambda item: (item[1], item[0]))
    return SeriesResult(
        bond_type=bond_type,
        label=label,
        current_time=current_time,
        current_value=newest.close,
        high_value=high_value,
        high_date=high_date,
        window=(window[0], window[-1]),
    )


def format_result(result: SeriesResult) -> str:
    """Format one checked series for terminal output."""
    status = "ALERT" if result.qualifies else "no alert"
    return (
        f"term={result.bond_type} label={result.label} "
        f"current_yield={result.current_value:.3f}% "
        f"{LOOKBACK_TRADING_DAYS}D_high={result.high_value:.3f}% "
        f"high_date={result.high_date.isoformat()} "
        f"window={result.window_label} "
        f"as_of={result.current_time.isoformat()} [{status}]"
    )


def build_alert_message(results: list[SeriesResult]) -> str:
    """Build the message shown in the persistent macOS popup."""
    lines = [
        "The following series just made a new "
        f"{LOOKBACK_TRADING_DAYS}-trading-day high:"
    ]
    for result in results:
        lines.append(
            f"{result.label}: {result.current_value:.3f}% "
            f"vs. previous high {result.high_value:.3f}% "
            f"on {result.high_date.isoformat()} "
            f"(as of {result.current_time.strftime('%Y-%m-%d %H:%M %Z')})"
        )
    return "\n".join(lines)


def synthesize_sound(segments: tuple[tuple[float, float], ...], waveform: str) -> array:
    """Render tone segments to 16-bit mono PCM samples."""
    samples: list[float] = []
    fade = max(1, int(SAMPLE_RATE * FADE_SECONDS))

    for frequency, seconds in segments:
        sample_count = int(SAMPLE_RATE * seconds)
        if frequency <= 0 or sample_count <= 0:
            samples.extend([0.0] * sample_count)
            continue
        for index in range(sample_count):
            value = math.sin(2 * math.pi * frequency * index / SAMPLE_RATE)
            if waveform == "square":
                value = 1.0 if value >= 0 else -1.0
            if index < fade:
                value *= index / fade
            elif index > sample_count - fade:
                value *= (sample_count - index) / fade
            samples.append(value)

    if waveform == "square" and samples:
        window = 5
        smoothed: list[float] = []
        total = 0.0
        for index, value in enumerate(samples):
            total += value
            if index >= window:
                total -= samples[index - window]
            smoothed.append(total / min(index + 1, window))
        samples = smoothed

    peak = max((abs(value) for value in samples), default=0.0)
    scale = PEAK_AMPLITUDE / peak if peak else 0.0
    pcm = array("h", (int(max(-1.0, min(1.0, value * scale)) * 32767) for value in samples))
    if sys.byteorder == "big":
        pcm.byteswap()
    return pcm


def sound_file_for(preset: str) -> Path:
    """Return the cached WAV path for a preset, rendering it if absent."""
    path = CACHE_DIR / f"alert_{preset}.wav"
    if path.exists():
        return path
    waveform, segments = SOUND_PRESETS[preset]
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    pcm = synthesize_sound(segments, waveform)
    handle: Any = wave.open(str(path), "wb")
    with handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(SAMPLE_RATE)
        handle.writeframes(pcm.tobytes())
    return path


def applescript_string(value: str) -> str:
    """Escape a string for use as an AppleScript string literal."""
    escaped = value.replace("\\", "\\\\").replace('"', '\\"')
    escaped = escaped.replace("\r", "").replace("\n", "\\n")
    escaped = escaped.replace("\t", "\\t")
    return f'"{escaped}"'


def system_output_volume() -> int | None:
    """Return the current macOS output volume, or None if unavailable."""
    try:
        result = subprocess.run(
            ["/usr/bin/osascript", "-e", "output volume of (get volume settings)"],
            check=True,
            capture_output=True,
            text=True,
            timeout=REQUEST_TIMEOUT,
        )
        return int(result.stdout.strip())
    except (OSError, ValueError, subprocess.SubprocessError):
        return None


def output_is_muted() -> bool:
    """Return whether the macOS output device is muted."""
    try:
        result = subprocess.run(
            ["/usr/bin/osascript", "-e", "output muted of (get volume settings)"],
            check=True,
            capture_output=True,
            text=True,
            timeout=REQUEST_TIMEOUT,
        )
        return result.stdout.strip() == "true"
    except (OSError, subprocess.SubprocessError):
        return False


def sound_command(sound: SoundOptions) -> tuple[list[str], str] | None:
    """Resolve the playback command and description for the alert sound.

    Returns None when the run should be silent.
    """
    repeats = max(1, sound.repeats)

    if sound.sound_file:
        path = Path(sound.sound_file).expanduser()
        if not path.is_file():
            raise RuntimeError(f"sound file not found: {path}")
        return ["/usr/bin/afplay", str(path)], f"{path.name} x{repeats}"

    if sound.preset == "none":
        return None

    if sound.preset == "asset":
        if not BUNDLED_SOUND_FILE.is_file():
            raise RuntimeError(f"bundled alert sound not found: {BUNDLED_SOUND_FILE}")
        return (
            ["/usr/bin/afplay", str(BUNDLED_SOUND_FILE)],
            f"{BUNDLED_SOUND_FILE.name} x{repeats}",
        )

    if sound.preset == "beeps":
        return ["/usr/bin/osascript", "-e", f"beep {BEEP_COUNT}"], f"{BEEP_COUNT} system beeps"

    path = sound_file_for(sound.preset)
    return ["/usr/bin/afplay", str(path)], f"{sound.preset} ({path.name}) x{repeats}"


def play_alert_sound(sound: SoundOptions) -> str:
    """Play the alert sound and return a description of what was played."""
    resolved = sound_command(sound)
    if resolved is None:
        return "no sound"
    command, description = resolved
    repeats = max(1, sound.repeats)

    previous_volume = system_output_volume() if sound.raise_volume else None
    try:
        if previous_volume is not None and previous_volume < 100:
            subprocess.run(
                ["/usr/bin/osascript", "-e", "set volume output volume 100"],
                check=True,
                timeout=REQUEST_TIMEOUT,
            )
        for index in range(repeats):
            subprocess.run(command, check=True)
            if index + 1 < repeats:
                time.sleep(SOUND_REPEAT_GAP)
    except (OSError, subprocess.SubprocessError) as error:
        raise RuntimeError(f"alert sound failed: {error}") from error
    finally:
        if previous_volume is not None and previous_volume < 100:
            subprocess.run(
                [
                    "/usr/bin/osascript",
                    "-e",
                    f"set volume output volume {previous_volume}",
                ],
                check=False,
                timeout=REQUEST_TIMEOUT,
            )

    if previous_volume is not None:
        description += f" at volume 100 (restored to {previous_volume})"
    return description


def send_macos_alert(message: str, dry_run: bool, sound: SoundOptions) -> None:
    """Play the alert sound and show a modal alert until it is acknowledged."""
    if dry_run:
        resolved = sound_command(sound)
        planned = "no sound" if resolved is None else resolved[1]
        print(colorize(f"Dry run: would play {planned}, then show a popup.", YELLOW))
        print(colorize("Dry run popup message:", CYAN))
        print(colorize(message, GREEN))
        return

    if sys.platform != "darwin":
        raise RuntimeError("macOS alerts require running this script on macOS")

    if output_is_muted():
        print(colorize("Warning: output is muted; the alert will be silent.", RED))

    played = play_alert_sound(sound)
    log(f"Played alert sound: {played}")

    script = (
        f"display alert {applescript_string(ALERT_TITLE)} "
        f"message {applescript_string(message)}"
    )
    try:
        subprocess.run(["/usr/bin/osascript", "-e", script], check=True)
    except (OSError, subprocess.SubprocessError) as error:
        raise RuntimeError(f"macOS popup failed: {error}") from error


def log(message: str) -> None:
    """Append a timestamped message to the monitor log."""
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().astimezone().strftime("%Y-%m-%d %H:%M:%S %Z")
    with LOG_FILE.open("a", encoding="utf-8") as log_file:
        log_file.write(f"{timestamp} {message}\n")


def parse_schedule_minutes(raw_value: str) -> int:
    """Parse and validate a schedule cadence in minutes."""
    try:
        minutes = int(str(raw_value).strip())
    except (TypeError, ValueError) as error:
        raise ValueError(
            f"Schedule cadence must be a whole number of minutes, got {raw_value!r}"
        ) from error
    if not SCHEDULE_MINUTES_MIN <= minutes <= SCHEDULE_MINUTES_MAX:
        raise ValueError(
            f"Schedule cadence must be between {SCHEDULE_MINUTES_MIN} and "
            f"{SCHEDULE_MINUTES_MAX} minutes, got {minutes}"
        )
    return minutes


def schedule_minutes_argument(raw_value: str) -> int:
    """Adapt the schedule cadence to argparse's error reporting."""
    try:
        return parse_schedule_minutes(raw_value)
    except ValueError as error:
        raise argparse.ArgumentTypeError(str(error)) from error


def schedule_slots(minutes: int) -> list[tuple[int, int, int]]:
    """Enumerate the (weekday, hour, minute) firings of one session schedule.

    Covers the cash session only: every `minutes`-th minute from the session open
    through the last full hour, then the closing print itself, on weekdays.
    """
    slots: list[tuple[int, int, int]] = []
    for weekday in SCHEDULE_WEEKDAYS:
        for hour in range(SESSION_START_HOUR, SESSION_END_HOUR):
            for minute in range(0, 60, minutes):
                slots.append((weekday, hour, minute))
        slots.append((weekday, SESSION_END_HOUR, SESSION_CLOSE_MINUTE))
    return slots


def schedule_command(sound: SoundOptions) -> list[str]:
    """Build the command the scheduled runs use, carrying the sound choice."""
    command = [sys.executable, str(Path(__file__).resolve())]
    if sound.preset != DEFAULT_SOUND:
        command += ["--sound", sound.preset]
    if sound.repeats != 1:
        command += ["--sound-repeats", str(sound.repeats)]
    if sound.sound_file:
        command += ["--sound-file", sound.sound_file]
    if sound.raise_volume:
        command.append("--raise-volume")
    return command


def build_schedule_plist(minutes: int, sound: SoundOptions) -> str:
    """Render the LaunchAgent plist that runs this monitor on a cadence.

    launchd's StartCalendarInterval takes an array of dictionaries, so each
    firing is listed explicitly rather than relying on hour or minute ranges.
    """
    command = schedule_command(sound)
    arguments = "\n".join(f"        <string>{item}</string>" for item in command)
    intervals = "\n".join(
        "        <dict>\n"
        "            <key>Weekday</key>\n"
        f"            <integer>{weekday}</integer>\n"
        "            <key>Hour</key>\n"
        f"            <integer>{hour}</integer>\n"
        "            <key>Minute</key>\n"
        f"            <integer>{minute}</integer>\n"
        "        </dict>"
        for weekday, hour, minute in schedule_slots(minutes)
    )
    return f"""<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>Label</key>
    <string>{LAUNCHD_LABEL}</string>
    <key>ProgramArguments</key>
    <array>
{arguments}
    </array>
    <key>WorkingDirectory</key>
    <string>{SCRIPT_DIR}</string>
    <key>StartCalendarInterval</key>
    <array>
{intervals}
    </array>
    <key>StandardOutPath</key>
    <string>{SCHEDULE_LOG_FILE}</string>
    <key>StandardErrorPath</key>
    <string>{SCHEDULE_LOG_FILE}</string>
    <key>RunAtLoad</key>
    <false/>
</dict>
</plist>
"""


def launchctl(*arguments: str) -> subprocess.CompletedProcess:
    """Run a launchctl subcommand in this user's GUI domain."""
    return subprocess.run(
        ["launchctl", *arguments],
        capture_output=True,
        text=True,
        check=False,
        timeout=REQUEST_TIMEOUT,
    )


def _schedule_target() -> str:
    """Return the launchd target string for this user's GUI domain."""
    return f"gui/{os.getuid()}/{LAUNCHD_LABEL}"


def install_schedule(minutes: int, sound: SoundOptions, dry_run: bool) -> int:
    """Write the LaunchAgent plist and load it, replacing any earlier version."""
    plist = build_schedule_plist(minutes, sound)
    command = " ".join(schedule_command(sound))

    if dry_run:
        print(f"Would write {LAUNCHD_PLIST_PATH}")
        print(f"Would bootout {_schedule_target()} (ignored if not loaded)")
        print(f"Would bootstrap gui/{os.getuid()} {LAUNCHD_PLIST_PATH}")
        print(f"Would log scheduled runs to {SCHEDULE_LOG_FILE}")
        print(f"Command: {command}")
        print(f"Firings: {len(schedule_slots(minutes))} per week")
        return 0

    LAUNCHD_PLIST_PATH.parent.mkdir(parents=True, exist_ok=True)
    SCHEDULE_LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
    LAUNCHD_PLIST_PATH.write_text(plist, encoding="utf-8")
    print(f"Wrote {LAUNCHD_PLIST_PATH}")

    # bootout first: bootstrap fails when the label is already loaded, and
    # booting out an unloaded label fails harmlessly, so this is idempotent.
    launchctl("bootout", _schedule_target())
    result = launchctl("bootstrap", f"gui/{os.getuid()}", str(LAUNCHD_PLIST_PATH))
    if result.returncode != 0:
        print(f"ERROR: launchctl bootstrap failed: {result.stderr.strip()}", file=sys.stderr)
        return 1

    sessions_per_day = len(schedule_slots(minutes)) // len(SCHEDULE_WEEKDAYS)
    print(
        f"Loaded. Runs this monitor {sessions_per_day} times per session, "
        f"every {minutes} minute(s) from "
        f"{SESSION_START_HOUR:02d}:00 to "
        f"{SESSION_END_HOUR:02d}:{SESSION_CLOSE_MINUTE:02d} ET on weekdays"
    )
    print(f"Plist:  {LAUNCHD_PLIST_PATH}")
    print(f"Log:    {SCHEDULE_LOG_FILE}")
    print(f"Status: python3 {Path(__file__).name} --schedule-status")
    return 0


def remove_schedule(dry_run: bool) -> int:
    """Unload the LaunchAgent and delete its plist."""
    if dry_run:
        print(f"Would bootout {_schedule_target()}")
        print(f"Would delete {LAUNCHD_PLIST_PATH}")
        return 0

    launchctl("bootout", _schedule_target())
    if LAUNCHD_PLIST_PATH.is_file():
        LAUNCHD_PLIST_PATH.unlink()
        print(f"Deleted {LAUNCHD_PLIST_PATH}")
    else:
        print(f"No plist at {LAUNCHD_PLIST_PATH}")
    print("Schedule removed.")
    return 0


def installed_slots() -> list[tuple[int, int, int]] | None:
    """Read the firings of the installed plist, or None when it can't be read.

    Status reports what launchd was actually handed, not what this script's
    constants would generate today, so a stale plist is visible as stale.
    """
    if not LAUNCHD_PLIST_PATH.is_file():
        return None
    try:
        parsed = plistlib.loads(LAUNCHD_PLIST_PATH.read_bytes())
    except (OSError, ValueError, plistlib.InvalidFileException):
        return None
    intervals = parsed.get("StartCalendarInterval")
    if not isinstance(intervals, list):
        return None

    slots: set[tuple[int, int, int]] = set()
    for interval in intervals:
        if not isinstance(interval, dict):
            continue
        weekday = interval.get("Weekday")
        hour = interval.get("Hour")
        minute = interval.get("Minute")
        if all(isinstance(value, int) for value in (weekday, hour, minute)):
            slots.add((weekday, hour, minute))
    return sorted(slots)


def describe_installed_slots(slots: list[tuple[int, int, int]]) -> str:
    """Summarize installed firings as a cadence a person can check at a glance."""
    per_weekday = {weekday: 0 for weekday, _, _ in slots}
    for weekday, _, _ in slots:
        per_weekday[weekday] += 1
    firings = min(per_weekday.values()) if per_weekday else 0

    one_day = sorted((hour, minute) for weekday, hour, minute in slots if weekday == 1)
    if not one_day:
        one_day = sorted((hour, minute) for _, hour, minute in slots)
    if not one_day:
        return "unreadable"

    hours = sorted({hour for hour, _ in one_day})
    gaps: list[int] = []
    for hour in hours:
        minutes = sorted(minute for candidate, minute in one_day if candidate == hour)
        gaps += [later - earlier for earlier, later in zip(minutes, minutes[1:])]
    step = min(gaps) if gaps else 60
    close = min(minute for hour, minute in one_day if hour == hours[-1])
    weekdays = sorted(per_weekday)
    if weekdays == list(range(weekdays[0], weekdays[0] + len(weekdays))):
        days = f"{WEEKDAY_NAMES[weekdays[0] - 1]}-{WEEKDAY_NAMES[weekdays[-1] - 1]}"
    else:
        days = ",".join(WEEKDAY_NAMES[weekday - 1] for weekday in weekdays)
    return (
        f"every {step} min, {days} "
        f"{hours[0]:02d}:00-{hours[-1]:02d}:{close:02d} ET "
        f"({firings} firings per session)"
    )


def next_firing(
    slots: list[tuple[int, int, int]], now: datetime | None = None
) -> datetime | None:
    """Return the first installed firing strictly after `now`, in market time."""
    if not slots:
        return None
    moment = (now or datetime.now(MARKET_TIMEZONE)).astimezone(MARKET_TIMEZONE)
    wanted = set(slots)
    probe = moment.replace(second=0, microsecond=0) + timedelta(minutes=1)
    for _ in range(60 * 24 * 8):
        if (probe.isoweekday(), probe.hour, probe.minute) in wanted:
            return probe
        probe += timedelta(minutes=1)
    return None


def _launchctl_field(output: str, field: str) -> str | None:
    """Return one `key = value` field from launchctl print output."""
    for line in output.splitlines():
        stripped = line.strip()
        if stripped.startswith(f"{field} ="):
            return stripped.split("=", 1)[1].strip()
    return None


def last_scheduled_run() -> datetime | None:
    """Return when the scheduled job last wrote output, in market time.

    The schedule log is the job's stdout, so only a run of the installed agent
    writes to it -- a manual run in the foreground does not.
    """
    if not SCHEDULE_LOG_FILE.is_file():
        return None
    return datetime.fromtimestamp(SCHEDULE_LOG_FILE.stat().st_mtime, MARKET_TIMEZONE)


def _field(label: str, value: str) -> None:
    """Print one aligned status field."""
    print(f"{label + ':':<15}{value}")


def schedule_status() -> int:
    """Report whether a run will fire next, when, and whether it has been working."""
    result = launchctl("print", _schedule_target())
    loaded = result.returncode == 0
    installed = LAUNCHD_PLIST_PATH.is_file()

    if loaded:
        _field("Scheduled", "yes")
    elif installed:
        _field("Scheduled", "no  (plist installed but not loaded, so it will not fire)")
        print(f"{'':<15}launchctl bootstrap gui/{os.getuid()} {LAUNCHD_PLIST_PATH}")
    else:
        _field("Scheduled", "no  (no plist installed, so nothing will fire)")

    executed = last_scheduled_run()
    _field(
        "Last executed",
        executed.strftime("%Y-%m-%d %H:%M:%S %Z") if executed else "never",
    )

    slots = installed_slots()
    if slots:
        _field("Cadence", describe_installed_slots(slots))
        upcoming = next_firing(slots) if loaded else None
        if upcoming:
            minutes_away = round(
                (upcoming - datetime.now(MARKET_TIMEZONE)).total_seconds() / 60
            )
            when = "now" if minutes_away <= 0 else f"in {minutes_away} min"
            _field("Next run", f"{upcoming.strftime('%a %d %b %H:%M %Z')} ({when})")

    if loaded:
        runs = _launchctl_field(result.stdout, "runs") or "?"
        exit_code = _launchctl_field(result.stdout, "last exit code") or "?"
        _field("Runs", f"{runs} (last exit code {exit_code})")
    else:
        print("Install one with:")
        print(f"  python3 {Path(__file__).name} --install-schedule 5")

    _field("Label", LAUNCHD_LABEL)
    _field("Plist", str(LAUNCHD_PLIST_PATH))
    _field("Log", str(SCHEDULE_LOG_FILE))
    return 0


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser."""
    supported = ", ".join(sorted(BOND_SERIES))
    sounds = ", ".join(SOUND_CHOICES)
    script = Path(__file__).name
    return argparse.ArgumentParser(
        description=(
            "Check configured Treasury maturity-series yields against their last "
            f"{LOOKBACK_TRADING_DAYS}-trading-day highs and raise a macOS sound "
            "and persistent popup when one or more are above them."
        ),
        epilog=(
            "Configuration in .env:\n"
            "  BOND_TYPES=2Y_NOTE,10Y_NOTE,20Y_BOND,30Y_BOND,10Y_TIPS\n\n"
            f"Supported BOND_TYPES values:\n  {supported}\n\n"
            f"Alert sounds:\n  {sounds}\n"
            f"  asset plays {BUNDLED_SOUND_FILE.name} (the default); the\n"
            "  others are synthesized. Use --sound-file to play your own audio\n"
            "  instead, --sound-repeats to repeat it, and --raise-volume to raise\n"
            "  the system output volume for the alert and restore it afterwards.\n\n"
            f"The window spans the last {LOOKBACK_TRADING_DAYS} trading days, the current\n"
            "session's intraday prints included. The high the live yield is compared\n"
            f"against is measured before the live print, so an alert means a new\n"
            f"{LOOKBACK_TRADING_DAYS}-trading-day high. The popup stays open until it is\n"
            "acknowledged. Run --dry-run to inspect the check without sound, popup,\n"
            "or log output.\n"
            "\n"
            "SCHEDULED RUNS (macOS LaunchAgent)\n"
            "  Watch the market on a cadence by installing a LaunchAgent for this\n"
            "  script instead of cron -- launchd runs a missed check as soon as the\n"
            "  Mac wakes, and a GUI-domain agent can raise the popup. The agent\n"
            f"  runs this monitor during the cash session only, {SESSION_START_HOUR:02d}:00 to\n"
            f"  the {SESSION_END_HOUR:02d}:{SESSION_CLOSE_MINUTE:02d} ET close, on weekdays.\n"
            "  Scheduled runs are silent unless a series qualifies; because the\n"
            "  comparison excludes the live print, a level that keeps rising can\n"
            "  alert again, but a level that merely holds will not.\n"
            "\n"
            f"  Install (cadence in minutes):\n"
            f"    {YELLOW}python3 {script} --install-schedule 5{RESET}\n"
            f"  Preview without changing anything:\n"
            f"    {YELLOW}python3 {script} --install-schedule 5 --dry-run{RESET}\n"
            f"  Check whether it will fire, when it next fires, and when it last ran:\n"
            f"    {YELLOW}python3 {script} --schedule-status{RESET}\n"
            f"  Remove it:\n"
            f"    {YELLOW}python3 {script} --remove-schedule{RESET}\n"
            "  Re-run --install-schedule to change the cadence; it regenerates the\n"
            "  plist and reloads it, so don't hand-edit the plist.\n"
            "\n"
            f"  Plist location: {LAUNCHD_PLIST_PATH}\n"
            f"  Log file:       {SCHEDULE_LOG_FILE}\n"
            f"  Label:          {LAUNCHD_LABEL}\n"
            "  Trigger a run right now (for testing):\n"
            f"    launchctl kickstart -p gui/$(id -u)/{LAUNCHD_LABEL}\n"
            f"  Tail the log:\n"
            f"    tail -f {SCHEDULE_LOG_FILE}\n"
            "  Disable temporarily (stays installed, won't fire):\n"
            f"    launchctl bootout gui/$(id -u)/{LAUNCHD_LABEL}\n"
            "  Re-enable after disabling:\n"
            f"    launchctl bootstrap gui/$(id -u) {LAUNCHD_PLIST_PATH}"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )


def _format_request_failure(error: Any) -> str:
    """Format a request failure without exposing the request URL."""
    response = getattr(error, "response", None)
    status_code = getattr(response, "status_code", None)
    status = f" (HTTP {status_code})" if status_code else ""
    return f"request failed ({type(error).__name__}){status}"


def _record_failure(failure: str, failures: list[str], dry_run: bool) -> None:
    """Record and print one isolated series failure."""
    failures.append(failure)
    print(colorize(f"Warning: {failure}", RED))
    if not dry_run:
        log(f"ERROR: {failure}")


def check_configured_series(
    bond_types: list[str],
    requests_module: Any,
    dry_run: bool,
) -> tuple[list[SeriesResult], list[str]]:
    """Fetch and evaluate every configured series, isolating failures."""
    results: list[SeriesResult] = []
    failures: list[str] = []

    for bond_type in bond_types:
        label, source = BOND_SERIES[bond_type]
        print(colorize(f"Checking {bond_type} ({source})...", CYAN))
        try:
            intraday, daily = fetch_source_series(source, requests_module)
            result = evaluate_series(
                bond_type,
                label,
                source,
                intraday,
                daily,
            )
        except requests_module.RequestException as error:
            failure = f"{bond_type} ({source}): {_format_request_failure(error)}"
            _record_failure(failure, failures, dry_run)
            continue
        except ValueError as error:
            failure = f"{bond_type} ({source}): {error}"
            _record_failure(failure, failures, dry_run)
            continue

        results.append(result)
        output = format_result(result)
        output_color = GREEN if result.qualifies else YELLOW
        print(colorize(output, output_color))
        if not dry_run:
            log(output)

    return results, failures


def report_results(
    results: list[SeriesResult],
    failures: list[str],
    dry_run: bool,
    sound: SoundOptions,
) -> int:
    """Report checked series and send a combined alert when needed."""
    if not results:
        print("ERROR: no configured series could be checked", file=sys.stderr)
        return 1

    qualifying = [result for result in results if result.qualifies]
    if qualifying:
        message = build_alert_message(qualifying)
        print(colorize("Qualifying series:", f"{BOLD}{CYAN}"))
        print(colorize(message, GREEN))
        try:
            send_macos_alert(message, dry_run, sound)
        except RuntimeError as error:
            if not dry_run:
                log(f"ERROR: {error}")
            print(f"ERROR: {error}", file=sys.stderr)
            return 1
        if not dry_run:
            log(f"ALERT: {len(qualifying)} series qualified")
    else:
        summary = (
            f"No configured series is above its {LOOKBACK_TRADING_DAYS}-trading-day high."
        )
        print(colorize(summary, YELLOW))
        if not dry_run:
            log(summary)

    if failures:
        print(f"Completed with {len(failures)} series failure(s).")
        return 1
    return 0


def main(argv: list[str] | None = None) -> int:
    """Parse options, load configuration, check series, and send alerts."""
    parser = build_parser()
    parser.add_argument(
        "-v",
        "--version",
        action="version",
        version=f"%(prog)s {VERSION}",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help=(
            "Check series without playing sound, showing a popup, or logging, "
            "and preview schedule changes without installing them"
        ),
    )
    parser.add_argument(
        "--sound",
        choices=SOUND_CHOICES,
        default=DEFAULT_SOUND,
        help=f"Alert sound to play (default: {DEFAULT_SOUND})",
    )
    parser.add_argument(
        "--sound-repeats",
        type=int,
        default=1,
        metavar="N",
        help="Play the alert sound N times (default: 1)",
    )
    parser.add_argument(
        "--sound-file",
        metavar="PATH",
        help="Play this audio file instead of a synthesized preset",
    )
    parser.add_argument(
        "--raise-volume",
        action="store_true",
        help="Raise the system output volume for the alert and restore it after",
    )
    schedule = parser.add_mutually_exclusive_group()
    schedule.add_argument(
        "--install-schedule",
        type=schedule_minutes_argument,
        metavar="MINUTES",
        help=(
            "Install the macOS LaunchAgent that runs this monitor every MINUTES "
            "during the cash session, replacing any schedule already installed"
        ),
    )
    schedule.add_argument(
        "--remove-schedule",
        action="store_true",
        help="Unload the installed LaunchAgent and delete its plist",
    )
    schedule.add_argument(
        "--schedule-status",
        action="store_true",
        help="Report whether the LaunchAgent is installed and loaded",
    )
    args = parser.parse_args(argv)

    sound = SoundOptions(
        preset=args.sound,
        repeats=args.sound_repeats,
        sound_file=args.sound_file,
        raise_volume=args.raise_volume,
    )

    if args.schedule_status:
        return schedule_status()
    if args.remove_schedule:
        return remove_schedule(args.dry_run)
    if args.install_schedule is not None:
        return install_schedule(args.install_schedule, sound, args.dry_run)

    try:
        import requests
        from dotenv import load_dotenv
    except ModuleNotFoundError as error:
        print(
            f"error: missing dependency {error.name}. "
            "Install the dependencies in requirements.txt.",
            file=sys.stderr,
        )
        return 1

    load_dotenv(str(ENV_FILE))

    try:
        bond_types = parse_bond_types(os.getenv("BOND_TYPES", ""))
    except ValueError as error:
        message = str(error)
        if not args.dry_run:
            log(f"ERROR: {message}")
        print(f"ERROR: {message}", file=sys.stderr)
        return 1

    results, failures = check_configured_series(bond_types, requests, args.dry_run)
    return report_results(results, failures, args.dry_run, sound)


if __name__ == "__main__":
    sys.exit(main())
