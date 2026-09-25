#!/usr/bin/env python3
"""Unit tests for bond_yield_high_alert.py.

Path: tests/test_bond_yield_high_alert.py (relative to the repo root)

Covers the monitor's own logic with synthetic data, so it runs without network
access or API keys: the session-hours filter that drops Webull's overnight junk
bars, the lookback window assembled from intraday plus daily bars, the
strict ">" trigger, TIPS CUSIP discovery, and alert-sound resolution. The
live-source smoke test lives in tests/test_financial_scripts.py.

Usage:
    pytest tests/test_bond_yield_high_alert.py -v
"""

import os
import plistlib
import subprocess
import sys
import wave
from datetime import date, datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import pytest

BOND_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "bond-analytics"
)
sys.path.insert(0, BOND_DIR)

import bond_yield_high_alert as alert  # noqa: E402
from webull_bond_fetcher import YieldBar  # noqa: E402

MARKET_TZ = ZoneInfo("America/New_York")
SESSION_DAYS = 7


def make_bar(day: date, hour: int, minute: int, close: float, high: float | None = None) -> YieldBar:
    """Build an intraday bar at a given market-local time."""
    moment = datetime(day.year, day.month, day.day, hour, minute, tzinfo=MARKET_TZ)
    return YieldBar(
        timestamp=int(moment.timestamp()),
        close=close,
        open=close,
        high=close if high is None else high,
        low=close,
    )


def make_daily_bar(day: date, high: float, sentinel_close: float = 3.979) -> YieldBar:
    """Build a daily bar, mimicking Webull's sentinel close field."""
    moment = datetime(day.year, day.month, day.day, tzinfo=MARKET_TZ)
    return YieldBar(
        timestamp=int(moment.timestamp()),
        close=sentinel_close,
        open=high - 0.05,
        high=high,
        low=high - 0.08,
    )


def trading_days(count: int, end: date = date(2026, 9, 25)) -> list[date]:
    """Return `count` consecutive weekdays ending on `end`."""
    days: list[date] = []
    cursor = end
    while len(days) < count:
        if cursor.weekday() < 5:
            days.append(cursor)
        cursor -= timedelta(days=1)
    return list(reversed(days))


def build_series(values: dict[date, float]) -> list[YieldBar]:
    """Build one session of intraday bars per day, two prints per day."""
    bars: list[YieldBar] = []
    for day, value in values.items():
        bars.append(make_bar(day, 9, 30, value - 0.02))
        bars.append(make_bar(day, 14, 0, value))
    return bars


def test_session_bars_drops_overnight_junk():
    """Bars outside 07:00-17:00 ET are discarded as sentinel noise."""
    day = date(2026, 9, 25)
    kept = [make_bar(day, 2, 5, 3.979), make_bar(day, 0, 5, 3.979)]
    kept.append(make_bar(day, 11, 40, 4.881))
    kept.append(make_bar(day, 16, 55, 4.890))
    kept.append(make_bar(day, 17, 0, 4.895))
    after_close = [make_bar(day, 17, 5, 4.895), make_bar(day, 17, 40, 4.896)]

    kept_bars = alert.session_bars(kept + after_close)
    assert [bar.close for bar in kept_bars] == [4.881, 4.890, 4.895]


def test_session_bars_keeps_closing_print():
    """The 17:00 ET close is a genuine print and stays in the window."""
    day = date(2026, 9, 25)
    close_bar = make_bar(day, 17, 0, 4.943)
    assert alert.session_bars([close_bar]) == [close_bar]


def test_session_bars_keeps_early_tips_prints():
    """The 07:35 ET TIPS session start stays inside the window."""
    day = date(2026, 9, 25)
    early = make_bar(day, 7, 35, 2.845)
    assert alert.session_bars([early]) == [early]


def test_build_daily_highs_prefers_intraday_over_daily_bar():
    """A day covered intraday uses the 30-second maximum, not the daily summary."""
    days = trading_days(2)
    intraday = [make_bar(days[0], 9, 30, 4.70), make_bar(days[0], 14, 0, 4.76)]
    daily = [make_daily_bar(days[0], 4.755), make_daily_bar(days[1], 4.80)]

    highs = alert.build_daily_highs(intraday, daily)
    assert highs[days[0]] == pytest.approx(4.76)
    assert highs[days[1]] == pytest.approx(4.80)


def test_build_daily_highs_ignores_sentinel_close():
    """The daily bar's sentinel close never becomes a day's high."""
    days = trading_days(1)
    highs = alert.build_daily_highs([], [make_daily_bar(days[0], 4.943)])
    assert highs[days[0]] == pytest.approx(4.943)


def test_evaluate_series_window_spans_the_lookback():
    """The window is the last LOOKBACK_TRADING_DAYS sessions, and only those."""
    days = trading_days(SESSION_DAYS)
    values = {day: 4.0 + index * 0.1 for index, day in enumerate(days)}
    result = alert.evaluate_series("2Y_NOTE", "2-Year", "2Y", build_series(values), [])

    assert result.window == (days[0], days[-1])
    assert result.current_value == pytest.approx(values[days[-1]])
    assert result.high_value == pytest.approx(values[days[-1]] - 0.02)
    assert result.high_date == days[-1]


def test_evaluate_series_uses_daily_bars_for_older_days():
    """Sessions outside the intraday period come from the daily bars."""
    days = trading_days(SESSION_DAYS)
    intraday_days = days[-3:]
    older_highs = {days[0]: 5.10, days[1]: 5.05, days[2]: 5.01, days[3]: 4.90}
    intraday = build_series({day: 4.80 + index * 0.01 for index, day in enumerate(intraday_days)})
    daily = [make_daily_bar(day, high) for day, high in older_highs.items()]

    result = alert.evaluate_series("10Y_NOTE", "10-Year", "10Y", intraday, daily)
    assert result.window == (days[0], days[-1])
    assert result.high_value == pytest.approx(5.10)
    assert result.high_date == days[0]


def test_lookback_variable_is_the_single_source_of_truth(monkeypatch):
    """Changing LOOKBACK_TRADING_DAYS changes the window, the verdict, and labels."""
    days = trading_days(10)
    values = {day: 4.00 for day in days}
    values[days[5]] = 4.90
    values[days[-1]] = 4.10
    intraday = build_series(values)

    # A three-day window excludes the 4.90 spike, so today's print is a new high.
    monkeypatch.setattr(alert, "LOOKBACK_TRADING_DAYS", 3)
    narrow = alert.evaluate_series("2Y_NOTE", "2-Year", "2Y", intraday, [])
    assert narrow.window == (days[-3], days[-1])
    assert narrow.high_value == pytest.approx(4.08)
    assert narrow.qualifies is True
    assert "3D_high=4.080%" in alert.format_result(narrow)
    assert "new 3-trading-day high" in alert.build_alert_message([narrow])

    # A five-day window includes it, so the same print is nothing new.
    monkeypatch.setattr(alert, "LOOKBACK_TRADING_DAYS", 5)
    wide = alert.evaluate_series("2Y_NOTE", "2-Year", "2Y", intraday, [])
    assert wide.window == (days[-5], days[-1])
    assert wide.high_value == pytest.approx(4.90)
    assert wide.high_date == days[5]
    assert wide.qualifies is False
    assert "5D_high=4.900%" in alert.format_result(wide)


def test_lookback_variable_determines_the_required_history(monkeypatch):
    """A window wider than the available sessions is refused, not silently trimmed."""
    days = trading_days(4)
    intraday = build_series({day: 4.0 for day in days})

    monkeypatch.setattr(alert, "LOOKBACK_TRADING_DAYS", 6)
    with pytest.raises(ValueError, match="4 trading days of history; 6 are required"):
        alert.evaluate_series("2Y", "2-Year", "2Y", intraday, [])


def test_evaluate_series_triggers_only_above_high():
    """A level above the settled high alerts; a level equal to it does not."""
    days = trading_days(SESSION_DAYS)
    values = {day: 4.00 for day in days}
    values[days[-1]] = 4.10
    above = alert.evaluate_series("2Y", "2-Year", "2Y", build_series(values), [])
    assert above.current_value == pytest.approx(4.10)
    assert above.high_value == pytest.approx(4.08)
    assert above.qualifies is True

    values = {day: 4.10 for day in days}
    equal = alert.evaluate_series("2Y", "2-Year", "2Y", build_series(values), [])
    assert equal.current_value == pytest.approx(4.10)
    assert equal.high_value == pytest.approx(4.10)
    assert equal.qualifies is False


def test_evaluate_series_first_print_of_day_above_prior_high():
    """The day's only print still alerts when it clears the prior six sessions."""
    days = trading_days(SESSION_DAYS)
    intraday = []
    for day in days[:-1]:
        intraday.extend([make_bar(day, 9, 30, 4.50), make_bar(day, 14, 0, 4.60)])
    intraday.append(make_bar(days[-1], 9, 31, 4.99))

    result = alert.evaluate_series("30Y_BOND", "30-Year", "30Y", intraday, [])
    assert result.window == (days[0], days[-1])
    assert result.current_value == pytest.approx(4.99)
    assert result.high_value == pytest.approx(4.60)
    assert result.high_date == days[-2]
    assert result.qualifies is True


def test_evaluate_series_ignores_todays_earlier_high():
    """Today's earlier intraday print is the high until a later one exceeds it."""
    days = trading_days(SESSION_DAYS)
    intraday = []
    for day in days[:-1]:
        intraday.extend([make_bar(day, 9, 30, 4.50), make_bar(day, 14, 0, 4.60)])
    intraday.extend([make_bar(days[-1], 9, 30, 4.90), make_bar(days[-1], 11, 40, 4.70)])

    result = alert.evaluate_series("30Y_BOND", "30-Year", "30Y", intraday, [])
    assert result.current_value == pytest.approx(4.70)
    assert result.high_value == pytest.approx(4.90)
    assert result.high_date == days[-1]
    assert result.qualifies is False


def test_evaluate_series_requires_full_window():
    """Fewer sessions than the lookback is an error, not a short comparison."""
    days = trading_days(3)
    with pytest.raises(ValueError, match="trading days of history"):
        alert.evaluate_series("2Y", "2-Year", "2Y", build_series({day: 4.0 for day in days}), [])


def test_evaluate_series_requires_in_session_bars():
    """A series with only overnight prints cannot be evaluated."""
    days = trading_days(SESSION_DAYS)
    overnight = [make_bar(day, 2, 5, 3.979) for day in days]
    with pytest.raises(ValueError, match="no in-session bars"):
        alert.evaluate_series("2Y", "2-Year", "2Y", overnight, [])


def test_evaluate_series_reports_bar_timestamp_in_market_time():
    """The current value carries the bar's own market-local timestamp."""
    days = trading_days(SESSION_DAYS)
    intraday = []
    for day in days[:-1]:
        intraday.extend([make_bar(day, 9, 30, 4.00), make_bar(day, 14, 0, 4.00)])
    intraday.extend([make_bar(days[-1], 9, 30, 4.00), make_bar(days[-1], 11, 40, 4.12)])

    result = alert.evaluate_series("2Y", "2-Year", "2Y", intraday, [])
    assert result.current_value == pytest.approx(4.12)
    assert (result.current_time.hour, result.current_time.minute) == (11, 40)
    assert result.current_time.tzinfo is not None


class FakeResponse:
    """Minimal stand-in for a requests response."""

    def __init__(self, payload, status_code: int = 200):
        self.payload = payload
        self.status_code = status_code

    def raise_for_status(self):
        """Accept the response; the tests only exercise success paths."""

    def json(self):
        """Return the canned payload."""
        return self.payload


class FakeRequests:
    """Stand-in for the requests module recording the URL it was called with."""

    RequestException = OSError

    def __init__(self, payload):
        self.payload = payload
        self.calls: list[dict] = []

    def get(self, url, params=None, timeout=None):
        """Record the call and return the canned payload."""
        self.calls.append({"url": url, "params": params, "timeout": timeout})
        return FakeResponse(self.payload)


def test_discover_tips_cusips_returns_newest_first():
    """The current and previous on-the-run TIPS are returned newest first."""
    payload = [
        {"cusip": "91282CPU9", "issueDate": "2026-01-30", "maturityDate": "2036-01-15T00:00:00"},
        {"cusip": "91282CRE3", "issueDate": "2026-09-30", "maturityDate": "2036-07-15T00:00:00"},
        {"cusip": "91282CRE3", "issueDate": "2026-07-31", "maturityDate": "2036-07-15T00:00:00"},
        {"cusip": "912810US5", "issueDate": "2026-08-31", "maturityDate": "2056-02-15T00:00:00"},
        {"cusip": "91282CQP9", "issueDate": "2026-06-30", "maturityDate": "2031-04-15T00:00:00"},
        {"cusip": "garbage", "issueDate": "not-a-date", "maturityDate": "2036-07-15"},
    ]
    fake = FakeRequests(payload)

    assert alert.discover_tips_cusips(fake) == ["91282CRE3", "91282CPU9"]
    assert fake.calls[0]["params"] == {"format": "json", "type": "TIPS"}


def test_discover_tips_cusips_without_match_raises():
    """An empty result is an error rather than a silent skip."""
    with pytest.raises(ValueError, match="no 10-Year TIPS"):
        alert.discover_tips_cusips(FakeRequests([{"cusip": "1", "issueDate": "2026-01-01", "maturityDate": "2026-02-01"}]))


def test_parse_bond_types_normalizes_and_rejects_unknown():
    """Configured names are normalized and unknown ones are rejected."""
    assert alert.parse_bond_types(" 2y note, 10Y_NOTE ,20Y_BOND") == [
        "2Y_NOTE",
        "10Y_NOTE",
        "20Y_BOND",
    ]
    with pytest.raises(ValueError, match="Unknown BOND_TYPES"):
        alert.parse_bond_types("7Y_NOTE")


def test_format_result_includes_window_and_timestamp():
    """The terminal line carries the window, high date, and bar timestamp."""
    days = trading_days(SESSION_DAYS)
    values = {day: 4.00 for day in days}
    values[days[-1]] = 4.10
    result = alert.evaluate_series("2Y_NOTE", "2-Year Treasury Note", "2Y", build_series(values), [])

    line = alert.format_result(result)
    assert "term=2Y_NOTE" in line
    assert "current_yield=4.100%" in line
    assert f"{alert.LOOKBACK_TRADING_DAYS}D_high=4.080%" in line
    assert f"window={days[0].isoformat()}..{days[-1].isoformat()}" in line
    assert "[ALERT]" in line


def test_build_alert_message_lists_qualifying_series():
    """The popup message names each qualifying series and its high."""
    days = trading_days(SESSION_DAYS)
    values = {day: 4.00 for day in days}
    values[days[-1]] = 4.10
    result = alert.evaluate_series("2Y_NOTE", "2-Year Treasury Note", "2Y", build_series(values), [])

    message = alert.build_alert_message([result])
    assert "2-Year Treasury Note" in message
    assert "4.100%" in message
    assert f"previous high 4.080% on {days[-1].isoformat()}" in message


def test_sound_command_defaults_to_bundled_asset():
    """The default alert plays the bundled mp3 from the script's assets dir."""
    assert alert.BUNDLED_SOUND_FILE.is_file()
    command, description = alert.sound_command(alert.SoundOptions())
    assert command == ["/usr/bin/afplay", str(alert.BUNDLED_SOUND_FILE)]
    assert alert.BUNDLED_SOUND_FILE.name in description


def test_sound_command_none_is_silent():
    """The silent preset resolves to no command at all."""
    assert alert.sound_command(alert.SoundOptions(preset="none")) is None


def test_sound_command_missing_file_raises():
    """A missing --sound-file path fails loudly instead of falling back."""
    with pytest.raises(RuntimeError, match="sound file not found"):
        alert.sound_command(alert.SoundOptions(sound_file="/nonexistent/alert.mp3"))


def test_sound_command_custom_file_and_repeats():
    """A caller-supplied file overrides the preset and reports its repeats."""
    command, description = alert.sound_command(
        alert.SoundOptions(sound_file=str(alert.BUNDLED_SOUND_FILE), repeats=3)
    )
    assert command[0] == "/usr/bin/afplay"
    assert description.endswith("x3")


@pytest.mark.parametrize("preset", ["tone", "klaxon", "siren", "chime", "buzz"])
def test_synthesized_presets_render_playable_wave(preset, tmp_path, monkeypatch):
    """Each synthesized preset renders a full-scale, non-empty mono WAV."""
    monkeypatch.setattr(alert, "CACHE_DIR", tmp_path)
    path = alert.sound_file_for(preset)

    with wave.open(str(path), "rb") as handle:
        frames = handle.getnframes()
        assert handle.getnchannels() == 1
        assert handle.getsampwidth() == 2
        assert handle.getframerate() == alert.SAMPLE_RATE
        samples = handle.readframes(frames)

    assert frames > alert.SAMPLE_RATE
    peak = max(
        abs(int.from_bytes(samples[index : index + 2], "little", signed=True))
        for index in range(0, len(samples) - 1, 2)
    )
    assert peak == pytest.approx(alert.PEAK_AMPLITUDE * 32767, abs=2)


def test_applescript_string_escapes_quotes_and_newlines():
    """The popup message is escaped for AppleScript string literals."""
    escaped = alert.applescript_string('a "b"\nc\td')
    assert escaped == '"a \\"b\\"\\nc\\td"'


def stub_source(monkeypatch, series):
    """Replace the live fetcher with per-source (intraday, daily) fixtures."""
    def fake_fetch(source, requests_module):
        return series[source]

    monkeypatch.setattr(alert, "fetch_source_series", fake_fetch)


def test_main_dry_run_reports_alert_without_side_effects(monkeypatch, capsys, tmp_path):
    """A qualifying series prints the planned alert and touches nothing."""
    days = trading_days(SESSION_DAYS)
    values = {day: 4.00 for day in days}
    values[days[-1]] = 4.10
    stub_source(monkeypatch, {"2Y": (build_series(values), [])})
    monkeypatch.setenv("BOND_TYPES", "2Y_NOTE")
    monkeypatch.setattr(alert, "LOG_DIR", tmp_path / "logs")

    def forbidden(*args, **kwargs):
        raise AssertionError("dry run must not play sound or show a popup")

    monkeypatch.setattr(alert, "play_alert_sound", forbidden)
    monkeypatch.setattr(alert.subprocess, "run", forbidden)

    assert alert.main(["--dry-run"]) == 0
    out = capsys.readouterr().out
    assert "[ALERT]" in out
    assert f"would play {alert.BUNDLED_SOUND_FILE.name} x1" in out
    assert "new 7-trading-day high" in out
    assert not (tmp_path / "logs").exists()


def test_main_dry_run_reports_no_alert(monkeypatch, capsys):
    """A series below its settled high reports no alert and still exits zero."""
    days = trading_days(SESSION_DAYS)
    values = {day: 4.10 for day in days}
    values[days[-1]] = 4.05
    stub_source(monkeypatch, {"10Y": (build_series(values), [])})
    monkeypatch.setenv("BOND_TYPES", "10Y_NOTE")

    assert alert.main(["--dry-run"]) == 0
    out = capsys.readouterr().out
    assert "[no alert]" in out
    assert f"No configured series is above its {alert.LOOKBACK_TRADING_DAYS}-trading-day high." in out


def test_main_isolates_one_failing_series(monkeypatch, capsys):
    """A broken source is reported and the run still checks the others."""
    days = trading_days(SESSION_DAYS)
    values = {day: 4.00 for day in days}
    values[days[-1]] = 4.10

    def fake_fetch(source, requests_module):
        if source == "2Y":
            raise ValueError("2Y has 3 trading days of history; 7 are required")
        return build_series(values), []

    monkeypatch.setattr(alert, "fetch_source_series", fake_fetch)
    monkeypatch.setenv("BOND_TYPES", "2Y_NOTE,10Y_NOTE")

    assert alert.main(["--dry-run"]) == 1
    out = capsys.readouterr().out
    assert "Warning: 2Y_NOTE (2Y): 2Y has 3 trading days of history" in out
    assert "[ALERT]" in out
    assert "Completed with 1 series failure(s)." in out


def test_main_rejects_unknown_bond_types(monkeypatch, capsys):
    """An unsupported BOND_TYPES value fails before any network call."""
    monkeypatch.setenv("BOND_TYPES", "7Y_NOTE")
    assert alert.main(["--dry-run"]) == 1
    assert "Unknown BOND_TYPES value(s): 7Y_NOTE" in capsys.readouterr().err


def test_parse_schedule_minutes_accepts_cadences_in_range():
    """Whole numbers of minutes inside the supported range are accepted."""
    assert alert.parse_schedule_minutes("5") == 5
    assert alert.parse_schedule_minutes(" 60 ") == 60
    assert alert.parse_schedule_minutes(1) == 1


@pytest.mark.parametrize("raw_value", ["0", "61", "-5", "abc", "", "5m", "3.5"])
def test_parse_schedule_minutes_rejects_bad_input(raw_value):
    """Out-of-range and non-numeric cadences are rejected with a clear message."""
    with pytest.raises(ValueError, match="Schedule cadence"):
        alert.parse_schedule_minutes(raw_value)


def test_schedule_slots_cover_the_cash_session_on_weekdays():
    """Slots span the session on weekdays only, and include the closing print."""
    slots = alert.schedule_slots(5)

    assert {weekday for weekday, _, _ in slots} == {1, 2, 3, 4, 5}
    assert len(slots) == 5 * (10 * 12 + 1)
    assert (1, alert.SESSION_START_HOUR, 0) in slots
    assert (5, alert.SESSION_END_HOUR, alert.SESSION_CLOSE_MINUTE) in slots
    assert (1, alert.SESSION_START_HOUR - 1, 0) not in slots
    assert (1, alert.SESSION_END_HOUR, 5) not in slots


def test_schedule_slots_never_exceed_the_cadence():
    """Consecutive firings are at most the requested cadence apart."""
    for cadence in (1, 5, 15, 30, 60):
        per_day = [slot for slot in alert.schedule_slots(cadence) if slot[0] == 1]
        minutes = [hour * 60 + minute for _, hour, minute in per_day]
        gaps = [b - a for a, b in zip(minutes, minutes[1:])]
        assert max(gaps) <= cadence


def test_schedule_command_carries_sound_choices():
    """The scheduled command replays the sound options given at install time."""
    defaults = alert.schedule_command(alert.SoundOptions())
    assert defaults == [sys.executable, str(Path(alert.__file__).resolve())]

    configured = alert.schedule_command(
        alert.SoundOptions(preset="siren", repeats=3, sound_file="/tmp/a.mp3", raise_volume=True)
    )
    assert configured[2:] == [
        "--sound", "siren", "--sound-repeats", "3", "--sound-file", "/tmp/a.mp3", "--raise-volume",
    ]


def test_build_schedule_plist_is_valid_and_complete():
    """The rendered plist parses and carries every requested firing."""
    plist = alert.build_schedule_plist(5, alert.SoundOptions())
    parsed = plistlib.loads(plist.encode())

    assert parsed["Label"] == alert.LAUNCHD_LABEL
    assert parsed["WorkingDirectory"] == str(alert.SCRIPT_DIR)
    assert parsed["RunAtLoad"] is False
    assert parsed["StandardOutPath"] == str(alert.SCHEDULE_LOG_FILE)
    assert parsed["StandardErrorPath"] == str(alert.SCHEDULE_LOG_FILE)
    assert len(parsed["StartCalendarInterval"]) == len(alert.schedule_slots(5))
    assert parsed["StartCalendarInterval"][0] == {
        "Weekday": 1, "Hour": alert.SESSION_START_HOUR, "Minute": 0,
    }


def test_install_schedule_dry_run_changes_nothing(monkeypatch, capsys, tmp_path):
    """A dry-run install reports the plan without writing or calling launchctl."""
    plist_path = tmp_path / "LaunchAgents" / "agent.plist"
    monkeypatch.setattr(alert, "LAUNCHD_PLIST_PATH", plist_path)

    def forbidden(*args, **kwargs):
        raise AssertionError("dry run must not touch launchctl")

    monkeypatch.setattr(alert, "launchctl", forbidden)

    assert alert.main(["--install-schedule", "5", "--dry-run"]) == 0
    out = capsys.readouterr().out
    assert f"Would write {plist_path}" in out
    assert f"Firings: {len(alert.schedule_slots(5))} per week" in out
    assert not plist_path.exists()
    assert not (tmp_path / "LaunchAgents").exists()


def test_remove_schedule_dry_run_deletes_nothing(monkeypatch, capsys, tmp_path):
    """A dry-run remove leaves the installed plist in place."""
    plist_path = tmp_path / "agent.plist"
    plist_path.write_text("stub", encoding="utf-8")
    monkeypatch.setattr(alert, "LAUNCHD_PLIST_PATH", plist_path)

    def forbidden(*args, **kwargs):
        raise AssertionError("dry run must not touch launchctl")

    monkeypatch.setattr(alert, "launchctl", forbidden)

    assert alert.main(["--remove-schedule", "--dry-run"]) == 0
    assert f"Would delete {plist_path}" in capsys.readouterr().out
    assert plist_path.is_file()


def test_remove_schedule_bootouts_then_deletes(monkeypatch, tmp_path):
    """Removal unloads the job and deletes the plist it wrote."""
    plist_path = tmp_path / "agent.plist"
    plist_path.write_text("stub", encoding="utf-8")
    monkeypatch.setattr(alert, "LAUNCHD_PLIST_PATH", plist_path)
    calls: list[tuple[str, ...]] = []
    monkeypatch.setattr(alert, "launchctl", lambda *args: calls.append(args))

    assert alert.main(["--remove-schedule"]) == 0
    assert calls == [("bootout", alert._schedule_target())]
    assert not plist_path.exists()


def test_install_schedule_bootouts_before_bootstrapping(monkeypatch, tmp_path):
    """Installing is idempotent: the old job is unloaded before the new one loads."""
    plist_path = tmp_path / "LaunchAgents" / "agent.plist"
    log_file = tmp_path / "logs" / "schedule.log"
    monkeypatch.setattr(alert, "LAUNCHD_PLIST_PATH", plist_path)
    monkeypatch.setattr(alert, "SCHEDULE_LOG_FILE", log_file)
    calls: list[tuple[str, ...]] = []
    monkeypatch.setattr(
        alert,
        "launchctl",
        lambda *args: calls.append(args) or subprocess.CompletedProcess(args, 0, "", ""),
    )

    assert alert.main(["--install-schedule", "15", "--dry-run"]) == 0
    assert not plist_path.exists()
    assert calls == []

    assert alert.main(["--install-schedule", "15"]) == 0
    assert plist_path.is_file()
    assert len(plistlib.loads(plist_path.read_bytes())["StartCalendarInterval"]) == len(
        alert.schedule_slots(15)
    )
    assert calls[0] == ("bootout", alert._schedule_target())
    assert calls[1][:1] == ("bootstrap",)


def test_install_schedule_reports_bootstrap_failure(monkeypatch, tmp_path, capsys):
    """A refused bootstrap is an error, not a silently unscheduled monitor."""
    monkeypatch.setattr(alert, "LAUNCHD_PLIST_PATH", tmp_path / "agent.plist")
    monkeypatch.setattr(alert, "SCHEDULE_LOG_FILE", tmp_path / "schedule.log")
    monkeypatch.setattr(
        alert,
        "launchctl",
        lambda *args: subprocess.CompletedProcess(args, 1, "", "Load failed: 5: Input/output error"),
    )

    assert alert.main(["--install-schedule", "5"]) == 1
    assert "launchctl bootstrap failed" in capsys.readouterr().err


def _write_agent_plist(path: Path, minutes: int) -> None:
    """Write a LaunchAgent plist with the production cadence for `minutes`."""
    payload = {
        "Label": alert.LAUNCHD_LABEL,
        "StartCalendarInterval": [
            {"Weekday": weekday, "Hour": hour, "Minute": minute}
            for weekday, hour, minute in alert.schedule_slots(minutes)
        ],
    }
    path.write_bytes(plistlib.dumps(payload))


def test_schedule_status_reports_missing_schedule(monkeypatch, capsys, tmp_path):
    """Status explains how to install when nothing is loaded."""
    monkeypatch.setattr(alert, "LAUNCHD_PLIST_PATH", tmp_path / "agent.plist")
    monkeypatch.setattr(alert, "SCHEDULE_LOG_FILE", tmp_path / "schedule.log")
    monkeypatch.setattr(
        alert, "launchctl", lambda *args: subprocess.CompletedProcess(args, 113, "", "Could not find service")
    )

    assert alert.main(["--schedule-status"]) == 0
    out = capsys.readouterr().out
    assert "Scheduled:     no" in out
    assert "no plist installed" in out
    assert "Last executed: never" in out
    assert "--install-schedule 5" in out


def test_schedule_status_reports_cadence_and_last_execution(monkeypatch, capsys, tmp_path):
    """Status states the installed cadence and when the agent last ran."""
    plist_path = tmp_path / "agent.plist"
    _write_agent_plist(plist_path, 5)
    log_path = tmp_path / "schedule.log"
    log_path.write_text("run output\n", encoding="utf-8")
    ran_at = datetime(2026, 9, 25, 12, 40, 2, tzinfo=alert.MARKET_TIMEZONE)
    os.utime(log_path, (ran_at.timestamp(), ran_at.timestamp()))

    monkeypatch.setattr(alert, "LAUNCHD_PLIST_PATH", plist_path)
    monkeypatch.setattr(alert, "SCHEDULE_LOG_FILE", log_path)
    output = "\n".join(["\truns = 7", "\tlast exit code = 0"])
    monkeypatch.setattr(
        alert, "launchctl", lambda *args: subprocess.CompletedProcess(args, 0, output, "")
    )

    assert alert.main(["--schedule-status"]) == 0
    out = capsys.readouterr().out
    assert "Scheduled:     yes" in out
    assert "Last executed: 2026-09-25 12:40:02 EDT" in out
    assert "Cadence:       every 5 min, Mon-Fri 07:00-17:00 ET (121 firings per session)" in out
    assert "Next run:      " in out
    assert "Runs:          7 (last exit code 0)" in out
    assert "state" not in out


def test_schedule_status_flags_installed_but_not_loaded(monkeypatch, capsys, tmp_path):
    """An installed-but-unloaded plist is reported as not scheduled, with a fix."""
    plist_path = tmp_path / "agent.plist"
    _write_agent_plist(plist_path, 5)
    monkeypatch.setattr(alert, "LAUNCHD_PLIST_PATH", plist_path)
    monkeypatch.setattr(alert, "SCHEDULE_LOG_FILE", tmp_path / "schedule.log")
    monkeypatch.setattr(
        alert, "launchctl", lambda *args: subprocess.CompletedProcess(args, 113, "", "Could not find service")
    )

    assert alert.main(["--schedule-status"]) == 0
    out = capsys.readouterr().out
    assert "Scheduled:     no" in out
    assert "installed but not loaded" in out
    assert f"launchctl bootstrap gui/{os.getuid()}" in out
    assert "Next run:" not in out


def test_installed_slots_parses_plist_and_survives_garbage(monkeypatch, tmp_path):
    """Status reads firings from the plist and returns None when unreadable."""
    plist_path = tmp_path / "agent.plist"
    _write_agent_plist(plist_path, 5)
    monkeypatch.setattr(alert, "LAUNCHD_PLIST_PATH", plist_path)

    slots = alert.installed_slots()
    assert len(slots) == 121 * 5
    assert (1, 7, 0) in slots
    assert (5, 17, 0) in slots

    plist_path.write_text("not a plist", encoding="utf-8")
    assert alert.installed_slots() is None

    plist_path.unlink()
    assert alert.installed_slots() is None


def test_describe_installed_slots_reads_hourly_cadence():
    """A cadence with no repeated minute in an hour reads as hourly."""
    described = alert.describe_installed_slots(alert.schedule_slots(60))
    assert described == "every 60 min, Mon-Fri 07:00-17:00 ET (11 firings per session)"


def test_next_firing_skips_the_weekend():
    """After Friday's close the next firing is the following Monday."""
    friday_close = datetime(2026, 9, 25, 17, 5, tzinfo=alert.MARKET_TIMEZONE)
    monday_first = datetime(2026, 9, 28, 9, 0, tzinfo=alert.MARKET_TIMEZONE)
    slots = [(1, 9, 0), (1, 11, 0)]

    assert alert.next_firing(slots, now=friday_close) == monday_first
    assert alert.next_firing([], now=friday_close) is None
