"""In-process watchdog: notices when the refresh has stopped or hung, and restarts the server.

The LaunchAgent restarts a server that exits (`KeepAlive`). The watchdog makes a server
that is running but not refreshing exit, so the LaunchAgent starts a new one.

Log events: logger `quote_server.watchdog`, scope `quote_server`, action `watchdog_check`.
"""

import logging
import os
import subprocess
import sys
import threading
import time

log = logging.getLogger("quote_server.watchdog")

SCOPE = "quote_server"
ACTION_WATCHDOG_CHECK = "watchdog_check"

CHECK_SECONDS = 30.0       # time between checks
CONFIRM_SECONDS = 120.0    # a problem must last this long before the server restarts
GRACE_SECONDS = 60.0       # added to 3 refresh intervals before a refresh counts as overdue
EXIT_CODE = 70             # EX_SOFTWARE; the LaunchAgent restarts the server whatever the code
SAVE_TIMEOUT_SECONDS = 5.0
NOTIFY_TIMEOUT_SECONDS = 5.0


def notify_macos(title: str, message: str) -> bool:
    """Show a macOS notification.

    Returns:
        True when osascript accepted the notification; False elsewhere or on failure.
    """
    if sys.platform != "darwin":
        return False
    script = ["on run argv", "display notification (item 1 of argv) with title (item 2 of argv)",
              "end run"]
    command = ["osascript", *(arg for line in script for arg in ("-e", line)), message, title]
    try:
        result = subprocess.run(command, capture_output=True, timeout=NOTIFY_TIMEOUT_SECONDS)
    except (OSError, subprocess.SubprocessError) as exc:
        log.warning("notification not shown: %s", type(exc).__name__,
                    extra={"scope": SCOPE, "action": ACTION_WATCHDOG_CHECK,
                           "outcome": "failure", "actor": "system"})
        return False
    return result.returncode == 0


def _hard_exit(code: int) -> None:
    os._exit(code)


class Watchdog:
    """Checks the refresher on a timer; notifies, and optionally restarts the server.

    Args:
        refresher: The Refresher to watch (needs `alive` and `last_cycle_at`).
        interval: The refresher's full-refresh interval in seconds.
        restart: When True, a confirmed problem makes the process exit. When False the
            problem is logged and notified once, and the server keeps running.
        save: Called before exit to write the cache; it gets SAVE_TIMEOUT_SECONDS.
        notify: Shows a notification, `notify(title, message)`.
        exit_process: Ends the process, `exit_process(code)`.
        clock: Returns the current time in seconds (wall clock).
    """

    def __init__(self, refresher, interval: float, restart: bool = False, save=None,
                 notify=notify_macos, exit_process=_hard_exit, clock=time.time) -> None:
        self._refresher = refresher
        self._interval = interval
        self._restart = restart
        self._save = save
        self._notify = notify
        self._exit = exit_process
        self._clock = clock
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, name="watchdog", daemon=True)
        self.started_at = clock()
        self._problem_since: float | None = None
        self._notified = False

    @property
    def restart(self) -> bool:
        """True when a confirmed problem ends the process."""
        return self._restart

    def start(self) -> None:
        """Start checking."""
        self.started_at = self._clock()
        self._thread.start()

    def stop(self) -> None:
        """Stop checking and wait for the thread."""
        self._stop.set()
        self._thread.join(timeout=CHECK_SECONDS)

    def problem(self, now: float) -> str | None:
        """Return the reason the refresh is not working, or None when it is."""
        if not self._refresher.alive:
            return "the refresh thread has stopped"
        last = max(self._refresher.last_cycle_at or 0.0, self.started_at)
        limit = 3 * self._interval + GRACE_SECONDS
        if now - last > limit:
            return f"no full refresh for {int((now - last) / 60)} min (one is due every {self._interval:g} s)"
        return None

    def check(self) -> None:
        """Run one check; act when a problem has lasted CONFIRM_SECONDS."""
        now = self._clock()
        reason = self.problem(now)
        if reason is None:
            if self._problem_since is not None:
                log.info("refresh is working again",
                         extra={"scope": SCOPE, "action": ACTION_WATCHDOG_CHECK,
                                "outcome": "success", "actor": "system"})
            self._problem_since, self._notified = None, False
            return
        if self._problem_since is None:
            self._problem_since = now
            log.warning("%s; restarting if it lasts %d s", reason, CONFIRM_SECONDS,
                        extra={"scope": SCOPE, "action": ACTION_WATCHDOG_CHECK,
                               "outcome": "failure", "actor": "system"})
            return
        if now - self._problem_since < CONFIRM_SECONDS:
            return
        if self._restart:
            self._restart_server(reason)
        elif not self._notified:
            self._notified = True
            log.error("%s; the server was started without --restart-on-hang, so it keeps running", reason,
                      extra={"scope": SCOPE, "action": ACTION_WATCHDOG_CHECK,
                             "outcome": "failure", "actor": "system"})
            self._notify("Quote server", f"Not refreshing: {reason}. Restart it.")

    def _restart_server(self, reason: str) -> None:
        log.error("%s; saving the cache and exiting so the LaunchAgent restarts the server", reason,
                  extra={"scope": SCOPE, "action": ACTION_WATCHDOG_CHECK,
                         "outcome": "failure", "actor": "system"})
        self._notify("Quote server", f"Restarting: {reason}.")
        if self._save is not None:
            saver = threading.Thread(target=self._save, name="watchdog-save", daemon=True)
            saver.start()
            saver.join(SAVE_TIMEOUT_SECONDS)
        self._exit(EXIT_CODE)

    def _run(self) -> None:
        while not self._stop.wait(CHECK_SECONDS):
            try:
                self.check()
            except Exception:  # isolation point: a failed check must not end the watchdog
                log.error("watchdog check crashed", exc_info=True,
                          extra={"scope": SCOPE, "action": ACTION_WATCHDOG_CHECK,
                                 "outcome": "failure", "actor": "system"})
