#!/usr/bin/env python3
"""Generate and (re)install the macOS LaunchAgent that runs `bond_market_analyzer.py
--daily` every day at 10:00 local time.

Safe to re-run: it always regenerates the plist from these constants and reloads
it, so editing the schedule is just "change SCHEDULE_HOUR/SCHEDULE_MINUTE below,
then re-run this" -- do NOT hand-edit the installed plist directly, since the next
re-run of this script will silently overwrite it back to whatever's set here.

See bond_market_analyzer.py --help's LAUNCHD section for day-to-day management
commands (checking status, disabling, tailing the log, etc).
"""

import os
import subprocess
import sys

from bond_market_analyzer import LAUNCHD_LABEL, LAUNCHD_PLIST_PATH, DAILY_LOG_FILE, SCRIPT_DIR

# Change these and re-run this script to reschedule -- don't hand-edit the plist,
# see the module docstring above for why.
SCHEDULE_HOUR = 10
SCHEDULE_MINUTE = 0

PLIST_TEMPLATE = """<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>Label</key>
    <string>{label}</string>
    <key>ProgramArguments</key>
    <array>
        <string>{python}</string>
        <string>{script}</string>
        <string>--daily</string>
    </array>
    <key>WorkingDirectory</key>
    <string>{workdir}</string>
    <key>StartCalendarInterval</key>
    <dict>
        <key>Hour</key>
        <integer>{hour}</integer>
        <key>Minute</key>
        <integer>{minute}</integer>
    </dict>
    <key>StandardOutPath</key>
    <string>{log}</string>
    <key>StandardErrorPath</key>
    <string>{log}</string>
    <key>RunAtLoad</key>
    <false/>
</dict>
</plist>
"""


def main():
    script_path = os.path.join(SCRIPT_DIR, "bond_market_analyzer.py")
    os.makedirs(os.path.dirname(DAILY_LOG_FILE), exist_ok=True)
    os.makedirs(os.path.dirname(LAUNCHD_PLIST_PATH), exist_ok=True)

    plist = PLIST_TEMPLATE.format(
        label=LAUNCHD_LABEL,
        python=sys.executable,
        script=script_path,
        workdir=SCRIPT_DIR,
        hour=SCHEDULE_HOUR,
        minute=SCHEDULE_MINUTE,
        log=DAILY_LOG_FILE,
    )
    with open(LAUNCHD_PLIST_PATH, "w") as f:
        f.write(plist)
    print(f"Wrote {LAUNCHD_PLIST_PATH}")

    uid = os.getuid()
    # bootout first in case it's already loaded from a previous run of this script --
    # bootstrap fails if the label is already loaded, and bootout on an unloaded
    # label just fails harmlessly, so this ordering is idempotent either way.
    subprocess.run(["launchctl", "bootout", f"gui/{uid}/{LAUNCHD_LABEL}"], capture_output=True)
    result = subprocess.run(
        ["launchctl", "bootstrap", f"gui/{uid}", LAUNCHD_PLIST_PATH],
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        print(f"launchctl bootstrap failed: {result.stderr.strip()}")
        sys.exit(1)

    print(f"Loaded. Runs daily at {SCHEDULE_HOUR:02d}:{SCHEDULE_MINUTE:02d} -- fetching {script_path} --daily")
    print(f"Log file: {DAILY_LOG_FILE}")
    print(f"Check status:  launchctl list | grep {LAUNCHD_LABEL}")
    print(f"Test it now:   launchctl kickstart -p gui/{uid}/{LAUNCHD_LABEL}")


if __name__ == "__main__":
    main()
