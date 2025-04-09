#!/bin/bash

# Full path to Python and script
PY="/opt/homebrew/bin/python3"
SCRIPT="/Users/asanchez/dev/financial-analytics/get_forwardpe.py"

# Kill all inherited env vars to break out of LibreOffice's Python
exec env -i "$PY" "$SCRIPT" "$1"
