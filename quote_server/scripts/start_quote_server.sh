#!/bin/bash
#
# start_quote_server.sh - Run the quote server in the foreground.
#
# Usage:
#   start_quote_server.sh [run_quote_server.py options]
#   start_quote_server.sh --refresh-seconds 60
#   start_quote_server.sh -h | --help
#   start_quote_server.sh -v | --version
#
# Requires: the repo virtualenv (.venv) created per the README setup commands.

VERSION="1.0.0"

main() {
  local here repo python
  here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
  repo="$(cd "${here}/../.." && pwd)"
  python="${repo}/.venv/bin/python"

  case "${1:-}" in
    -v|--version) echo "start_quote_server.sh ${VERSION}"; return 0 ;;
    -h|--help) sed -n '2,10p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'; return 0 ;;
  esac

  if [[ ! -x "${python}" ]]; then
    echo "Missing ${python}. Run the setup commands in the README." >&2
    return 1
  fi
  exec "${python}" "${here}/run_quote_server.py" "$@"
}

main "$@"
