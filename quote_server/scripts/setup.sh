#!/bin/bash
#
# setup.sh - Create the repo virtualenv and install the quote server's dependencies.
#
# Usage:
#   setup.sh          Run from any directory. Safe to re-run.
#   setup.sh -h | --help
#   setup.sh -v | --version
#
# Requires: Python 3.10 or later on PATH as python3.
# Creates <repo>/.venv, installs <repo>/requirements.txt, and runs pip install -e <repo>.

VERSION="1.0.0"

main() {
  local here repo
  here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
  repo="$(cd "${here}/../.." && pwd)"

  case "${1:-}" in
    -v|--version) echo "setup.sh ${VERSION}"; return 0 ;;
    -h|--help) sed -n '2,10p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'; return 0 ;;
  esac

  if ! python3 -c 'import sys; sys.exit(sys.version_info < (3, 10))' 2>/dev/null; then
    echo "Python 3.10 or later is required as python3." >&2
    return 1
  fi

  if [[ ! -x "${repo}/.venv/bin/python" ]]; then
    echo "Creating ${repo}/.venv"
    python3 -m venv "${repo}/.venv" || return 1
  fi

  echo "Installing dependencies"
  "${repo}/.venv/bin/pip" install --quiet --disable-pip-version-check \
    -r "${repo}/requirements.txt" || return 1
  "${repo}/.venv/bin/pip" install --quiet --disable-pip-version-check -e "${repo}" || return 1

  if ! "${repo}/.venv/bin/python" -c 'import core, quote_server'; then
    echo "Install finished but 'core' and 'quote_server' do not import." >&2
    return 1
  fi
  echo "Ready. Start the server with: ${here}/start_quote_server.sh"
}

main "$@"
