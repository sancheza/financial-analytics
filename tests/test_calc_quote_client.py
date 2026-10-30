"""Tests for the Calc client: lookups, error mapping, the offline breaker, and the installer."""

import importlib.util
import json
import threading
import time
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

import pytest

from quote_server.libreoffice import calc_quote_client as client

INSTALLER_PATH = Path(__file__).resolve().parents[1] / "quote_server" / "scripts" / "install_calc_integration.py"
spec = importlib.util.spec_from_file_location("install_calc_integration", INSTALLER_PATH)
installer = importlib.util.module_from_spec(spec)
spec.loader.exec_module(installer)


class FakeServer:
    """Local HTTP server returning canned (status, body, delay) per request path+query."""

    def __init__(self):
        self.responses = {}
        self.requests = []
        outer = self

        class Handler(BaseHTTPRequestHandler):
            def _answer(self):
                outer.requests.append((self.command, self.path))
                canned = (outer.responses.get(self.path.split("&")[0])
                          or outer.responses.get(self.path.split("?")[0]) or (500, {}, 0))
                status, body, delay = canned
                time.sleep(delay)
                payload = json.dumps(body).encode()
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)

            do_GET = do_POST = _answer

            def log_message(self, *args):
                pass

        self.httpd = HTTPServer(("127.0.0.1", 0), Handler)
        self.url = f"http://127.0.0.1:{self.httpd.server_address[1]}"
        threading.Thread(target=self.httpd.serve_forever, kwargs={"poll_interval": 0.01}, daemon=True).start()

    def close(self):
        self.httpd.shutdown()
        self.httpd.server_close()


@pytest.fixture
def server(monkeypatch):
    fake = FakeServer()
    monkeypatch.setattr(client, "SERVER_URL", fake.url)
    client._unblock()
    yield fake
    fake.close()
    client._unblock()


def quote_body(value):
    return {"symbol": "X", "value": value}


def test_returns_float(server):
    server.responses["/quote?symbol=AAPL"] = (200, quote_body(343.91), 0)
    assert client.getquote("aapl ") == 343.91


def test_sends_field_lowercased(server):
    server.responses["/quote?symbol=912810UF3"] = (200, quote_body(5.3), 0)
    assert client.getquote("912810UF3", "YTW") == 5.3
    assert server.requests[-1][1] == "/quote?symbol=912810UF3&field=ytw"


def test_blank_symbol_is_empty_and_makes_no_request(server):
    assert client.getquote("") == ""
    assert client.getquote(None) == ""
    assert client.getquote("  ") == ""
    assert server.requests == []


@pytest.mark.parametrize("status,code", [
    (202, "PENDING"), (404, "NOT_FOUND"), (400, "BAD_SYMBOL"), (400, "NEED_TICKER_OR_CUSIP"), (400, "AMBIGUOUS_SYMBOL"),
    (503, "STALE"), (503, "UPSTREAM"),
])
def test_server_error_codes_become_err_strings(server, status, code):
    server.responses["/quote?symbol=X"] = (status, {"error": {"code": code, "message": "the cause"}}, 0)
    assert client.getquote("X") == f"#ERR: {code} - the cause"


def test_non_numeric_value_names_the_field(server):
    server.responses["/quote?symbol=X"] = (200, quote_body(None), 0)
    assert client.getquote("X", "ytw") == "#ERR: NO_VALUE - the quote server returned no number for ytw"


def test_error_without_message_still_explains(server):
    server.responses["/quote?symbol=X"] = (503, {"error": {"code": "UPSTREAM"}}, 0)
    assert client.getquote("X") == "#ERR: UPSTREAM - the quote server reported an error"


def test_garbage_response_is_bad_response(server):
    server.httpd.RequestHandlerClass.do_GET = lambda self: (
        self.send_response(200), self.send_header("Content-Length", "3"), self.end_headers(),
        self.wfile.write(b"not"))
    result = client.getquote("X")
    assert result.startswith("#ERR: BAD_RESPONSE - ") and "port" in result


def test_offline_and_breaker(monkeypatch):
    monkeypatch.setattr(client, "SERVER_URL", "http://127.0.0.1:1")
    client._unblock()
    first = client.getquote("AAPL")
    assert first.startswith("#ERR: OFFLINE - ") and "http://127.0.0.1:1" in first and "start it" in first
    started = time.monotonic()
    assert client.getquote("MSFT") == first   # answered by the breaker
    assert time.monotonic() - started < 0.05
    client._unblock()


def test_timeout_and_breaker(server, monkeypatch):
    monkeypatch.setattr(client, "TIMEOUT_SEC", 0.1)
    server.responses["/quote?symbol=SLOW"] = (200, quote_body(1.0), 0.5)
    started = time.monotonic()
    timed_out = client.getquote("SLOW")
    assert timed_out.startswith("#ERR: TIMEOUT - ") and "0.1 s" in timed_out
    assert time.monotonic() - started < 0.4
    server.requests.clear()
    assert client.getquote("SLOW").startswith("#ERR: OFFLINE - ")
    assert server.requests == []


def test_breaker_expires(server, monkeypatch):
    monkeypatch.setattr(client, "OFFLINE_RETRY_SECONDS", 0.05)
    client._block()
    assert client.getquote("X").startswith("#ERR: OFFLINE - ")
    time.sleep(0.06)
    server.responses["/quote?symbol=X"] = (200, quote_body(2.5), 0)
    assert client.getquote("X") == 2.5


class FakeCell:
    def __init__(self, formula):
        self.Formula = formula


def test_recalculate_reenters_every_formula():
    class Recording(FakeCell):
        sets = 0

        def __setattr__(self, name, value):
            if name == "Formula" and hasattr(self, "Formula"):
                type(self).sets += 1
            super().__setattr__(name, value)

    cells = [Recording("=GETQUOTE(A1)"), Recording("=GETQUOTE(A2;B$1)")]
    client._recalculate(cells)
    assert Recording.sets == 2


def test_refresh_server_reports_unreachable(monkeypatch):
    monkeypatch.setattr(client, "SERVER_URL", "http://127.0.0.1:1")
    assert client._refresh_server(0.1) is False


def test_refresh_server_posts(server):
    server.responses["/refresh-all"] = (200, {"symbols": 1, "complete": True}, 0)
    assert client._refresh_server(2) is True
    assert server.requests[-1] == ("POST", "/refresh-all?wait=2")
    server.responses["/refresh-all"] = (200, {"symbols": 1, "complete": False}, 0)
    assert client._refresh_server(2) is True   # slow refresh still counts: the cached prices are shown


def test_refresh_server_rejects_error_reply(server):
    server.responses["/refresh-all"] = (404, {"error": {"code": "HTTP_ERROR", "message": "Not Found"}}, 0)
    assert client._refresh_server(2) is False


def test_refresh_document_without_quote_cells_does_nothing(server, monkeypatch):
    monkeypatch.setattr(client, "_quote_cells", lambda doc: [])
    assert client.refresh_document(object()) == 0
    assert server.requests == []


def test_refresh_document_recalculates_twice_around_server_refresh(server, monkeypatch):
    server.responses["/refresh-all"] = (200, {"symbols": 1, "complete": True}, 0)
    events = []
    monkeypatch.setattr(client, "_quote_cells", lambda doc: [FakeCell("=GETQUOTE(A1)")])
    monkeypatch.setattr(client, "_recalculate", lambda cells: events.append("recalc"))
    monkeypatch.setattr(client, "_refresh_server", lambda wait: events.append("server") or True)
    assert client.refresh_document(object()) == 1
    assert events == ["recalc", "server", "recalc"]


def test_refresh_document_skips_server_when_offline(monkeypatch):
    events = []
    monkeypatch.setattr(client, "_quote_cells", lambda doc: [FakeCell("=GETQUOTE(A1)")])
    monkeypatch.setattr(client, "_recalculate", lambda cells: (events.append("recalc"), client._block()))
    monkeypatch.setattr(client, "_refresh_server", lambda wait: events.append("server") or True)
    client.refresh_document(object())
    assert events == ["recalc"]
    client._unblock()


# --- installer -----------------------------------------------------------------------------------

@pytest.fixture
def profile(tmp_path):
    library = tmp_path / "user" / "basic" / "Standard"
    library.mkdir(parents=True)
    (library / "script.xlb").write_text(
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        '<library:library xmlns:library="http://openoffice.org/2000/library" library:name="Standard">\n'
        ' <library:element library:name="Module1"/>\n'
        '</library:library>\n')
    return tmp_path / "user"


def test_install_is_idempotent_and_uninstall_restores(profile):
    for _ in range(2):
        installer.link_client(profile, dry_run=False)
        installer.install_module(profile, dry_run=False)
    library = profile / "basic" / "Standard"
    assert (profile / "Scripts" / "python" / "calc_quote_client.py").is_symlink()
    assert (profile / "Scripts" / "python" / "calc_quote_client.py").resolve() == installer.SOURCE / "calc_quote_client.py"
    assert (library / "script.xlb").read_text().count('library:name="QuoteServer"') == 1
    module = (library / "QuoteServer.xba").read_text()
    assert 'script:name="QuoteServer"' in module and "Function GETQUOTE" in module
    assert "&quot;" in module and '"price"' not in module   # quotes are escaped for XML
    installer.remove_module(profile, dry_run=False)
    assert not (library / "QuoteServer.xba").exists()
    assert 'QuoteServer' not in (library / "script.xlb").read_text()
    assert 'Module1' in (library / "script.xlb").read_text()


def test_dry_run_changes_nothing(profile):
    installer.link_client(profile, dry_run=True)
    installer.install_module(profile, dry_run=True)
    assert not (profile / "Scripts").exists()
    assert not (profile / "basic" / "Standard" / "QuoteServer.xba").exists()


def test_missing_standard_library_is_reported(tmp_path):
    with pytest.raises(installer.InstallError):
        installer.install_module(tmp_path, dry_run=False)


def test_extension_archive_contains_manifest_and_config(tmp_path):
    import zipfile
    oxt = installer.build_extension(tmp_path)
    names = set(zipfile.ZipFile(oxt).namelist())
    assert {"META-INF/manifest.xml", "Addons.xcu", "Events.xcu", "description.xml"} <= names


def test_installed_urls_point_at_the_client_file():
    extension = installer.SOURCE / "extension"
    for name in ("Addons.xcu", "Events.xcu"):
        assert "calc_quote_client.py$" in (extension / name).read_text()
