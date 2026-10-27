#!/usr/bin/env python3
"""install_calc_integration.py

Install (or remove) the LibreOffice Calc side of the quote server: the GETQUOTE()
cell function, the Quotes menu, the Refresh Prices toolbar button, and the refresh
on open.

Run it with LibreOffice quit. Safe to re-run.

Usage:
    install_calc_integration.py
    install_calc_integration.py --dry-run
    install_calc_integration.py --uninstall
    install_calc_integration.py --profile /path/to/LibreOffice/4/user
"""

import argparse
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path
from xml.sax.saxutils import escape

VERSION = "1.0.0"

REPO = Path(__file__).resolve().parents[2]
SOURCE = REPO / "quote_server" / "libreoffice"
CLIENT_NAME = "calc_quote_client.py"
MODULE_NAME = "QuoteServer"
EXTENSION_ID = "org.quoteserver.calcui"
DEFAULT_PROFILE = Path.home() / "Library" / "Application Support" / "LibreOffice" / "4" / "user"
UNOPKG = Path("/Applications/LibreOffice.app/Contents/MacOS/unopkg")

MODULE_HEADER = ('<?xml version="1.0" encoding="UTF-8"?>\n'
                 '<!DOCTYPE script:module PUBLIC "-//OpenOffice.org//DTD OfficeDocument 1.0//EN" "module.dtd">\n'
                 '<script:module xmlns:script="http://openoffice.org/2000/script" '
                 f'script:name="{MODULE_NAME}" script:language="StarBasic">')
MODULE_FOOTER = "</script:module>\n"
LIBRARY_ENTRY = f' <library:element library:name="{MODULE_NAME}"/>\n'


class InstallError(Exception):
    """A step failed in a way the user must fix."""


def libreoffice_running() -> bool:
    """Return True when a LibreOffice process is running."""
    return subprocess.run(["pgrep", "-x", "soffice"], capture_output=True).returncode == 0


def basic_module_text() -> str:
    """Return QuoteServer.bas wrapped as a LibreOffice Basic module file."""
    source = (SOURCE / "QuoteServer.bas").read_text()
    return MODULE_HEADER + escape(source, {'"': "&quot;"}) + MODULE_FOOTER


def standard_library(profile: Path) -> Path:
    """Return the profile's Standard Basic library directory."""
    library = profile / "basic" / "Standard"
    if not (library / "script.xlb").exists():
        raise InstallError(f"No Standard library at {library}. Start LibreOffice once, quit it, and retry.")
    return library


def link_client(profile: Path, dry_run: bool) -> None:
    """Link the Python client into the profile's Scripts/python folder."""
    target = profile / "Scripts" / "python" / CLIENT_NAME
    print(f"Link {target} -> {SOURCE / CLIENT_NAME}")
    if dry_run:
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.is_symlink() or target.exists():
        target.unlink()
    target.symlink_to(SOURCE / CLIENT_NAME)


def install_module(profile: Path, dry_run: bool) -> None:
    """Write the Basic module into the Standard library and list it."""
    library = standard_library(profile)
    print(f"Write {library / (MODULE_NAME + '.xba')} and list it in script.xlb")
    if dry_run:
        return
    (library / f"{MODULE_NAME}.xba").write_text(basic_module_text())
    listing = library / "script.xlb"
    text = listing.read_text()
    if f'library:name="{MODULE_NAME}"' not in text:
        listing.write_text(text.replace("</library:library>", LIBRARY_ENTRY + "</library:library>"))


def remove_module(profile: Path, dry_run: bool) -> None:
    """Delete the Basic module and its listing."""
    library = standard_library(profile)
    print(f"Remove {library / (MODULE_NAME + '.xba')} and its script.xlb entry")
    if dry_run:
        return
    (library / f"{MODULE_NAME}.xba").unlink(missing_ok=True)
    listing = library / "script.xlb"
    listing.write_text(listing.read_text().replace(LIBRARY_ENTRY, ""))


def build_extension(destination: Path) -> Path:
    """Zip quote_server/libreoffice/extension into an .oxt file."""
    oxt = destination / "quote_server_calc.oxt"
    root = SOURCE / "extension"
    with zipfile.ZipFile(oxt, "w", zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(root.rglob("*")):
            if path.is_file():
                archive.write(path, path.relative_to(root).as_posix())
    return oxt


def unopkg(profile_root: Path | None, *args: str) -> subprocess.CompletedProcess:
    """Run unopkg, against a specific profile when profile_root is given."""
    command = [str(UNOPKG), *args, "--suppress-license"] if args[0] == "add" else [str(UNOPKG), *args]
    if profile_root is not None:
        command.insert(1, f"-env:UserInstallation=file://{profile_root}")
    return subprocess.run(command, capture_output=True, text=True)


def install_extension(profile_root: Path | None, dry_run: bool) -> None:
    """Install the menu, toolbar button, and on-open binding."""
    print(f"Install extension {EXTENSION_ID} with unopkg")
    if dry_run:
        return
    unopkg(profile_root, "remove", EXTENSION_ID)
    with tempfile.TemporaryDirectory() as tmp:
        result = unopkg(profile_root, "add", str(build_extension(Path(tmp))))
    if result.returncode != 0:
        raise InstallError("unopkg failed: " + (result.stdout + result.stderr).strip())


def remove_extension(profile_root: Path | None, dry_run: bool) -> None:
    """Remove the menu, toolbar button, and on-open binding."""
    print(f"Remove extension {EXTENSION_ID} with unopkg")
    if not dry_run:
        unopkg(profile_root, "remove", EXTENSION_ID)


def main() -> int:
    """Parse arguments and install or remove the Calc integration."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("-v", "--version", action="version", version=f"%(prog)s {VERSION}")
    parser.add_argument("--profile", type=Path, default=DEFAULT_PROFILE,
                        help="LibreOffice user profile folder (default: %(default)s)")
    parser.add_argument("--dry-run", action="store_true", help="print the steps; change nothing")
    parser.add_argument("--uninstall", action="store_true", help="remove the Calc integration")
    args = parser.parse_args()

    custom_profile = args.profile != DEFAULT_PROFILE
    if not custom_profile and not args.dry_run and libreoffice_running():
        print("LibreOffice is running. Quit it (Cmd+Q), then run this again.", file=sys.stderr)
        return 1
    profile_root = args.profile.parent if custom_profile else None
    try:
        if args.uninstall:
            remove_extension(profile_root, args.dry_run)
            remove_module(args.profile, args.dry_run)
            target = args.profile / "Scripts" / "python" / CLIENT_NAME
            print(f"Remove {target}")
            if not args.dry_run:
                target.unlink(missing_ok=True)
        else:
            link_client(args.profile, args.dry_run)
            install_module(args.profile, args.dry_run)
            install_extension(profile_root, args.dry_run)
    except InstallError as exc:
        print(str(exc), file=sys.stderr)
        return 1
    print("Done. Start LibreOffice." if not args.dry_run else "Dry run: nothing changed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
