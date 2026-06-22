"""Finalise the active ``CHANGES.md`` section and clean ``_version.py`` for a release.

Encapsulates the *Finalise* protocol shared by the development workflow
(`.claude/docs/sys-changelog.md`), used by the ``/release`` stage: turn the
topmost ``## X.Y.Z (in progress)`` section into a dated, released section and strip
the ``b0`` beta suffix from ``src/sctoolbox/_version.py``. The notebook metadata
version bump stays separate (``scripts/change_notebook_version.py``).

Examples
--------
    python3 scripts/finalize_release.py                 # release the in-progress version as-is
    python3 scripts/finalize_release.py --version 0.16.0  # deliberate bump at release
    python3 scripts/finalize_release.py --dry-run
"""

import argparse
import datetime
import re
import sys
from pathlib import Path

CHANGES_PATH = Path("CHANGES.md")
VERSION_PATH = Path("src/sctoolbox/_version.py")
IN_PROGRESS_RE = re.compile(r"^## (\d+\.\d+\.\d+)\s*\(in progress\)", re.IGNORECASE)
VERSION_ARG_RE = re.compile(r"^\d+\.\d+\.\d+$")


def find_in_progress(lines: list[str]) -> tuple[int, str]:
    """Locate the topmost ``## X.Y.Z (in progress)`` header.

    Parameters
    ----------
    lines : list[str]
        The changelog split into lines.

    Returns
    -------
    tuple[int, str]
        The header's line index and its ``"X.Y.Z"`` version string.

    Raises
    ------
    SystemExit
        If no in-progress section exists.
    """
    for i, line in enumerate(lines):
        match = IN_PROGRESS_RE.match(line)
        if match:
            return i, match.group(1)
    sys.exit("No '## X.Y.Z (in progress)' section found — nothing to finalise (manual entry may be needed).")


def main() -> None:
    """Parse arguments and finalise the active changelog section."""
    parser = argparse.ArgumentParser(description="Finalise the in-progress CHANGES.md section for a release.")
    parser.add_argument("--version", default=None, help="release version X.Y.Z (defaults to the in-progress one)")
    parser.add_argument("--dry-run", action="store_true", help="print the result, write nothing")
    args = parser.parse_args()

    if args.version is not None and not VERSION_ARG_RE.match(args.version):
        parser.error("--version must look like X.Y.Z")

    lines = CHANGES_PATH.read_text().splitlines(keepends=True)
    index, in_progress = find_in_progress(lines)
    release = args.version or in_progress

    today = datetime.date.today().strftime("%d-%m-%Y")
    lines[index] = f"## {release} ({today})\n"
    version_line = f'__version__ = "{release}"\n'

    if not args.dry_run:
        CHANGES_PATH.write_text("".join(lines))
        VERSION_PATH.write_text(version_line)

    prefix = "[dry-run] " if args.dry_run else ""
    print(f"{prefix}finalised ## {release} ({today})")
    print(f"{prefix}_version.py -> {release}")


if __name__ == "__main__":
    main()
