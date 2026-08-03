"""Append a bullet to the active ``CHANGES.md`` section and keep ``_version.py`` in sync.

Encapsulates the *Append* protocol shared by the development workflow
(`.claude/docs/sys-changelog.md`): locate the topmost ``## X.Y.Z (in progress)``
section, insert a bullet in the right place, and ensure ``src/sctoolbox/_version.py``
carries that version with the ``b0`` beta suffix. When no in-progress section
exists (the fresh state right after a release), create one by bumping the latest
released version (patch by default).

Examples
--------
    python3 scripts/add_change.py "fix off-by-one in qc filter" --scope package --issue 451
    python3 scripts/add_change.py "clarify clustering notebook" --scope notebook
    python3 scripts/add_change.py "minor: add new public helper" --scope package --minor --dry-run
"""

import argparse
import re
import sys
from pathlib import Path

CHANGES_PATH = Path("CHANGES.md")
VERSION_PATH = Path("src/sctoolbox/_version.py")
NOTEBOOK_HEADER = "### Changes to notebooks"
VERSION_RE = re.compile(r"^## (\d+)\.(\d+)\.(\d+)")
IN_PROGRESS_RE = re.compile(r"^## (\d+)\.(\d+)\.(\d+)\s*\(in progress\)", re.IGNORECASE)


def find_active_section(lines: list[str]) -> tuple[int, int] | None:
    """Locate the topmost ``## ... (in progress)`` section.

    Parameters
    ----------
    lines : list[str]
        The changelog split into lines (each retaining its trailing newline).

    Returns
    -------
    tuple[int, int] | None
        ``(start, end)`` line indices of the active section — ``start`` is the
        header line, ``end`` is the line index of the next ``## `` header (or
        ``len(lines)`` if none follows). ``None`` if no in-progress section
        exists.
    """
    start = -1
    for i, line in enumerate(lines):
        if IN_PROGRESS_RE.match(line):
            start = i
            break
    if start == -1:
        return None

    end = len(lines)
    for i in range(start + 1, len(lines)):
        if lines[i].startswith("## "):
            end = i
            break
    return start, end


def bump_version(latest: tuple[int, int, int], level: str) -> tuple[int, int, int]:
    """Compute the next version number from the latest released one.

    Parameters
    ----------
    latest : tuple[int, int, int]
        The ``(major, minor, patch)`` of the latest released version.
    level : str
        One of ``"major"``, ``"minor"`` or ``"patch"``.

    Returns
    -------
    tuple[int, int, int]
        The bumped ``(major, minor, patch)``.
    """
    major, minor, patch = latest
    if level == "major":
        return major + 1, 0, 0
    if level == "minor":
        return major, minor + 1, 0
    return major, minor, patch + 1


def latest_released(lines: list[str]) -> tuple[int, int, int]:
    """Read the latest released (dated) version from the changelog.

    Parameters
    ----------
    lines : list[str]
        The changelog split into lines.

    Returns
    -------
    tuple[int, int, int]
        The ``(major, minor, patch)`` of the topmost ``## X.Y.Z`` header.

    Raises
    ------
    ValueError
        If no ``## X.Y.Z`` header is found.
    """
    for line in lines:
        match = VERSION_RE.match(line)
        if match:
            return int(match.group(1)), int(match.group(2)), int(match.group(3))
    raise ValueError("No '## X.Y.Z' version header found in CHANGES.md.")


def render_bullet(description: str, issue: int | None) -> str:
    """Build a changelog bullet line.

    Parameters
    ----------
    description : str
        The imperative change description.
    issue : int | None
        An optional GitLab issue number to reference.

    Returns
    -------
    str
        The bullet line, terminated with a newline.
    """
    suffix = f" (#{issue})" if issue is not None else ""
    return f"- {description}{suffix}\n"


def insert_into_section(lines: list[str], bounds: tuple[int, int], scope: str, bullet: str) -> list[str]:
    """Insert a bullet into an existing active section.

    Parameters
    ----------
    lines : list[str]
        The changelog split into lines.
    bounds : tuple[int, int]
        The ``(start, end)`` indices of the active section.
    scope : str
        One of ``"package"``, ``"docs"`` or ``"notebook"`` — selects the bullet's
        location within the section.
    bullet : str
        The rendered bullet line to insert.

    Returns
    -------
    list[str]
        A new list of lines with the bullet inserted.
    """
    start, end = bounds
    nb_index = next((i for i in range(start, end) if lines[i].strip() == NOTEBOOK_HEADER), -1)

    if scope == "notebook":
        if nb_index == -1:
            block = ["\n", NOTEBOOK_HEADER + "\n", bullet]
            at = _trim_trailing_blanks(lines, start, end)
            return lines[:at] + block + lines[at:]
        at = _last_bullet_end(lines, nb_index + 1, end)
        return lines[:at] + [bullet] + lines[at:]

    # package / docs → the top-level bullet list, above any notebook subsection
    region_end = nb_index if nb_index != -1 else end
    at = _last_bullet_end(lines, start + 1, region_end)
    return lines[:at] + [bullet] + lines[at:]


def _last_bullet_end(lines: list[str], lo: int, hi: int) -> int:
    """Return the index just after the last ``- `` bullet in ``lines[lo:hi]``.

    Parameters
    ----------
    lines : list[str]
        The changelog split into lines.
    lo : int
        Lower bound (inclusive).
    hi : int
        Upper bound (exclusive).

    Returns
    -------
    int
        The insertion index immediately after the last bullet, or ``lo`` if the
        range holds no bullets.
    """
    at = lo
    for i in range(lo, hi):
        if lines[i].startswith("- "):
            at = i + 1
    return at


def _trim_trailing_blanks(lines: list[str], start: int, end: int) -> int:
    """Return the index of the section end with trailing blank lines excluded.

    Parameters
    ----------
    lines : list[str]
        The changelog split into lines.
    start : int
        The section's header index.
    end : int
        The section's end index (exclusive).

    Returns
    -------
    int
        The index after the last non-blank line of the section.
    """
    at = end
    while at - 1 > start and lines[at - 1].strip() == "":
        at -= 1
    return at


def create_section(lines: list[str], version: tuple[int, int, int], scope: str, bullet: str) -> list[str]:
    """Prepend a fresh ``## X.Y.Z (in progress)`` section with one bullet.

    Parameters
    ----------
    lines : list[str]
        The changelog split into lines.
    version : tuple[int, int, int]
        The ``(major, minor, patch)`` of the new in-progress version.
    scope : str
        One of ``"package"``, ``"docs"`` or ``"notebook"``.
    bullet : str
        The rendered bullet line.

    Returns
    -------
    list[str]
        A new list of lines with the section prepended before the first ``## ``.
    """
    header = f"## {version[0]}.{version[1]}.{version[2]} (in progress)\n"
    block = [header]
    if scope == "notebook":
        block += [NOTEBOOK_HEADER + "\n", bullet, "\n"]
    else:
        block += [bullet, "\n"]

    first = next((i for i, line in enumerate(lines) if line.startswith("## ")), len(lines))
    return lines[:first] + block + lines[first:]


def version_string(lines: list[str]) -> str:
    """Read the active in-progress version as a string, else the latest released.

    Parameters
    ----------
    lines : list[str]
        The changelog split into lines.

    Returns
    -------
    str
        The ``"X.Y.Z"`` version string.
    """
    bounds = find_active_section(lines)
    target = bounds[0] if bounds else None
    if target is not None:
        match = IN_PROGRESS_RE.match(lines[target])
        return f"{match.group(1)}.{match.group(2)}.{match.group(3)}"
    major, minor, patch = latest_released(lines)
    return f"{major}.{minor}.{patch}"


def ensure_beta_version(version: str, dry_run: bool) -> str | None:
    """Ensure ``_version.py`` carries ``version`` with the ``b0`` beta suffix.

    Parameters
    ----------
    version : str
        The ``"X.Y.Z"`` in-progress version.
    dry_run : bool
        If ``True``, do not write — only report the intended change.

    Returns
    -------
    str | None
        A human-readable note if the file would change, else ``None``.
    """
    want = f'__version__ = "{version}b0"\n'
    current = VERSION_PATH.read_text()
    if current == want:
        return None
    if not dry_run:
        VERSION_PATH.write_text(want)
    return f"_version.py -> {version}b0"


def main() -> None:
    """Parse arguments and append the changelog bullet."""
    parser = argparse.ArgumentParser(description="Append a bullet to the active CHANGES.md section.")
    parser.add_argument("description", help="imperative change description")
    parser.add_argument("--scope", required=True, choices=["package", "docs", "notebook"])
    parser.add_argument("--issue", type=int, default=None, help="GitLab issue number to reference")
    parser.add_argument("--minor", action="store_true", help="minor bump when creating a section")
    parser.add_argument("--major", action="store_true", help="major bump when creating a section")
    parser.add_argument("--dry-run", action="store_true", help="print the result, write nothing")
    args = parser.parse_args()

    if args.minor and args.major:
        parser.error("pass at most one of --minor / --major")
    level = "major" if args.major else "minor" if args.minor else "patch"

    lines = CHANGES_PATH.read_text().splitlines(keepends=True)
    bullet = render_bullet(args.description, args.issue)
    bounds = find_active_section(lines)

    if bounds and bullet in lines[bounds[0]:bounds[1]]:
        sys.exit(f"Refusing: bullet already present in the active section: {bullet.strip()}")

    notices = []
    if bounds:
        new_lines = insert_into_section(lines, bounds, args.scope, bullet)
    else:
        version = bump_version(latest_released(lines), level)
        new_lines = create_section(lines, version, args.scope, bullet)
        notices.append(
            f"created ## {version[0]}.{version[1]}.{version[2]} (in progress) "
            f"[{level} bump — pass --minor/--major to change]"
        )

    if not args.dry_run:
        CHANGES_PATH.write_text("".join(new_lines))

    version_note = ensure_beta_version(version_string(new_lines), args.dry_run)
    if version_note:
        notices.append(version_note)

    prefix = "[dry-run] " if args.dry_run else ""
    print(f"{prefix}added to CHANGES.md ({args.scope}): {bullet.strip()}")
    for note in notices:
        print(f"{prefix}{note}")


if __name__ == "__main__":
    main()
