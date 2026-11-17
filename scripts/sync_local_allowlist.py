"""Regenerate ``.claude/settings.local.json`` from ``.claude/settings.json``.

The committed ``.claude/settings.json`` allow-lists the workflow's gate commands
for the **named** conda env (``conda run -n sctoolbox …``). A developer whose env
lives at a **prefix** path (``conda run -p /path …``) needs the same rules
rewritten for that prefix in the gitignored, per-user
``.claude/settings.local.json`` — otherwise every gate prompts. Keeping the two
in sync by hand drifts; this script regenerates the local file deterministically
from the committed one, so a new ``conda run`` rule only has to be added once (to
``settings.json``).

It copies every ``conda run -n <name> …`` allow rule from ``settings.json``,
rewrites the ``-n <name>`` selector to ``-p <prefix>``, and writes the result as
the ``settings.local.json`` allow list (any other keys in that file are kept).
Env-agnostic rules (``Edit``, ``Write``, ``git``, …) stay in ``settings.json``
only — they need no prefix variant.

The prefix is taken from ``--prefix``, else reused from the existing
``settings.local.json``, else ``$CONDA_PREFIX``. Run it after editing
``settings.json``.

Examples
--------
    python3 scripts/sync_local_allowlist.py
    python3 scripts/sync_local_allowlist.py --prefix /mnt/share/me/.conda/sctoolbox
    python3 scripts/sync_local_allowlist.py --name sctoolbox --dry-run
"""

import argparse
import json
import os
import re
import sys
from pathlib import Path

SETTINGS_PATH = Path(".claude/settings.json")
LOCAL_PATH = Path(".claude/settings.local.json")
PREFIX_RE = re.compile(r"conda run -p (\S+) ")


def load_json(path: Path) -> dict:
    """Read and parse a JSON file.

    Parameters
    ----------
    path : Path
        The JSON file to read.

    Returns
    -------
    dict
        The parsed contents.

    Raises
    ------
    SystemExit
        If the file is missing or is not valid JSON.
    """
    if not path.is_file():
        raise SystemExit(f"error: {path} not found (run from the repository root)")
    try:
        return json.loads(path.read_text())
    except json.JSONDecodeError as exc:
        raise SystemExit(f"error: {path} is not valid JSON: {exc}") from exc


def detect_prefix(local: dict) -> str | None:
    """Recover the env prefix from an existing local allow list.

    Parameters
    ----------
    local : dict
        Parsed ``settings.local.json`` contents.

    Returns
    -------
    str | None
        The prefix from the first ``conda run -p <prefix>`` rule, or ``None`` if
        none is present.
    """
    for rule in local.get("permissions", {}).get("allow", []):
        match = PREFIX_RE.search(rule)
        if match:
            return match.group(1)
    return None


def rewrite_rules(allow: list[str], name: str, prefix: str) -> list[str]:
    """Select the named-env ``conda run`` rules and rewrite them to the prefix.

    Parameters
    ----------
    allow : list[str]
        The committed ``settings.json`` allow list.
    name : str
        The named env in ``settings.json`` to mirror (e.g. ``sctoolbox``).
    prefix : str
        The prefix path to rewrite ``-n <name>`` to.

    Returns
    -------
    list[str]
        The matching rules, rewritten to ``conda run -p <prefix>`` (declaration
        order preserved).
    """
    marker = f"conda run -n {name} "
    replacement = f"conda run -p {prefix} "
    return [rule.replace(marker, replacement) for rule in allow if marker in rule]


def main() -> int:
    """Regenerate the local allow list and return a process exit code.

    Returns
    -------
    int
        ``0`` on success.

    Raises
    ------
    SystemExit
        If no env prefix can be determined, or ``settings.json`` has no
        matching ``conda run -n <name>`` rules.
    """
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--name",
        default="sctoolbox",
        help="named env in settings.json to mirror (default: sctoolbox).",
    )
    parser.add_argument(
        "--prefix",
        default=None,
        help="env prefix path; defaults to the existing settings.local.json, "
        "else $CONDA_PREFIX.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="print the regenerated settings.local.json without writing it.",
    )
    args = parser.parse_args()

    settings = load_json(SETTINGS_PATH)
    allow = settings.get("permissions", {}).get("allow", [])

    local = load_json(LOCAL_PATH) if LOCAL_PATH.is_file() else {}
    prefix = args.prefix or detect_prefix(local) or os.environ.get("CONDA_PREFIX")
    if not prefix:
        raise SystemExit(
            "error: no env prefix found — pass --prefix or set $CONDA_PREFIX."
        )

    rules = rewrite_rules(allow, args.name, prefix)
    if not rules:
        raise SystemExit(
            f"error: no 'conda run -n {args.name}' rules in {SETTINGS_PATH}."
        )

    local.setdefault("permissions", {})["allow"] = rules
    output = json.dumps(local, indent=2) + "\n"

    if args.dry_run:
        sys.stdout.write(output)
        return 0
    LOCAL_PATH.write_text(output)
    print(f"Wrote {len(rules)} prefix-env rule(s) to {LOCAL_PATH} (prefix: {prefix})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
