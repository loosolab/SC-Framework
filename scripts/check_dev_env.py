"""Verify the active environment carries the required development packages.

Reads the authoritative dev-tooling lists from ``pyproject.toml``
``[dependency-groups]`` (``test`` / ``lint`` / ``spellcheck`` / ``docs`` /
``dev``) and checks each required distribution against what is installed in the
interpreter running this script. Run it **inside** the target conda env so its
``importlib.metadata`` reflects that env::

    conda run -p <env> python scripts/check_dev_env.py --scope package

This replaces the manual, per-scope ``--version`` / ``import`` probing the
development workflow used to do by hand (``.claude/skills/sys-env-check``): one
read-only check, exit 0 when everything required for the scope is present and a
non-zero exit listing what is missing otherwise. It mutates nothing — installing
what is missing stays the caller's (confirmed) decision.

Scopes map to dependency groups. ``lint`` (ruff) and ``spellcheck`` (codespell)
gate every change repo-wide, so they — plus an editable ``sctoolbox`` install
(``pip install -e .``) — are always checked:

* ``package``   → ``test`` + importable ``scar`` (a git-only test dep kept out
  of ``.[all]`` for size; CI's test job installs it)
* ``docs``      → ``docs`` + the system ``pandoc`` binary
* ``notebooks`` → ``nbconvert`` importable (notebook tooling is conda-only, from
  ``sctoolbox_env.yml`` — it has no pip dependency group)

With no ``--scope`` the full ``dev`` group is verified (a complete dev env),
including every scope's extra checks.

Examples
--------
    python3 scripts/check_dev_env.py                      # full dev env
    python3 scripts/check_dev_env.py --scope package      # package gate tooling
    python3 scripts/check_dev_env.py --scope docs --scope notebooks
"""

import argparse
import importlib.metadata as metadata
import importlib.util
import json
import shutil
import sys
from pathlib import Path

try:
    import tomllib
except ModuleNotFoundError:  # Python < 3.11
    import tomli as tomllib

from packaging.requirements import Requirement

PYPROJECT_PATH = Path("pyproject.toml")
# Distribution name on PyPI; the import package is ``sctoolbox``.
SCTOOLBOX_DIST = "SC-Framework"

# Groups checked for every scope (ruff + codespell gate any change, repo-wide),
# plus the editable-install import probe.
ALWAYS_GROUPS = ("lint", "spellcheck")
# scope -> extra dependency groups it pulls in.
SCOPE_GROUPS = {
    "package": ("test",),
    "docs": ("docs",),
    "notebooks": (),  # conda-only tooling (sctoolbox_env.yml); probed via nbconvert
}


def load_groups(pyproject: Path) -> dict[str, list]:
    """Read the ``[dependency-groups]`` table from ``pyproject.toml``.

    Parameters
    ----------
    pyproject : Path
        Path to the ``pyproject.toml`` to read.

    Returns
    -------
    dict[str, list]
        The raw dependency-groups mapping (group name -> list of requirement
        strings and ``{"include-group": ...}`` dicts).

    Raises
    ------
    SystemExit
        If the file is missing or declares no ``[dependency-groups]``.
    """
    if not pyproject.is_file():
        raise SystemExit(f"error: {pyproject} not found (run from the repository root)")
    data = tomllib.loads(pyproject.read_text())
    groups = data.get("dependency-groups")
    if not groups:
        raise SystemExit(f"error: no [dependency-groups] table in {pyproject}")
    return groups


def resolve_group(name: str, groups: dict[str, list], _seen: set[str] | None = None) -> list[str]:
    """Flatten a dependency group into its requirement strings.

    Resolves ``{"include-group": ...}`` references recursively (PEP 735), so an
    aggregate group such as ``dev`` expands to every requirement it bundles.

    Parameters
    ----------
    name : str
        The group to resolve.
    groups : dict[str, list]
        The full ``[dependency-groups]`` mapping.
    _seen : set[str] | None, default None
        Groups already visited, guarding against include cycles.

    Returns
    -------
    list[str]
        The PEP 508 requirement strings the group pulls in, in declaration
        order.

    Raises
    ------
    SystemExit
        If a referenced group is undefined.
    """
    _seen = _seen or set()
    if name in _seen:
        return []
    _seen.add(name)
    if name not in groups:
        raise SystemExit(f"error: dependency group '{name}' not declared in pyproject.toml")

    requirements: list[str] = []
    for entry in groups[name]:
        if isinstance(entry, dict):
            requirements.extend(resolve_group(entry["include-group"], groups, _seen))
        else:
            requirements.append(entry)
    return requirements


def check_requirement(req_string: str) -> tuple[str, bool, str]:
    """Check one requirement against the installed distributions.

    Presence is resolved by distribution name (not import name), and any version
    specifier is enforced against the installed version. Extras are ignored —
    only the base distribution is verified.

    Parameters
    ----------
    req_string : str
        A PEP 508 requirement string (e.g. ``"codespell[toml]"``, ``"ruff>=0.1"``).

    Returns
    -------
    tuple[str, bool, str]
        ``(name, ok, detail)`` — the distribution name, whether it is present and
        satisfies its specifier, and a human-readable status (the version, or why
        it failed).
    """
    req = Requirement(req_string)
    try:
        installed = metadata.version(req.name)
    except metadata.PackageNotFoundError:
        return req.name, False, "MISSING"

    if req.specifier and installed not in req.specifier:
        return req.name, False, f"{installed} (needs {req.specifier})"
    return req.name, True, installed


def select_checks(scopes: list[str]) -> tuple[list[str], bool, bool, bool]:
    """Map requested scopes to the dependency groups and extra probes to run.

    Parameters
    ----------
    scopes : list[str]
        The requested scopes (subset of :data:`SCOPE_GROUPS`); empty means the
        full ``dev`` group.

    Returns
    -------
    tuple[list[str], bool, bool, bool]
        ``(group_names, check_nbconvert, check_pandoc, check_scar)`` — the
        dependency groups to verify and whether the ``nbconvert`` / ``pandoc`` /
        ``scar`` extra probes apply.
    """
    if not scopes:
        return ["dev"], True, True, True  # the aggregate group: every dev tool
    group_names = list(ALWAYS_GROUPS)
    for scope in scopes:
        group_names.extend(SCOPE_GROUPS[scope])
    return group_names, "notebooks" in scopes, "docs" in scopes, "package" in scopes


def check_groups(group_names: list[str], groups_table: dict[str, list]) -> list[str]:
    """Check and print every group's requirements; return the failures.

    Parameters
    ----------
    group_names : list[str]
        The dependency groups to verify (deduplicated, order preserved).
    groups_table : dict[str, list]
        The full ``[dependency-groups]`` mapping.

    Returns
    -------
    list[str]
        One entry per missing or version-mismatched requirement.
    """
    missing: list[str] = []
    seen_names: set[str] = set()
    for group in dict.fromkeys(group_names):
        print(f"  group '{group}'")
        for req_string in resolve_group(group, groups_table):
            name, ok, detail = check_requirement(req_string)
            if name in seen_names:
                continue
            seen_names.add(name)
            print(f"    {name:<22} {detail}")
            if not ok:
                missing.append(f"{req_string} (group: {group})")
    return missing


def check_extras(check_nbconvert: bool, check_pandoc: bool, check_scar: bool) -> list[str]:
    """Probe tooling that has no pip dependency group; return the failures.

    Notebook execution (``nbconvert``), the docs ``pandoc`` binary, and ``scar``
    come from conda / git, not ``[dependency-groups]``, so they are probed
    directly rather than looked up. ``scar`` is a git-only test dependency
    (``git+https://github.com/Novartis/scar.git``) deliberately kept out of
    ``.[all]`` and the docker image for size, but CI's test job installs it to
    exercise scar-related functions — so the package gate requires it.

    Parameters
    ----------
    check_nbconvert : bool
        Whether to probe for an importable ``nbconvert`` (notebooks scope).
    check_pandoc : bool
        Whether to probe for a ``pandoc`` binary on ``PATH`` (docs scope).
    check_scar : bool
        Whether to probe for an importable ``scar`` (package scope).

    Returns
    -------
    list[str]
        One entry per missing probe.
    """
    missing: list[str] = []
    if check_nbconvert or check_pandoc or check_scar:
        print("  extra probes")
    if check_nbconvert:
        ok = importlib.util.find_spec("nbconvert") is not None
        print(f"    {'nbconvert':<22} {'ok' if ok else 'MISSING (notebook execution)'}")
        if not ok:
            missing.append("nbconvert (notebook tooling, see sctoolbox_env.yml)")
    if check_pandoc:
        path = shutil.which("pandoc")
        print(f"    {'pandoc':<22} {path or 'MISSING (system binary, needed for docs build)'}")
        if path is None:
            missing.append("pandoc (system binary on PATH, needed for the docs build)")
    if check_scar:
        ok = importlib.util.find_spec("scar") is not None
        print(f"    {'scar':<22} {'ok' if ok else 'MISSING (git-only test dep)'}")
        if not ok:
            missing.append("scar (git-only test dep: pip install git+https://github.com/Novartis/scar.git)")
    return missing


def check_editable_install() -> tuple[bool, str]:
    """Check ``sctoolbox`` is importable *and* installed editable.

    A plain ``pip install .`` (or a PyPI wheel) is importable but would run
    tests against stale installed code rather than the working ``src/`` tree, so
    development requires the editable install (``pip install -e .``). Editability
    is read from the distribution's ``direct_url.json`` (``dir_info.editable``).

    Returns
    -------
    tuple[bool, str]
        ``(ok, detail)`` — whether the package is importable and an editable
        install, plus a human-readable status string.
    """
    if importlib.util.find_spec("sctoolbox") is None:
        return False, "MISSING (pip install -e .)"
    try:
        raw = metadata.distribution(SCTOOLBOX_DIST).read_text("direct_url.json")
    except metadata.PackageNotFoundError:
        raw = None
    editable = bool(raw) and json.loads(raw).get("dir_info", {}).get("editable", False)
    if editable:
        return True, "ok (editable)"
    return False, "importable but NOT editable (pip install -e .)"


def main() -> int:
    """Run the verification and return a process exit code.

    Returns
    -------
    int
        ``0`` when every required package for the requested scope is present,
        ``1`` otherwise.
    """
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--scope",
        action="append",
        choices=sorted(SCOPE_GROUPS),
        help="restrict the check to this scope (repeatable); "
        "omit to verify the full 'dev' group.",
    )
    parser.add_argument(
        "--pyproject",
        type=Path,
        default=PYPROJECT_PATH,
        help="path to pyproject.toml (default: ./pyproject.toml).",
    )
    args = parser.parse_args()

    groups_table = load_groups(args.pyproject)
    scopes = args.scope or []
    group_names, check_nbconvert, check_pandoc, check_scar = select_checks(scopes)

    print(f"Verifying dev packages — Python {sys.version.split()[0]} at {sys.prefix}")
    print(f"Scope: {', '.join(scopes) if scopes else 'full (dev)'}\n")

    # The package must import AND be an editable install (src/ edits take effect).
    editable_ok, detail = check_editable_install()
    print("  editable install")
    print(f"    {'sctoolbox':<22} {detail}")
    missing = [] if editable_ok else ["sctoolbox editable install (pip install -e .)"]

    missing += check_groups(group_names, groups_table)
    missing += check_extras(check_nbconvert, check_pandoc, check_scar)

    print()
    if missing:
        print(f"FAILED: {len(missing)} required item(s) missing or mismatched:")
        for item in missing:
            print(f"  - {item}")
        print("\nInstall the dev tooling with:")
        print("  pip install -e '.[all]' --group dev")
        print("(pandoc and notebook tooling come from conda / sctoolbox_env.yml;")
        print(" scar from git+https://github.com/Novartis/scar.git.)")
        return 1

    print("OK: all required development packages present.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
