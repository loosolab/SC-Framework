"""Read-only GitLab client for the development workflow: search and fetch issues / MRs.

Resolves the project from the ``origin`` git remote and prints markdown to stdout,
so a caller can capture the result (e.g. into ``.work/<slug>/related.md``) to
inform planning. Used by the ``/design`` stage to find work related to a change
and to pull the requirements/discussion of specific issues or merge requests.

Authentication has **no self token-handling**: the client is built via
``gitlab.Gitlab.from_config()`` when a ``python-gitlab`` configuration is present
(``~/.python-gitlab.cfg`` / ``PYTHON_GITLAB_CFG``) and targets the same host as the
remote; otherwise it falls back to an anonymous client (the project is publicly
readable). The ``python-gitlab`` library handles any token internally — this
script never reads, echoes, or persists a secret, and makes no CI assumptions.

Examples
--------
    conda run -p <prefix> python scripts/gitlab.py search "batch correction mnn"
    conda run -p <prefix> python scripts/gitlab.py search "atac qc" --type issues --state opened
    conda run -p <prefix> python scripts/gitlab.py fetch issue 445
    conda run -p <prefix> python scripts/gitlab.py fetch mr 515 --full
"""

import argparse
import subprocess

import gitlab


def project_from_remote() -> tuple[str, str]:
    """Derive the GitLab base URL and project path from the ``origin`` remote.

    Returns
    -------
    tuple[str, str]
        ``(base_url, project_path)`` — e.g. ``("https://gitlab.gwdg.de",
        "loosolab/software/sc_framework")``.

    Raises
    ------
    ValueError
        If the ``origin`` remote URL cannot be parsed.
    """
    url = subprocess.check_output(["git", "remote", "get-url", "origin"], text=True).strip()
    cleaned = url[:-4] if url.endswith(".git") else url
    if cleaned.startswith("git@"):  # git@host:group/project
        host, path = cleaned[len("git@"):].split(":", 1)
        return f"https://{host}", path
    if "://" in cleaned:  # scheme://host/group/project
        scheme, rest = cleaned.split("://", 1)
        host, path = rest.split("/", 1)
        return f"{scheme}://{host}", path
    raise ValueError(f"Cannot parse origin remote URL: {url}")


def make_client(base_url: str) -> gitlab.Gitlab:
    """Build a GitLab client, preferring a configured (possibly authenticated) one.

    Parameters
    ----------
    base_url : str
        The GitLab base URL derived from the remote.

    Returns
    -------
    gitlab.Gitlab
        A configured client when ``from_config()`` targets the same host, else an
        anonymous client for ``base_url``.
    """
    try:
        client = gitlab.Gitlab.from_config()
        if client.url.rstrip("/") == base_url.rstrip("/"):
            return client
    except Exception:  # noqa: BLE001 — any config issue falls back to anonymous
        pass
    return gitlab.Gitlab(base_url)


def _labels(obj: object) -> str:
    """Format an issue/MR's labels as a bracketed string, or empty.

    Parameters
    ----------
    obj : object
        A ``python-gitlab`` issue or merge-request object.

    Returns
    -------
    str
        ``" [a, b]"`` when labels exist, else ``""``.
    """
    labels = getattr(obj, "labels", []) or []
    return f"  [{', '.join(labels)}]" if labels else ""


def _list_kwargs(state: str, terms: str, limit: int) -> dict:
    """Build keyword arguments for a ``python-gitlab`` list call.

    Parameters
    ----------
    state : str
        ``"all"``, ``"opened"`` or ``"closed"``.
    terms : str
        Free-text search terms.
    limit : int
        Maximum number of results.

    Returns
    -------
    dict
        Keyword arguments for ``.list()``.
    """
    kwargs = {"search": terms, "per_page": limit, "get_all": False}
    if state != "all":
        kwargs["state"] = state
    return kwargs


def search(project: object, terms: str, kind: str, state: str, limit: int) -> None:
    """Print issues and/or MRs matching ``terms``.

    Parameters
    ----------
    project : object
        The resolved ``python-gitlab`` project.
    terms : str
        Free-text search terms.
    kind : str
        ``"issues"``, ``"mrs"`` or ``"both"``.
    state : str
        ``"all"``, ``"opened"`` or ``"closed"``.
    limit : int
        Maximum number of results per kind.
    """
    kwargs = _list_kwargs(state, terms, limit)
    if kind in ("issues", "both"):
        print(f"## Issues matching {terms!r}\n")
        for item in project.issues.list(**kwargs):
            print(f"- #{item.iid} ({item.state}) {item.title}{_labels(item)}\n  {item.web_url}")
        print()
    if kind in ("mrs", "both"):
        print(f"## Merge requests matching {terms!r}\n")
        for item in project.mergerequests.list(**kwargs):
            print(f"- !{item.iid} ({item.state}) {item.title}{_labels(item)}\n  {item.web_url}")
        print()


def fetch(project: object, kind: str, number: int, full: bool) -> None:
    """Print one issue or MR with its description and discussion.

    Parameters
    ----------
    project : object
        The resolved ``python-gitlab`` project.
    kind : str
        ``"issue"`` or ``"mr"``.
    number : int
        The issue/MR ``iid``.
    full : bool
        If ``True``, include system notes; otherwise only human content.
    """
    obj = project.issues.get(number) if kind == "issue" else project.mergerequests.get(number)
    marker = "#" if kind == "issue" else "!"
    print(f"# {marker}{number} {obj.title}\n")
    print(f"- state: {obj.state}{_labels(obj)}")
    print(f"- author: {obj.author['username']}")
    if kind == "mr":
        print(f"- branch: {obj.source_branch} -> {obj.target_branch}")
    print(f"- url: {obj.web_url}\n")
    print("## Description\n")
    print((obj.description or "*(no description)*").strip(), "\n")

    notes = sorted(obj.notes.list(get_all=True), key=lambda note: note.created_at)
    print("## Discussion\n")
    shown = 0
    for note in notes:
        if note.system and not full:
            continue
        tag = " (system)" if note.system else ""
        print(f"**{note.author['username']}**{tag} — {note.created_at[:10]}\n")
        print(note.body.strip(), "\n")
        shown += 1
    if shown == 0:
        print("*(no discussion)*")


def main() -> None:
    """Parse arguments and dispatch to ``search`` or ``fetch``."""
    parser = argparse.ArgumentParser(description="Read-only GitLab search/fetch for the workflow.")
    sub = parser.add_subparsers(dest="command", required=True)

    sp = sub.add_parser("search", help="search issues and/or merge requests")
    sp.add_argument("terms", help="free-text search terms")
    sp.add_argument("--type", choices=["issues", "mrs", "both"], default="both")
    sp.add_argument("--state", choices=["all", "opened", "closed"], default="all")
    sp.add_argument("--limit", type=int, default=20)

    fp = sub.add_parser("fetch", help="fetch one issue or merge request")
    fp.add_argument("kind", choices=["issue", "mr"])
    fp.add_argument("number", type=int)
    fp.add_argument("--full", action="store_true", help="include system notes")

    args = parser.parse_args()
    base_url, project_path = project_from_remote()
    project = make_client(base_url).projects.get(project_path)

    if args.command == "search":
        search(project, args.terms, args.type, args.state, args.limit)
    else:
        fetch(project, args.kind, args.number, args.full)


if __name__ == "__main__":
    main()
