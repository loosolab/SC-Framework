---
name: release
description: Standalone pre-MR release preparation skill. Bumps _version.py, finalises the CHANGES.md (in progress) entry with a real date, and updates all notebook metadata versions. Run manually on dev before opening an MR to main. Never invoked automatically by the design/plan/implement workflow.
---

# release

Standalone skill for preparing a release on the `dev` branch before opening
a merge request to `main`. Performs the three steps from the project's MR
release template, then prints the post-merge checklist.

**Never invoked automatically** — the developer runs this when the branch is
ready to ship.

## Preflight

1. Confirm the current branch is `dev` (`git branch --show-current`). If not,
   warn the user and ask whether to proceed anyway — do not abort silently.
2. Confirm there are no uncommitted changes (`git status --porcelain`). If
   there are, stop and ask the user to commit or stash them first.
3. Read the current version from `src/sctoolbox/_version.py`
   (first line: `__version__ = "X.Y.Z"`).
4. **Confirm the conda environment** for Step 4's notebook-version script.
   Default `sctoolbox`; verify it exists with `conda env list`. The rest of the
   workflow always runs project scripts via `conda run -n <env>` — release does
   the same rather than assuming an env is active.
5. **Resolve the commit mode** for Step 5. Ask the user: **manual** (default —
   the user stages and commits the release themselves) or
   **claude (Name <email>)** (this skill commits with that identity, set
   per-commit via `-c` flags). Mirrors the design/plan/implement commit mode;
   default to **manual**, since the local setup can have issues with automatic
   commits.

## Process

### Step 1 — Choose the new version

Ask the user for the new version number. Show the current version as context.
Accept any valid semver `X.Y.Z`. Do not infer or auto-increment — the user
decides.

### Step 2 — Update `_version.py`

Set `src/sctoolbox/_version.py` to the clean `__version__ = "<new_version>"`.
Per the version lifecycle in `.claude/docs/sys-changelog.md`, the
working tree carries the in-progress `X.Y.Zb0`; this strips the `b0` suffix to
the released number.

### Step 3 — Finalise `CHANGES.md`

Apply the **Finalise** procedure in `.claude/docs/sys-changelog.md`: in
the active `## … (in progress)` section header, replace `(in progress)` with
today's date in `(DD-MM-YYYY)` format (updating the header version too if the
chosen release number differs from the in-progress one). If no `(in progress)`
entry exists, warn the user — the CHANGES.md may need a manual entry.

### Step 4 — Update notebook versions

Run the version update script for each notebook directory, in the conda env
confirmed in preflight:

```bash
conda run -n <env> python scripts/change_notebook_version.py rna_analysis/notebooks/ <new_version>
conda run -n <env> python scripts/change_notebook_version.py atac_analysis/notebooks/ <new_version>
conda run -n <env> python scripts/change_notebook_version.py general_notebooks/ <new_version>
```

### Step 5 — Commit

The modified files are:
- `src/sctoolbox/_version.py`
- `CHANGES.md`
- All `.ipynb` files touched by the version script

Honour the commit mode resolved in preflight:

- **`manual`** (default) — do **not** commit. List the modified files
  (`git status --short`) and tell the user to stage these explicitly and commit
  with message `release: <new_version>`.
- **`claude (Name <email>)`** — stage the files explicitly, then commit setting
  both author and committer via per-invocation `-c` flags (never `git config`),
  with message `release: <new_version>`:
  `git -c user.name="Name" -c user.email="email" commit --author="Name <email>" -m "release: <new_version>"`.

Do not use `sys-commit` — this release commit has its own convention and no
slug. Never blind `git add -A` / `git commit -a`; stage the listed paths
explicitly. Never run a git op denied by `.claude/settings.json`.

## Post-merge checklist

After the commit, print the following reminder to the user:

```
Release commit ready. Open an MR from dev → main, then work through this
checklist:

Before merge:
  [ ] MR description: use the "new_release_template" from .gitlab/merge_request_templates/

After merge (~1.5 hrs for the full pipeline):
  GitLab (automated)
    [ ] New release created
    [ ] Version tag added
    [ ] Add milestone to release and close it (if one exists)

  GitHub  (https://github.com/loosolab/SC-Framework — mirrored automatically)
    [ ] Confirm repository updated
    [ ] Create a new GitHub release (copy description from GitLab)

  Zenodo  (https://zenodo.org/records/14056105 — triggered by GitHub release)
    [ ] Confirm Zenodo release was triggered
    [ ] Adjust authors (see prior release for reference)

  PyPI  (final CI stage, ~1.5 hrs)
    [ ] Confirm release published successfully
```
