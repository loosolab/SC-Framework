# SC-Framework — Claude context

> **Authoritative sources** — conventions: `docs/source/development.rst`; tooling, deps, ruff, pytest, codespell: `pyproject.toml`; CI gates: `.gitlab-ci.yml`. This file is a curated summary; where it conflicts with them, they win — defer to the source and update this file.

## Project

Python package (`sctoolbox`) for single-cell analysis workflows covering scRNA-seq and scATAC-seq data. Distributed as `SC-Framework` on PyPI. Maintained at MPI Bad Nauheim (Looso lab).

## Stack

- **Language:** Python >=3.9, <3.13
- **Core deps:** AnnData, Scanpy (+ leiden community detection)
- **Optional dep groups:** declared in `pyproject.toml` under `[project.optional-dependencies]` (the authoritative list — read it rather than relying on a copy here)
- **Test framework:** pytest + pytest-cov + pytest-html (target coverage >90%)
- **Linter:** ruff (config in `pyproject.toml`)
- **Notebooks:** Jupyter (rna_analysis/notebooks/, atac_analysis/notebooks/, general_notebooks/)
- **Build:** setuptools + setuptools_scm
- **Runtime type checking:** beartype (>=0.18.2)

> Not every dev/test dependency lives in `pyproject.toml`. Some are git-only and installed **only by CI** — notably `scar` (lazy `import scar` in `tools/qc_filter.py`; commented out in `pyproject.toml` and `sctoolbox_env.yml`, excluded from `.[all]`) and the `papermill`/`mampok` installs in the notebook jobs. When reasoning about "all dev packages," check `.gitlab-ci.yml` too.

## Module map

Source lives under `src/sctoolbox/`, grouped by purpose — browse the directories for the full, current set:

```
src/sctoolbox/
  tools/        # analysis logic, one file per topic (e.g. clustering.py, qc_filter.py, tobias.py)
  plotting/     # plot functions mirroring tools (e.g. embedding.py, marker_genes.py)
  utils/        # shared helpers (e.g. adata.py, decorator.py — @log_anndata/@beartype)
  _settings.py  # global SCtoolboxConfig; access via sctoolbox.settings
  _modules.py   # lazy module loader
```

## Test layout

```
tests/
  conftest.py
  test_settings.py
  tools/      # mirrors src/sctoolbox/tools/
  plotting/   # mirrors src/sctoolbox/plotting/
  utils/      # mirrors src/sctoolbox/utils/
  data/       # test fixtures
```

Tests mirror the source module structure: a test for `src/sctoolbox/<group>/<mod>.py` lives at `tests/<group>/test_<mod>.py`. A new submodule must come with its mirrored test file.

## Commands

Run all of these **inside the project conda env** (the env name/path is per-user; see the workflow's recorded env / your local settings). The workflow's binding test command is authoritative for a given change — this is the general cheat-sheet:

**How to invoke conda:** `conda run -p <prefix> <cmd>` (or `-n <name>`) for gate and workflow commands — that is the form `.claude/settings.local.json` allow-lists, so it runs without a permission prompt — and bare `conda activate <path> && <cmd>` for ad-hoc probing. Do **not** prefix with `source $(conda info --base)/etc/profile.d/conda.sh`: the Bash tool's shell is already initialized from the user's profile. `mamba activate` may not be initialized; use `conda`. Shell state does not persist between calls, so repeat the prefix in each one.

```bash
ruff check          # lint + docstring checks (the binding test command always starts here)
pytest              # unit tests with coverage (target > 90%)
make -C docs html   # full Sphinx build — the authoritative check for renderable examples
```

## Conventions

The rules below are the working digest; `docs/source/development.rst` is the authoritative, fuller treatment (decorators, beartype, docstrings, examples, deprecation, changelog, notebooks, testing). Consult it when a case isn't covered here.

### Style fidelity

New code, tests, docstrings, examples, and changelog entries must read like the existing repository — not a generic house style. Before writing into a module, read its nearest existing sibling (the mirrored test file, a neighbouring `tools/`/`plotting/`/`utils/` module) and mirror what you find: import grouping and aliases, naming, helper/fixture reuse, error and logging idioms, docstring phrasing, and comment density. Prefer an existing helper over a new one; match the surrounding patterns rather than introducing a different-but-valid approach. Prefer the smallest change that solves the actual problem — no speculative abstraction, no new helper where an existing one fits, no scope creep beyond what was asked. The explicit conventions below take precedence where they apply.

### Decorators

Decorator order is strict — `@log_anndata` must always be outermost, `@beartype` directly above the function:

```python
import sctoolbox.utils.decorator as deco
from beartype import beartype

@deco.log_anndata   # only on top-level functions that receive an AnnData/MuData
@beartype           # on all public functions
def my_function(...):
    ...
```

`@log_anndata` is for top-level/important functions only, not every function. `@beartype` goes on all public functions.

### AnnData mutations

Functions either modify `adata` in-place (return `None`) or return a new object — never both. The docstring must state which.

### Logging

Acquire the logger as `logger = settings.logger` (module level) — never `print()` or a bare `logging.getLogger`. Emit via `logger.info` / `logger.warning`, matching the surrounding call sites.

### Imports

Keep module top-level imports light: import optional or heavy dependencies *inside* the function that needs them, not at module top (e.g. `scar` and often `scanpy` in `tools/qc_filter.py`). `import sctoolbox` must succeed without any optional dep installed, and the docs build must be able to import every module — a top-level optional import breaks both.

### Plotting functions

Accept an optional `ax` parameter (matplotlib `Axes`); create one internally if `None`. Always return the axes object.

### Docstrings

Numpy-style docstrings are required on all functions and enforced by ruff. Renderable `Examples` sections are best practice but never mandatory: plotting functions are **highly encouraged** to add a `.. plot::` example, other public functions **encouraged** to add an `.. exec_code::` example where the output is illustrative. See the "Example code and results" section of `docs/source/development.rst` for the directives, pre-code fixtures, and render-checking.

### New submodules

A new file in `tools/`, `plotting/`, or `utils/` must be registered in the `__all__` list of the parent `__init__.py`.

### Settings

Functions should use `sctoolbox.settings` for defaults (threads, file paths). Allow a parameter to override the setting where appropriate.

### Tests

Don't mask a missing optional dependency to make the suite go green — no `pytest.importorskip`, skip markers, or try/except import guards added for that purpose. Leave a genuine environment gap as a visible failure and report it; fix real test bugs (e.g. a fixture setup error) instead. Add a guard only if explicitly asked.

### Bug fixes

Fix causes, not symptoms. When a regenerable artifact/cache lands in the wrong place (e.g. scanpy's default `./data/` datasetdir polluting the repo root), route the write to the right location rather than adding a `.gitignore` entry to hide it; offer a `.gitignore` only as a last resort.

### Deprecation

Mark deprecated functions with the `deprecation` package, targeting removal in 2 minor versions. Add `@deprecation.fail_if_not_removed` to the function's test.

### Notebooks

- Outputs must be cleared before committing (`.gitconfig` in the repo handles this automatically when the sctoolbox env is active).
- Location: RNA-specific → `rna_analysis/notebooks/`; ATAC-specific → `atac_analysis/notebooks/`; data-agnostic → `general_notebooks/`.
- Naming: numeric prefix for ordered notebooks (`01_`, `02_`, ...); letter prefix for optional (`0A_`, `0B_`); `99-report.ipynb` always last.
- Input cells must be colored `PowderBlue` via `bgcolor()` in the hidden init cell.
- Output files: numeric prefix, plots in PDF format.

### CHANGES.md

The changelog covers **sc-framework changes only** — the package, the notebooks, and the documentation. Changes to the Claude workflow itself (`CLAUDE.md`, `.claude/**`) get **no** entry: they ship no user-visible behaviour.

Every change to the package, notebooks, or documentation requires an entry in `CHANGES.md`. Package- and notebook-scope changes are enforced by CI; docs-only changes are not gated by CI, but the dev workflow still requires an entry. Docs entries go under the main section header (like package changes). Format:

```markdown
## 0.xy.z (in progress)
- description of change (#<issue number>)

### Changes to notebooks
- description (#<issue number>)
```

Keep each bullet a terse phrase (e.g. "enables parallel test execution"), not a multi-clause sentence enumerating every sub-change — the detail lives in the commit history and `.work/`.

## Agentic workflow

Skills live in `.claude/skills/`. The development loop is `/design → /plan → /implement → /retro`.

### Working agreements

These bind every Claude installation working in this repo — they are checked in
precisely so two installations do not work from different ground.

- **No agent memory for this project.** Never write to the memory store, and never
  rely on a recalled memory as a source of truth about this repo. Everything
  project-related — conventions, gotchas, tooling details, decisions — belongs in
  the repository: `CLAUDE.md` for durable conventions, `.claude/skills/` and
  `.claude/docs/` for workflow mechanics, `.work/<item>/` for a work item's audit
  trail, `CHANGES.md` and the commit history for what shipped. If you find a stray
  memory carrying project knowledge, fold it into the right repo file and delete
  it (`/retro` does this sweep). **Why:** a private memory store is per-installation
  and invisible to everyone else, so anything kept there silently diverges.
- **Discuss before acting.** Summarise what is known, name the intended next step,
  and get agreement — including before a long chain of read-only probing. Offer
  `/design` for anything non-trivial rather than starting in.
- **Check the project's own helpers first.** Before hand-rolling a
  `curl`/`bash`/`python -c`, look in `.claude/skills/` and `scripts/` for the
  canonical helper (e.g. a GitLab issue or MR goes through
  `scripts/gitlab_query.py search|fetch`, never a raw API call). The shipped
  helpers are vetted, read-only and allow-listed; hand-rolled equivalents bypass
  that and trigger prompts or fail.
- **This file is team documentation.** `CLAUDE.md` is checked in and doubles as
  human onboarding, so it is keep-by-default: do not propose trimming a section
  because a manifest or `ls` could reconstruct it. If a section is factually stale,
  correct it in place rather than deleting it.

The user need not type `/design` to start. For a **non-trivial change** — a new submodule or public function, edits spanning several files, anything that should carry tests + a `CHANGES.md` entry, or behaviour that wants the regression gate — proactively offer to begin at `/design` and proceed only if the user agrees. For **trivial work** (a one-line fix, typo or docstring tweak, single localized edit, or exploratory question) skip the workflow and handle it directly. When in doubt, name the choice and let the user decide.

Artifacts (`design.md`, `plan.md`, `review-*.md`) live in `.work/<YYYY-MM-DD>-<slug>/` — **gitignored, a local-only audit trail**. `design.md` records the **scope** (`package` / `notebooks` / `docs`), conda env, commit mode, and autonomy (`per-task` / `end`); `plan.md` declares the binding test command (always starts with `ruff check`; a `docs` scope gates on `make -C docs html`, mixed scopes chain the gates). The per-skill mechanics — including staging and commit identity — live in the skills (`.claude/skills/sys-commit/SKILL.md` for commits).

**Commit mode** (default `manual` — the workflow never touches git; the local setup can misbehave on auto-commits) is set at `/design`. `/design` and `/plan` make no commit; code commits begin in `/implement`, stage only code/tests/`CHANGES.md`, and carry the `<slug>` so `git log --grep=<slug>` recovers a work item's code history.
