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

## Conventions

The rules below are the working digest; `docs/source/development.rst` is the authoritative, fuller treatment (decorators, beartype, docstrings, examples, deprecation, changelog, notebooks, testing). Consult it when a case isn't covered here.

### Style fidelity

New code, tests, docstrings, examples, and changelog entries must read like the existing repository — not a generic house style. Before writing into a module, read its nearest existing sibling (the mirrored test file, a neighbouring `tools/`/`plotting/`/`utils/` module) and mirror what you find: import grouping and aliases, naming, helper/fixture reuse, error and logging idioms, docstring phrasing, and comment density. Prefer an existing helper over a new one; match the surrounding patterns rather than introducing a different-but-valid approach. The explicit conventions below take precedence where they apply.

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

### Plotting functions

Accept an optional `ax` parameter (matplotlib `Axes`); create one internally if `None`. Always return the axes object.

### Docstrings

Numpy-style docstrings are required on all functions and enforced by ruff. Renderable `Examples` sections are best practice but never mandatory: plotting functions are **highly encouraged** to add a `.. plot::` example, other public functions **encouraged** to add an `.. exec_code::` example where the output is illustrative. See the "Example code and results" section of `docs/source/development.rst` for the directives, pre-code fixtures, and render-checking.

### New submodules

A new file in `tools/`, `plotting/`, or `utils/` must be registered in the `__all__` list of the parent `__init__.py`.

### Settings

Functions should use `sctoolbox.settings` for defaults (threads, file paths). Allow a parameter to override the setting where appropriate.

### Deprecation

Mark deprecated functions with the `deprecation` package, targeting removal in 2 minor versions. Add `@deprecation.fail_if_not_removed` to the function's test.

### Notebooks

- Outputs must be cleared before committing (`.gitconfig` in the repo handles this automatically when the sctoolbox env is active).
- Location: RNA-specific → `rna_analysis/notebooks/`; ATAC-specific → `atac_analysis/notebooks/`; data-agnostic → `general_notebooks/`.
- Naming: numeric prefix for ordered notebooks (`01_`, `02_`, ...); letter prefix for optional (`0A_`, `0B_`); `99-report.ipynb` always last.
- Input cells must be colored `PowderBlue` via `bgcolor()` in the hidden init cell.
- Output files: numeric prefix, plots in PDF format.

### CHANGES.md

Every change to the package, notebooks, or documentation requires an entry in `CHANGES.md`. Package- and notebook-scope changes are enforced by CI; docs-only changes are not gated by CI, but the dev workflow still requires an entry. Docs entries go under the main section header (like package changes). Format:

```markdown
## 0.xy.z (in progress)
- description of change (#<issue number>)

### Changes to notebooks
- description (#<issue number>)
```

## Agentic workflow

Skills live in `.claude/skills/`. The development loop is:

```
/design  →  /plan  →  /implement
```

The user need not type `/design` to start. When a request implies a **non-trivial change** — a new submodule or public function, edits spanning several files, anything that should carry tests + a `CHANGES.md` entry, or behaviour changes that want the regression gate and per-task commits — proactively offer to begin the workflow at `/design` before writing code, and proceed into it only if the user agrees. For **trivial work** — a one-line fix, a typo or docstring tweak, a single localized edit, or an exploratory question — skip the workflow and handle it directly (mention the option at most in passing). When in doubt, name the choice and let the user decide.

Design artifacts (`design.md`, `plan.md`, `review-plan.md`, `review-code.md`) are stored in `.work/<YYYY-MM-DD>-<slug>/`. Each `design.md` records the change **scope** (one or more of `package` / `notebooks` / `docs`), the **conda environment** name, the **commit mode**, and the **autonomy** level (`per-task` / `end`); `plan.md` carries these forward to declare the binding test command, which always starts with `ruff check`. A `docs` scope gates on the Sphinx build (`make -C docs html`) instead of pytest; when a change spans several areas the relevant gates are chained.

**Commit mode** is chosen at the start of `/design` (or asked at `/implement` if unset): `manual` (default — the user stages and commits everything; the workflow never touches git) or `claude (Name <email>)` (the workflow commits per task with that identity). The exact staging and per-commit-identity mechanics live in `.claude/skills/sys-commit/SKILL.md`. The default is `manual`, since the local setup can have issues with automatic commits.

`.work/` is **gitignored — a local-only audit trail**, so `/design` and `/plan` make no commit; code commits begin in `/implement` and stage only code, tests, and `CHANGES.md`. Every code commit carries the `<slug>`, so `git log --grep=<slug>` recovers a work item's full *code* history (the design/plan/review prose stays on disk under `.work/`).
