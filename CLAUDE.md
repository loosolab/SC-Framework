# SC-Framework — Claude context

## Project

Python package (`sctoolbox`) for single-cell analysis workflows covering scRNA-seq and scATAC-seq data. Distributed as `SC-Framework` on PyPI. Maintained at MPI Bad Nauheim (Looso lab).

## Stack

- **Language:** Python >=3.9, <3.13
- **Core deps:** AnnData, Scanpy (+ leiden community detection)
- **Optional dep groups:** `atac`, `batch_correction`, `receptor_ligand`, `pseudotime`, `gsea`, `deseq2`, `annotation`, `proportion`, `velocity`, `palantir`, `converter`, `interactive`
- **Test framework:** pytest + pytest-cov + pytest-html (target coverage >90%)
- **Linter:** ruff (config in `pyproject.toml`; targets `src/**/*.py`, `scripts/*.py`, `tests/**/*.py`)
- **Notebooks:** Jupyter (rna_analysis/notebooks/, atac_analysis/notebooks/, general_notebooks/)
- **Build:** setuptools + setuptools_scm
- **Runtime type checking:** beartype (>=0.18.2)

## Module map

```
src/sctoolbox/
  tools/
    amulet.py           # doublet detection (ATAC)
    bam.py              # BAM file utilities
    calc_overlap_fc.py  # overlap fold-change calculations
    celltype_annotation.py
    clustering.py
    dim_reduction.py
    download_data.py
    embedding.py
    frip.py             # FRiP score (ATAC)
    gene_correlation.py
    gsea.py
    highly_variable.py
    insertsize.py       # insert size distribution (ATAC)
    marker_genes.py
    multiomics.py
    norm_correct.py
    palantir_analysis.py
    peak_annotation.py  # ATAC peak annotation
    qc_filter.py
    receptor_ligand.py
    report.py
    tobias.py           # TOBIAS TF footprinting
    tsse.py             # TSS enrichment score (ATAC)
  plotting/
    clustering.py
    embedding.py
    general.py
    genometracks.py
    gsea.py
    highly_variable.py
    marker_genes.py
    multiomics.py
    planet_plot.py
    qc_filter.py
    velocity.py
  utils/
    adata.py
    assemblers.py
    bioutils.py
    checker.py
    cli.py
    creators.py
    decorator.py        # @log_anndata and @beartype utilities
    general.py
    io.py
    jupyter.py
    multiprocessing.py
    tables.py
  _settings.py          # global SCtoolboxConfig; access via sctoolbox.settings
  _modules.py           # lazy module loader
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

## Conventions

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

Numpy-style docstrings are required on all functions and enforced by ruff. Add an `Examples` section to plotting functions.

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

Every change to the package or notebooks requires an entry in `CHANGES.md` (enforced by CI). Format:

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

Design artifacts are stored in `.work/<YYYY-MM-DD>-<slug>/`. Each `design.md` records the change **scope** (package / notebooks / both) and the **conda environment** name; `plan.md` uses these to declare the binding test command, which always starts with `ruff check`.
