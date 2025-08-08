"""sctoolbox tool functions."""

import importlib as _importlib

# define what is exported in this module
__all__ = [
    "bam",
    "calc_overlap_fc",
    "celltype_annotation",
    "clustering",
    "dim_reduction",
    "download_data",
    "embedding",
    "frip",
    "gene_correlation",
    "gsea",
    "highly_variable",
    "insertsize",
    "marker_genes",
    "multiomics",
    "norm_correct",
    "peak_annotation",
    "qc_filter",
    "receptor_ligand",
    "report",
    "tobias",
    "tsse"
]


def __dir__():
    """Return the defined submodules."""
    return __all__


def __getattr__(name):
    """Lazyload modules (inspired by scipy)."""
    if name in __all__:
        # return import to make it directly available
        return _importlib.import_module(f"sctoolbox.tools.{name}")
    else:
        raise AttributeError(f"Module 'sctoolbox.tools' does not contain a module named '{name}'.")
