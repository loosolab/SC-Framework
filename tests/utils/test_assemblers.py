"""Test functions to assemble adata objects."""

import os
import re
import anndata
import logging
import numpy as np
import pytest
import sctoolbox.utils.assemblers as assemblers
import sctoolbox.utils.general as general
import scanpy as sc
from tests.conftest import DATA_DIR


# --------------------------- FIXTURES ------------------------------ #


@pytest.fixture()
def h5ad_file1():
    """Return path to h5ad file.

    Returns
    -------
    str
        Path to h5ad file.
    """
    return os.path.join(DATA_DIR, 'adata.h5ad')


@pytest.fixture()
def h5ad_file2():
    """Return path to h5ad file.

    Returns
    -------
    str
        Path to h5ad file.
    """
    return os.path.join(DATA_DIR, 'scsa', 'adata_scsa.h5ad')


@pytest.fixture(scope="session")
def rds_file(tmp_path_factory):
    """Build a small Seurat .rds from a scanpy dataset and return its path.

    A small AnnData (slice of ``sc.datasets.pbmc68k_reduced``) is converted to a
    Seurat object via the same rpy2 + anndata2ri path that ``from_R`` uses in
    reverse, then serialized with ``saveRDS``. The Seurat ``RNA`` assay lets
    ``from_R`` read it back with ``layer=None`` and ``layer="RNA"``. The file is
    read-only and reused across the parametrized cases, hence session scope.

    Returns
    -------
    str
        Path to the generated .rds file.
    """
    # small, scanpy-backed AnnData; the from_R assertion is type-level only
    adata = sc.datasets.pbmc68k_reduced()[:50, :100].copy()
    adata.X = np.asarray(adata.X)
    # Seurat requires a non-negative counts assay
    adata.layers['counts'] = np.abs(np.rint(adata.X)).astype('float32')

    # set up the R <-> python interface (same entry point convertToAdata uses)
    general.setup_R(None)
    import anndata2ri
    from rpy2.robjects import r, default_converter, conversion, globalenv

    # AnnData -> SingleCellExperiment via the anndata2ri converter
    with conversion.localconverter(anndata2ri.converter):
        globalenv['sce'] = adata

    out_path = str(tmp_path_factory.mktemp('rds') / 'adata_rna.rds')
    with conversion.localconverter(default_converter):
        r('suppressPackageStartupMessages(library(SingleCellExperiment))')
        r('suppressPackageStartupMessages(library(Seurat))')
        # name the main experiment "RNA" so the resulting Seurat assay is "RNA"
        r('mainExpName(sce) <- "RNA"')
        # SCE -> Seurat so convertToAdata's UpdateSeuratObject/as.SingleCellExperiment path applies
        r('srt <- as.Seurat(sce, counts = "counts", data = "X")')
        globalenv['out_path'] = out_path
        r('saveRDS(srt, out_path)')
        # drop the temporary names we bound so we leave no state in the R session
        r('rm(sce, srt, out_path)')

    return out_path

# --------------------------- TESTS --------------------------------- #


@pytest.mark.parametrize("files", [
    "h5ad_file1",
    ["h5ad_file1", "h5ad_file2"],
    {"a": "h5ad_file1", "b": "h5ad_file2"}
])
def test_from_h5ad(files, request):
    """Test the from_h5ad function."""
    # enable fixture in parametrize https://engineeringfordatascience.com/posts/pytest_fixtures_with_parameterize/
    if isinstance(files, list):
        files = [request.getfixturevalue(f) for f in files]
    elif isinstance(files, dict):
        files = {k: request.getfixturevalue(v) for k, v in files.items()}
    else:
        files = request.getfixturevalue(files)

    assert isinstance(assemblers.from_h5ad(files), sc.AnnData)


@pytest.mark.parametrize("fixture, expected, coordinate_cols",
                         [("adata_atac", True, ["chr", "start", "end"]),  # expects var tables to be unchanged
                          ("adata_atac_emptyvar", KeyError, ["chr", "start", "end"]),
                          # expects var tables to be changed
                          ("adata", KeyError, ["chr", "start", "end"]),
                          # expects a valueerror due to missing columns
                          ("adata_atac_invalid", False, ["chr", "start", "end"]),
                          ("named_var_adata", True, 'coordinate_col')])  # expects coordinate_col to be parsed into a valid chr:start-stop var index
def test_prepare_atac_anndata(fixture, expected, coordinate_cols, request):
    """Test prepare_atac_anndata success."""

    adata_orig = request.getfixturevalue(fixture)  # fix for using fixtures in parametrize
    adata_cp = adata_orig.copy()  # make a copy to avoid changing the fixture

    expected_coordinates = ['chr', 'start', 'end']
    index_pattern = r"^(chr[0-9XYM]+)[\_\:\-]+[0-9]+[\_\:\-]+[0-9]+$"

    if isinstance(expected, type):
        with pytest.raises(expected):
            assemblers.prepare_atac_anndata(adata_cp, coordinate_cols=coordinate_cols)

    else:
        assemblers.prepare_atac_anndata(adata_cp, coordinate_cols=coordinate_cols)
        # check for the existence of the coordinate columns ['chr','start','end'] in the var table
        assert all(item in adata_cp.var.columns for item in expected_coordinates)

        # check if the first var index is in the correct format
        assert bool(re.fullmatch(index_pattern, adata_cp.var.index[0])) is True


def test_prepare_atac_anndata_default_coordinates(adata_atac, caplog, add_logger_handler):
    """Test that prepare_atac_anndata leaves valid default coordinate columns untouched."""

    adata = adata_atac.copy()
    var_before = adata.var.copy()

    with caplog.at_level(logging.INFO), add_logger_handler(assemblers.logger, caplog.handler):
        assemblers.prepare_atac_anndata(adata, coordinate_cols=None)

    assert list(adata.var.columns) == ['chr', 'start', 'end']
    assert np.array_equal(var_before.values, adata.var.values)  # the var table is not reformatted
    assert not [msg for _, level, msg in caplog.record_tuples if level == logging.WARNING]


def test_from_single_starsolo():
    """Test from_single_starsolo success."""

    SOLO_DIR = os.path.join(DATA_DIR, 'solo')
    adata = assemblers.from_single_starsolo(SOLO_DIR, dtype="filtered", header=None)

    assert isinstance(adata, anndata.AnnData)


def test_from_mtx_path():
    """Test from_mtx success with path as input."""

    # With variable file
    adata = assemblers.from_mtx(os.path.join(DATA_DIR, 'solo', 'Gene', 'filtered'))
    adata2 = assemblers.from_mtx(os.path.join(DATA_DIR, 'solo', 'Gene', 'filtered'),
                                 variables="*notfound.tsv",
                                 var_error=False)

    assert isinstance(adata, anndata.AnnData)
    assert isinstance(adata2, anndata.AnnData)


def test_from_mtx_dict():
    """Test from_mtx success with dict as input."""
    parent_path = os.path.join(os.path.dirname(__file__), '../data', 'solo', 'Gene', 'filtered')

    adata = assemblers.from_mtx(path={
        "cond_1": (os.path.join(parent_path, "matrix.mtx"), os.path.join(parent_path, "barcodes.tsv"), os.path.join(parent_path, "genes.tsv")),
        "cond_2": (os.path.join(parent_path, "matrix.mtx"), os.path.join(parent_path, "barcodes.tsv"))
    })

    assert isinstance(adata, anndata.AnnData)


def test_from_mtx_fail():
    """Test from_mtx fail."""

    with pytest.raises(ValueError):
        assemblers.from_mtx(os.path.join(DATA_DIR, 'solo', 'Gene', 'filtered'),
                            variables="*notfound.tsv")

    with pytest.raises(ValueError):
        assemblers.from_mtx(os.path.join(DATA_DIR, 'solo', 'Gene', 'filtered'),
                            barcodes="*notfound.tsv")

    with pytest.raises(ValueError):
        assemblers.from_mtx("./notfound/")


def test_from_single_mtx():
    """Test from_single_mtx success."""

    MTX_FILENAME = os.path.join(DATA_DIR, 'solo', 'Gene', 'filtered', 'matrix.mtx')
    BARCODES_FILENAME = os.path.join(DATA_DIR, 'solo', 'Gene', 'filtered', 'barcodes.tsv')
    GENES_FILENAME = os.path.join(DATA_DIR, 'solo', 'Gene', 'filtered', 'genes.tsv')

    # test full adata (matrix, barcodes, genes)
    adata = assemblers.from_single_mtx(MTX_FILENAME, BARCODES_FILENAME, GENES_FILENAME, header=None)
    assert isinstance(adata, anndata.AnnData)

    # test partial adata (matrix, barcodes)
    adata = assemblers.from_single_mtx(MTX_FILENAME, BARCODES_FILENAME, header=None)
    assert isinstance(adata, anndata.AnnData)


@pytest.mark.parametrize("files, layer", [
    ("rds_file", None),
    ("rds_file", "RNA"),
    (["rds_file", "rds_file"], None),
    (["rds_file", "rds_file"], "RNA"),
    (["rds_file", "rds_file"], ["RNA", "RNA"]),
    ({"a": "rds_file", "b": "rds_file"}, None),
    ({"a": "rds_file", "b": "rds_file"}, "RNA"),
    ({"a": "rds_file", "b": "rds_file"}, {"a": "RNA", "b": "RNA"})
])
def test_from_R(files, layer, request):
    """Test from_R success."""

    # enable fixture in parametrize https://engineeringfordatascience.com/posts/pytest_fixtures_with_parameterize/
    if isinstance(files, list):
        files = [request.getfixturevalue(f) for f in files]
    elif isinstance(files, dict):
        files = {k: request.getfixturevalue(v) for k, v in files.items()}
    else:
        files = request.getfixturevalue(files)

    assert isinstance(assemblers.from_R(files, layer=layer), sc.AnnData)


@pytest.mark.parametrize("files, layer", [
    ("rds_file", ["RNA", "RNA"]),
    (["rds_file", "rds_file"], ["RNA"]),
    (["rds_file", "rds_file"], ["RNA", "RNA", "RNA"]),
    ({"a": "rds_file", "b": "rds_file"}, {"a": "RNA"})
])
def test_from_R_fail(files, layer, request):
    """Test from_R fail."""

    # enable fixture in parametrize https://engineeringfordatascience.com/posts/pytest_fixtures_with_parameterize/
    if isinstance(files, list):
        files = [request.getfixturevalue(f) for f in files]
    elif isinstance(files, dict):
        files = {k: request.getfixturevalue(v) for k, v in files.items()}
    else:
        files = request.getfixturevalue(files)

    with pytest.raises(ValueError):
        assemblers.from_R(files, layer=layer)
