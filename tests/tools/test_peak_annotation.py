"""Test functions related to peak annotation required by scATAC-seq."""

import argparse
import pytest
import sctoolbox.tools as anno
import scanpy as sc
import os


# ------------------------- Fixtures ------------------------- #

uropa_config = {"queries": [{"distance": [10000, 1000]}]}


@pytest.fixture
def adata_atac():
    """Load atac anndata."""
    adata_f = os.path.join(os.path.dirname(__file__), '..', 'data', 'atac', 'mm10_atac.h5ad')
    return sc.read_h5ad(adata_f)


# ------------------------- Tests ------------------------- #


@pytest.mark.parametrize("inplace, threads, config, best, coordinate_cols",
                         [(True, 1, None, True, None),
                          (False, 2, uropa_config, False, ["chr", "start", "stop"])])
def test_annotate_adata(adata_atac, inplace, threads, config, best, coordinate_cols):
    """Test annotate_adata success."""

    adata_atac.var["distance_to_gene"] = 100  # initialize distance column to test the warning message
    gtf_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'atac', 'chr4_mm10_genes.gtf')

    out = anno.annotate_adata(adata_atac, gtf=gtf_path, threads=threads, inplace=inplace,
                              config=config, best=best, coordinate_cols=coordinate_cols)

    if inplace:
        assert out is None
        assert 'gene_id' in adata_atac.var.columns
    else:
        assert type(out).__name__ == 'AnnData'
        assert 'gene_id' in out.var.columns


@pytest.mark.parametrize("config", [None, uropa_config])
def test_annotate_narrowPeak(config):
    """Test annotate_narrowPeak success."""

    gtf_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'atac', 'mm10_genes.gtf')
    peaks_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'atac', 'cropped_testing.narrowPeak')

    annotation_table = anno.annotate_narrowPeak(peaks_path, gtf=gtf_path, config=config)

    assert 'gene_id' in annotation_table

# ------------------------- Tests for gtf formats ------------------------- #


gtf_files = {"noheader": os.path.join(os.path.dirname(__file__), '..', 'data', 'atac', 'mm10_genes.gtf'),
             "header": os.path.join(os.path.dirname(__file__), '..', 'data', 'atac', 'gtf_testdata', 'cropped_gencode.v41.gtf'),
             "unsorted": os.path.join(os.path.dirname(__file__), '..', 'data', 'atac', 'gtf_testdata', 'cropped_gencode.v41.unsorted.gtf'),
             "gtf_gz": os.path.join(os.path.dirname(__file__), '..', 'data', 'atac', 'gtf_testdata', 'cropped_gencode.v41.gtf.gz'),
             "gtf_missing_col": os.path.join(os.path.dirname(__file__), '..', 'data', 'atac', 'gtf_testdata', 'cropped_missing_column_gencode.v41.gtf'),
             "gtf_corrupted": os.path.join(os.path.dirname(__file__), '..', 'data', 'atac', 'gtf_testdata', 'cropped_corrupted_format_gencode.v41.gtf'),
             "gff": os.path.join(os.path.dirname(__file__), '..', 'data', 'atac', 'gtf_testdata', 'cropped_gencode.v41.gff3')}


# indirect test of gtf_integrity as well
@pytest.mark.parametrize("key, gtf", [(key, gtf_files[key]) for key in gtf_files])
def test_prepare_gtf(key, gtf):
    """Test _prepare_gtf success and failure."""

    if key in ["noheader", "header", "unsorted", "gtf_gz"]:  # these gtfs are valid and can be read
        gtf_out, tempfiles = anno._prepare_gtf(gtf, "")

        assert os.path.exists(gtf_out)  # assert if output gtf exists as a file

    elif key in ["gtf_missing_col", "gtf_corrupted", "gff"]:  # these gtfs are invalid and should raise an error

        with pytest.raises(argparse.ArgumentTypeError) as err:
            anno._prepare_gtf(gtf, "")

        # Assert if the error message is correct depending on input
        if key == "gtf_missing_col":
            assert err.value.args[0] == 'Number of columns in the gtf file unequal 9'

        elif key == "gtf_corrupted":
            assert err.value.args[0] == 'gtf file is corrupted'

        elif key == "gff":
            assert err.value.args[0] == 'Header in gtf file does not match gtf format'

    else:
        raise ValueError("Invalid key: {}".format(key))
