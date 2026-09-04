"""Test functions related to peak annotation required by scATAC-seq."""

import argparse
import logging
import pytest
import sctoolbox.tools.peak_annotation as anno
import scanpy as sc
import os
from tests.conftest import ATAC_DATA_DIR


# ------------------------- FIXTURES ------------------------- #


uropa_config = {"queries": [{"distance": [10000, 1000]}]}


# ------------------------- TESTS ------------------------- #


@pytest.mark.parametrize("inplace, threads, config, best, coordinate_cols",
                         [(True, 1, None, True, None),
                          (False, 2, uropa_config, False, ["chr", "start", "end"])])
def test_annotate_adata(adata_atac, inplace, threads, config, best, coordinate_cols):
    """Test annotate_adata success."""

    adata_atac.var["distance_to_gene"] = 100  # initialize distance column to test the warning message
    gtf_path = os.path.join(ATAC_DATA_DIR, 'chr4_mm10_genes.gtf')

    out = anno.annotate_adata(adata_atac, gtf=gtf_path, threads=threads, inplace=inplace,
                              config=config, best=best, coordinate_cols=coordinate_cols)

    if inplace:
        assert out is None
        assert 'gene_id' in adata_atac.var.columns
    else:
        assert isinstance(out, sc.AnnData)
        assert 'gene_id' in out.var.columns


def test_annotate_adata_missing_coordinates(adata_atac_emptyvar):
    """Test annotate_adata failure for an adata.var without coordinate columns."""

    gtf_path = os.path.join(ATAC_DATA_DIR, 'chr4_mm10_genes.gtf')

    with pytest.raises(KeyError, match="fewer than three columns"):
        anno.annotate_adata(adata_atac_emptyvar, gtf=gtf_path)


# TODO(0.18.0): remove together with the 'stop' acceptance in sctoolbox.utils.checker.validate_regions
def test_annotate_adata_deprecated_coordinates(adata_atac_stop, caplog, add_logger_handler):
    """Test annotate_adata success with the deprecated 'stop' coordinate column."""

    gtf_path = os.path.join(ATAC_DATA_DIR, 'chr4_mm10_genes.gtf')

    with caplog.at_level(logging.INFO), add_logger_handler(anno.logger, caplog.handler):
        anno.annotate_adata(adata_atac_stop, gtf=gtf_path, coordinate_cols=["chr", "start", "stop"])

    assert 'gene_id' in adata_atac_stop.var.columns

    # scoped to the deprecation message, as uropa emits warnings of its own
    warnings = [msg for _, level, msg in caplog.record_tuples if level == logging.WARNING and "deprecated" in msg]
    assert len(warnings) == 1
    assert "stop" in warnings[0] and "end" in warnings[0] and "0.18.0" in warnings[0]


@pytest.mark.parametrize("config", [None, uropa_config])
def test_annotate_narrowPeak(config):
    """Test annotate_narrowPeak success."""

    gtf_path = os.path.join(ATAC_DATA_DIR, 'mm10_genes.gtf')
    peaks_path = os.path.join(ATAC_DATA_DIR, 'cropped_testing.narrowPeak')

    annotation_table = anno.annotate_narrowPeak(peaks_path, gtf=gtf_path, config=config)

    assert 'gene_id' in annotation_table

# ------------------------- Tests for gtf formats ------------------------- #


gtf_files = {"noheader": os.path.join(ATAC_DATA_DIR, 'mm10_genes.gtf'),
             "header": os.path.join(ATAC_DATA_DIR, 'gtf_testdata', 'cropped_gencode.v41.gtf'),
             "unsorted": os.path.join(ATAC_DATA_DIR, 'gtf_testdata', 'cropped_gencode.v41.unsorted.gtf'),
             "gtf_gz": os.path.join(ATAC_DATA_DIR, 'gtf_testdata', 'cropped_gencode.v41.gtf.gz'),
             "gtf_missing_col": os.path.join(ATAC_DATA_DIR, 'gtf_testdata', 'cropped_missing_column_gencode.v41.gtf'),
             "gtf_corrupted": os.path.join(ATAC_DATA_DIR, 'gtf_testdata', 'cropped_corrupted_format_gencode.v41.gtf'),
             "gff": os.path.join(ATAC_DATA_DIR, 'gtf_testdata', 'cropped_gencode.v41.gff3')}


# indirect test of gtf_integrity as well
@pytest.mark.parametrize("key, gtf", [(key, gtf_files[key]) for key in gtf_files])
def test_prepare_gtf(key, gtf, tmp_path):
    """
    Test _prepare_gtf success and failure.

    Raises
    ------
    ValueError
        On invalid key, i.e., the file is neither correctly prepared nor one of the expected error messages is given.
    """

    if key in ["noheader", "header", "unsorted", "gtf_gz"]:  # these gtfs are valid and can be read
        gtf_out, tempfiles = anno._prepare_gtf(gtf, str(tmp_path))

        assert os.path.exists(gtf_out)  # assert if output gtf exists as a file

    elif key in ["gtf_missing_col", "gtf_corrupted", "gff"]:  # these gtfs are invalid and should raise an error

        with pytest.raises(argparse.ArgumentTypeError) as err:
            anno._prepare_gtf(gtf, str(tmp_path))

        # Assert if the error message is correct depending on input
        if key == "gtf_missing_col":
            assert err.value.args[0] == 'Number of columns in the gtf file unequal 9'

        elif key == "gtf_corrupted":
            assert err.value.args[0] == 'gtf file is corrupted'

        elif key == "gff":
            assert err.value.args[0] == 'Header in gtf file does not match gtf format'

    else:
        raise ValueError("Invalid key: {}".format(key))
