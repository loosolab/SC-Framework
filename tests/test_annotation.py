import argparse

import pytest
import sctoolbox.annotation as anno
import sctoolbox.utilities as utils
import scanpy as sc
import os


# ------------------------- Fixtures -------------------------#

uropa_config = {"queries": [{"distance": [10000, 1000]}]}


@pytest.fixture
def adata_atac():
    adata_f = os.path.join(os.path.dirname(__file__), 'data', 'atac', 'mm10_atac.h5ad')
    return sc.read_h5ad(adata_f)


@pytest.fixture
def adata_atac_emptyvar(adata_atac):
    adata = adata_atac.copy()
    adata.var = adata.var.drop(columns=adata.var.columns)
    return adata


@pytest.fixture
def adata_atac_invalid(adata_atac):
    adata = adata_atac.copy()
    adata.var.iloc[0, 1] = 500  # start
    adata.var.iloc[0, 2] = 100  # end
    adata.var.reset_index(inplace=True, drop=True)  # remove chromosome-start-stop index
    return adata


@pytest.fixture
def adata_rna():
    adata_f = os.path.join(os.path.dirname(__file__), 'data', 'adata.h5ad')
    return sc.read_h5ad(adata_f)


# ------------------------- Tests ------------------------- #

def test_add_cellxgene_annotation(adata_rna):
    """ Test if 'cellxgene' column is added to adata.obs. """

    csv_f = os.path.join(os.path.dirname(__file__), 'data', 'cellxgene_anno.csv')
    anno.add_cellxgene_annotation(adata_rna, csv_f)

    assert "cellxgene_clusters" in adata_rna.obs.columns


@pytest.mark.parametrize("inplace, threads, config, best, coordinate_cols",
                         [(True, 1, None, True, None),
                          (False, 2, uropa_config, False, ["chr", "start", "stop"])])
def test_annotate_adata(adata_atac, inplace, threads, config, best, coordinate_cols):

    adata_atac.var["distance_to_gene"] = 100  # initialize distance column to test the warning message
    gtf_path = os.path.join(os.path.dirname(__file__), 'data', 'atac', 'mm10_genes.gtf')

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

    gtf_path = os.path.join(os.path.dirname(__file__), 'data', 'atac', 'mm10_genes.gtf')
    peaks_path = os.path.join(os.path.dirname(__file__), 'data', 'atac', 'cropped_testing.narrowPeak')

    annotation_table = anno.annotate_narrowPeak(peaks_path, gtf=gtf_path, config=config)

    assert 'gene_id' in annotation_table


@pytest.mark.parametrize("inplace", [True, False])
def test_annot_HVG(adata_rna, inplace):
    """ Test if 'highly_variable' column is added to adata.var. """

    sc.pp.log1p(adata_rna)
    out = anno.annot_HVG(adata_rna, inplace=inplace)

    if inplace:
        assert out is None
        assert "highly_variable" in adata_rna.var.columns
    else:
        assert "highly_variable" in out.var.columns


def touch_file(f):
    try:
        open(f, 'x')
    except FileExistsError:
        pass


def test_rm_tmp():

    temp_dir = "tempdir"
    utils.create_dir(temp_dir)
    touch_file("tempdir/afile.gtf")
    touch_file("tempdir/tempfile1.txt")
    touch_file("tempdir/tempfile2.txt")

    # Remove tempfile in tempdir (but tempdir should still exist)
    tempfiles = ["tempdir/tempfile1.txt", "tempdir/tempfile2.txt"]
    anno.rm_tmp(temp_dir, tempfiles)

    dir_exists = os.path.exists(temp_dir)
    files_removed = sum([os.path.exists(f) for f in tempfiles]) == 0

    assert dir_exists and files_removed

    # Check that tempdir is removed if it is empty
    anno.rm_tmp(temp_dir)
    dir_exists = os.path.exists(temp_dir)

    assert dir_exists is False


# ------------------------- Tests for gtf formats ------------------------- #

gtf_files = {"noheader": os.path.join(os.path.dirname(__file__), 'data', 'atac', 'mm10_genes.gtf'),
             "header": os.path.join(os.path.dirname(__file__), 'data', 'atac', 'gtf_testdata', 'cropped_gencode.v41.gtf'),
             "unsorted": os.path.join(os.path.dirname(__file__), 'data', 'atac', 'gtf_testdata', 'cropped_gencode.v41.unsorted.gtf'),
             "gtf_gz": os.path.join(os.path.dirname(__file__), 'data', 'atac', 'gtf_testdata', 'cropped_gencode.v41.gtf.gz'),
             "gtf_missing_col": os.path.join(os.path.dirname(__file__), 'data', 'atac', 'gtf_testdata', 'cropped_missing_column_gencode.v41.gtf'),
             "gtf_corrupted": os.path.join(os.path.dirname(__file__), 'data', 'atac', 'gtf_testdata', 'cropped_corrupted_format_gencode.v41.gtf'),
             "gff": os.path.join(os.path.dirname(__file__), 'data', 'atac', 'gtf_testdata', 'cropped_gencode.v41.gff3')}


# indirect test of gtf_integrity as well
@pytest.mark.parametrize("key, gtf", [(key, gtf_files[key]) for key in gtf_files])
def test_prepare_gtf(key, gtf):

    if key in ["noheader", "header", "unsorted", "gtf_gz"]:  # these gtfs are valid and can be read
        gtf_out, tempfiles = anno.prepare_gtf(gtf, "", print)

        assert os.path.exists(gtf_out)  # assert if output gtf exists as a file

    elif key in ["gtf_missing_col", "gtf_corrupted", "gff"]:  # these gtfs are invalid and should raise an error

        with pytest.raises(argparse.ArgumentTypeError) as err:
            anno.prepare_gtf(gtf, "", print)

        # Assert if the error message is correct depending on input
        if key == "gtf_missing_col":
            assert err.value.args[0] == 'Number of columns in the gtf file unequal 9'

        elif key == "gtf_corrupted":
            assert err.value.args[0] == 'gtf file is corrupted'

        elif key == "gff":
            assert err.value.args[0] == 'Header in gtf file does not match gtf format'

    else:
        raise ValueError("Invalid key: {}".format(key))
