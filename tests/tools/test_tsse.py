"""Test the tsse_score related functions."""
import numpy as np
import sctoolbox.tools as tools
import os


# ------------------------------ TESTS --------------------------------- #


def test_write_TSS(atac_gtf, tmp_path):
    """Test write_TSS function."""
    tss_file = str(tmp_path / "mm10_genes_tss.bed")
    tss_list, tempfiles = tools.tsse.write_TSS_bed(atac_gtf, tss_file, temp_dir=str(tmp_path))

    # Check if file exists
    assert os.path.exists(tss_file)
    # Check if file is not empty
    assert os.path.getsize(tss_file) > 0
    # Check if file has 3 columns
    assert np.loadtxt(tss_file, dtype=str).shape[1] == 3
    # Check if the tss_list columns are in the expected format
    assert tss_list[0][0] == 'chr1'
    assert type(tss_list[0][1]) is int
    assert type(tss_list[0][2]) is int


def test_overlap_and_aggregate(atac_gtf, atac_fragments, tmp_path):
    """Test overlap_and_aggregate function."""
    tss_file = str(tmp_path / "mm10_genes_tss.bed")
    overlap = str(tmp_path / "overlap.bed")

    # Write TSS file
    tss_list, temp = tools.tsse.write_TSS_bed(atac_gtf, tss_file, temp_dir=str(tmp_path))

    # overlap_and_aggregate
    agg, temp = tools.tsse.overlap_and_aggregate(atac_fragments, tss_file, overlap, tss_list)

    # Check if agg is a dictionary
    assert isinstance(agg, dict)
    # check if overlap file exists
    assert os.path.exists(overlap)
    # Check if file has 5 columns
    assert np.loadtxt(overlap, dtype=str).shape[1] == 5


def test_add_tsse_score(adata_atac, atac_fragments, atac_gtf, tmp_path):
    """Test add_tsse_score function."""
    assert 'tsse_score' not in adata_atac.obs.columns

    adata_atac = tools.tsse.add_tsse_score(adata_atac,
                                            atac_fragments,
                                            atac_gtf,
                                      negativ_shift=2000,
                                      positiv_shift=2000,
                                      edge_size_total=100,
                                      edge_size_per_base=50,
                                      min_bias=0.01,
                                      keep_tmp=False,
                                      temp_dir=str(tmp_path))

    assert 'tsse_score' in adata_atac.obs.columns


def test_tsse_scoring(atac_fragments, atac_gtf, tmp_path):
    """Test the tsse_scoring function."""

    tSSe_df = tools.tsse.tsse_scoring(atac_fragments,
                                      atac_gtf,
                                      negativ_shift=2000,
                                      positiv_shift=2000,
                                      edge_size_total=100,
                                      edge_size_per_base=50,
                                      min_bias=0.01,
                                      keep_tmp=False,
                                      temp_dir=str(tmp_path),
                                      plot=True)

    assert all(tSSe_df.columns.isin(['TSS_agg', 'total_ov', 'tsse_score']))
    assert isinstance(tSSe_df['TSS_agg'][0], np.ndarray)
    assert isinstance(tSSe_df['total_ov'][0], np.int64)
    assert isinstance(tSSe_df['tsse_score'][0], np.float64)
