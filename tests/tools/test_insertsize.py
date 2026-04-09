"""Test add_insertsize function."""

import sctoolbox.tools.insertsize as ins


# ------------------------------ TESTS --------------------------------- #


def test_add_insertsize_fragments(adata_atac, atac_fragments):
    """Test if add_insertsize adds information from a fragmentsfile."""
    assert "insertsize_distribution" not in adata_atac.uns
    assert "mean_insertsize" not in adata_atac.obs.columns

    ins.add_insertsize(adata_atac, fragments=atac_fragments)

    assert "insertsize_distribution" in adata_atac.uns
    assert "mean_insertsize" in adata_atac.obs.columns


def test_add_insertsize_bam(adata_atac, atac_bam_file):
    """Test if add_insertsize adds information from a bamfile."""
    assert "insertsize_distribution" not in adata_atac.uns
    assert "mean_insertsize" not in adata_atac.obs.columns

    ins.add_insertsize(adata_atac, bam=atac_bam_file)

    assert "insertsize_distribution" in adata_atac.uns
    assert "mean_insertsize" in adata_atac.obs.columns
