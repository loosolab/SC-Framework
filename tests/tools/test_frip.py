"""Test the frip score calculation function."""
import sctoolbox.tools as tools


# ------------------------------ TESTS --------------------------------- #


def test_calc_frip_scores(adata_atac, sorted_fragments, tmp_path):
    """Test the calc_frip_scores function."""
    assert 'frip' not in adata_atac.obs.columns

    adata_atac, total_frip = tools.frip.calc_frip_scores(adata_atac, sorted_fragments, temp_dir=str(tmp_path))

    assert 'frip' in adata_atac.obs.columns
