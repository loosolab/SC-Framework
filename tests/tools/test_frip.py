"""Test the frip score calculation function."""
import sctoolbox.tools as tools


# ------------------------------ TESTS --------------------------------- #


def test_calc_frip_scores(adata_atac, sorted_fragments):
    """Test the calc_frip_scores function."""
    assert 'frip' not in adata_atac.obs.columns

    adata_atac, total_frip = tools.frip.calc_frip_scores(adata_atac, sorted_fragments, temp_dir='')

    assert 'frip' in adata_atac.obs.columns
