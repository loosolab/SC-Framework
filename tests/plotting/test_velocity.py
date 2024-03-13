import pytest
import sctoolbox.plotting as pl


@pytest.mark.parametrize("sortby, title, figsize, layer",
                         [("condition", "condition", None, "spliced"),
                          (None, None, (4, 4), None)],
                         )
def test_pseudotime_heatmap(adata, sortby, title, figsize, layer):
    """Test pseudotime_heatmap success."""
    ax = pl.pseudotime_heatmap(adata, ['ENSMUSG00000103377',
                                       'ENSMUSG00000102851'],
                               sortby=sortby, title=title,
                               figsize=figsize, layer=layer)
    ax_type = type(ax).__name__
    assert ax_type.startswith("Axes")