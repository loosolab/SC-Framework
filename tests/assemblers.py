import os
import anndata
import sctoolbox.assemblers

def test_from_single_starsolo():
    """ Test anndata assembly from starsolo output. """
    SOLO_DIR = os.path.join(os.path.dirname(__file__), 'data', 'solo')
    adata = sctoolbox.assemblers.from_single_starsolo(SOLO_DIR, dtype="filtered")

    assert isinstance(adata, anndata.AnnData)


def test_from_single_mtx():
    """ Test anndata assembly from mtx file. """
    MTX_FILENAME = os.path.join(os.path.dirname(__file__), 'data', 'solo', 'Gene', 'filtered', 'matrix.mtx')
    BARCODES_FILENAME = os.path.join(os.path.dirname(__file__), 'data', 'solo', 'Gene', 'filtered', 'barcodes.tsv')
    GENES_FILENAME = os.path.join(os.path.dirname(__file__), 'data', 'solo', 'Gene', 'filtered', 'genes.tsv')

    adata = sctoolbox.assemblers.from_single_mtx(MTX_FILENAME, BARCODES_FILENAME, GENES_FILENAME, is_10X=False)

    assert isinstance(adata, anndata.AnnData)
