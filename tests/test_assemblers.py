import os
import anndata
import sctoolbox.assemblers


def test_from_single_starsolo():
    """ Test anndata assembly from starsolo output. """
    solo_dir = os.path.join(os.path.dirname(__file__), 'data', 'solo')
    adata = sctoolbox.assemblers.from_single_starsolo(solo_dir, dtype="filtered")

    assert isinstance(adata, anndata.AnnData)


def test_from_single_mtx():
    """ Test anndata assembly from mtx file. """
    mtx_filename = os.path.join(os.path.dirname(__file__), 'data', 'solo', 'Gene', 'filtered', 'matrix.mtx')
    barcode_filename = os.path.join(os.path.dirname(__file__), 'data', 'solo', 'Gene', 'filtered', 'barcodes.tsv')
    gene_filename = os.path.join(os.path.dirname(__file__), 'data', 'solo', 'Gene', 'filtered', 'genes.tsv')

    adata = sctoolbox.assemblers.from_single_mtx(mtx_filename, barcode_filename, gene_filename, is_10X=False)

    assert isinstance(adata, anndata.AnnData)
