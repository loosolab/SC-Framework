import os
import anndata
import sctoolbox.assemblers

def test_from_single_mtx():

    MTX_FILENAME = os.path.join(os.path.dirname(__file__), 'data', 'matrix.mtx')
    BARCODES_FILENAME = os.path.join(os.path.dirname(__file__), 'data', 'barcodes.tsv')
    GENES_FILENAME = os.path.join(os.path.dirname(__file__), 'data', 'genes.tsv')

    adata = sctoolbox.assemblers.from_single_mtx(MTX_FILENAME, BARCODES_FILENAME, GENES_FILENAME, is_10X=False)

    assert isinstance(adata, anndata.AnnData)
