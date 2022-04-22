
def cell_mitochondrial():
    '''
    Filter cells based on the mitochondrial content.
    Exclude >= cutoff
            Default cutoff: automatic cutoff from from distribution analysis
            Custom cutoff: user define the cutoff
    Return anndata
    '''

def cell_numgenes():
    '''
    Filter cells based on the number of genes.
    Exclude <= cutoff
            Default cutoff: automatic cutoff from from distribution analysis
            Custom cutoff: user define the cutoff
    Return anndata
    '''

def genes_mitochondrial():
    '''
    Filter mitochondrial genes based on a custom list of genes defined by the co-worker.
    Exclude based on the custom list.
    Return anndata
    '''

def genes_genderspecific():
    '''
    Filter gender-specific genes based on a custom list of genes defined by the co-worker.
    Exclude based on the custom list.
    Return anndata
    '''

def genes_others():
    '''
    Filter other genes based on a custom list of genes defined by the co-worker.
    Exclude based on the custom list.
    Return anndata
    '''

def genes_mincells():
    '''
    Filter genes expressed in few cells.
    Exclude <= cutoff
            Default cutoff: automatic cutoff from from distribution analysis
            Custom cutoff: user define the cutoff
    Return anndata
    '''
