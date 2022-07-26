import pandas as pd
import scanpy as sc
import sctoolbox.creators as cr


def add_cellxgene_annotation(adata, csv):
    """ Add columns from cellxgene annotation to the adata .obs table.
    Parameters
    ------------
    adata : anndata object
        The adata object to add annotations to.
    csv : str
        Path to the annotation file from cellxgene containing cell annotation.
    Returns
    --------
    None - the annotation is added to adata in place.
    """
    anno_table = pd.read_csv(csv, sep=",", comment='#')
    anno_table.set_index("index", inplace=True)
    anno_name = anno_table.columns[-1]
    adata.obs.loc[anno_table.index, anno_name] = anno_table[anno_name].astype('category')


def annot_HVG(anndata):
    '''
    Annotate highly variable genes
    Parameters
    ----------
    anndata : anndata object
        adata object
    Return
    ----------
        Anndata.var["highly_variable"]
    '''
    # Messages and others
    repeat, repeat_limit, HVG_list, min_mean = 0, 10, list(), 0.0125 # Default for sc.pp.highly_variable_genes

    # Finding the highly variable genes
    print("Annotating highy variable genes (HVG)")
    sc.pp.highly_variable_genes(anndata, min_mean=min_mean)
    HVG = sum(anndata.var.highly_variable)
    HVG_list.append(HVG)
    while HVG < 1000 or HVG > 5000: # These 1K and 5K limits were proposed by DOI:10.15252/msb.20188746
        if HVG < 1000:
            min_mean = min_mean / 10
        if HVG > 5000:
            min_mean = min_mean * 10
        sc.pp.highly_variable_genes(anndata, min_mean=min_mean)
        HVG = sum(anndata.var.highly_variable)
        if HVG == HVG_list[-1]:
            repeat = repeat + 1
        if repeat == repeat_limit:
            break
        HVG_list.append(HVG)
    #Adding info in anndata.uns["infoprocess"]
    cr.build_infor(anndata, "Scanpy annotate HVG", "min_mean= " + str(min_mean) + "; Total HVG= " + str(HVG_list[-1]))
    return anndata.copy()
