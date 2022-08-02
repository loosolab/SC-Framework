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


def annot_HVG(anndata, min_mean=0.0125, max_iterations=10, inplace=True, **kwargs):
    """
    Annotate highly variable genes (HVG). Tries to annotate between 1,000 and 5,000 HVGs, by gradually in-/ decreasing min_mean of scanpy.pp.highly_variable_genes.
    Default limits are chosen as proposed by https://doi.org/10.15252/msb.20188746.

    Note: Logarithmized data is expected.

    Parameters
    ----------
    anndata : anndata.AnnData
        Anndata object to annotate.
    min_mean : float, default 0.0125
        Starting min_mean parameter for finding HVGs.
    max_iterations : int, default 10
        Maximum number of min_mean adjustments.
    inplace : boolean, default False
        Whether the anndata object is modified inplace.
    **kwargs :
        Additional arguments forwarded to scanpy.pp.highly_variable_genes().

    Returns
    -------
    anndata.Anndata or None:
        Adds annotation of HVG to anndata object. Information is added to Anndata.var["highly_variable"].
    """
    adata_m = anndata if inplace else anndata.copy()

    print("Annotating highy variable genes (HVG)")

    # adjust min_mean to get a HVG count in a certain range
    for i in range(max_iterations + 1):
        sc.pp.highly_variable_genes(adata_m, min_mean=min_mean, inplace=True, **kwargs)

        # counts True values in column
        hvg_count = sum(adata_m.var.highly_variable)

        # adjust min_mean
        # skip adjustment if in last iteration
        if i < max_iterations and hvg_count < 1000:
            min_mean /= 10
        elif i < max_iterations and hvg_count > 5000:
            min_mean *= 10
        else:
            break

    # Adding info in anndata.uns["infoprocess"]
    cr.build_infor(anndata, "Scanpy annotate HVG", "min_mean= " + str(min_mean) + "; Total HVG= " + str(hvg_count), inplace=True)

    if not inplace:
        return adata_m
