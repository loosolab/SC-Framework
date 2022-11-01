import scanpy as sc

def filter_mfl(adata, upper_threshold=160, lower_threshold=80):
    """
    filter the mean fragment length
    :param adata:
    :param upper_threshold:
    :param lower_threshold:
    :return:
    """
    adata = adata[adata.obs["mean_fragment_length"] < upper_threshold, :]
    adata = adata[adata.obs["mean_fragment_length"] > lower_threshold, :]

    return adata

def filter_pct_fragments_in_promoters(adata, upper_threshold=0.5, lower_threshold=0.1):
    """
    filter the percentage of fragments in promoters
    :param adata:
    :param upper_threshold:
    :param lower_threshold:
    :return:
    """
    adata = adata[adata.obs["pct_fragments_in_promoters"] < upper_threshold, :]
    adata = adata[adata.obs["pct_fragments_in_promoters"] > lower_threshold, :]

    return adata

def filter_n_fragments_in_promoters(adata, upper_threshold=100000, lower_threshold=10000):
    """
    filter the number of fragments in promoters
    :param adata:
    :param upper_threshold:
    :param lower_threshold:
    :return:
    """
    adata = adata[adata.obs["n_fragments_in_promoters"] < upper_threshold, :]
    adata = adata[adata.obs["n_fragments_in_promoters"] > lower_threshold, :]

    return adata

def filter_n_total_fragments(adata, upper_threshold=1000000, lower_threshold=100000):
    """
    filter the number of total fragments per barcode
    :param adata:
    :param upper_threshold:
    :param lower_threshold:
    :return:
    """
    adata = adata[adata.obs["n_total_fragments"] < upper_threshold, :]
    adata = adata[adata.obs["n_total_fragments"] > lower_threshold, :]

    return adata

if __name__ == '__main__':

    import plotting_atac as pa
    adata = sc.read_h5ad("/home/jan/python-workspace/sc-atac/processed_data/cropped_146/assembling/anndata/cropped_146.h5ad")
    pa.plot_obs_violin(adata, ["mean_fragment_length"])
    adata = filter_mfl(adata, upper_threshold=160, lower_threshold=85)
    pa.plot_obs_violin(adata, ["mean_fragment_length"])

