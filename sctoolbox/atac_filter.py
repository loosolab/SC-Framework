import scanpy as sc
import sctoolbox.qc_filter as qc_filter


def filter_adata(adata, manual_thresholds, automatic_thresholds=True):
    """
    filter the adata object based on the thresholds
    :param adata:
    :param manual_thresholds:
    :param automatic_thresholds:
    :return:
    """
    if automatic_thresholds:
        keys = list(manual_thresholds.keys())
        automatic_thresholds = qc_filter.automatic_thresholds(adata, columns=keys)
        qc_filter.apply_qc_thresholds(adata, automatic_thresholds)
    else:
        # thresholds which are not set by the user are set automatically
        for key, value in manual_thresholds.items():
            if value['min'] is None or value['max'] is None:
                auto_thr = qc_filter.automatic_thresholds(adata, columns=[key])
                manual_thresholds[key] = auto_thr[key]
        qc_filter.apply_qc_thresholds(adata, manual_thresholds)


def filter_mis(adata, upper_threshold=160, lower_threshold=80, col="mean_insertsize"):
    """
    filter the mean fragment length
    :param adata:
    :param upper_threshold:
    :param lower_threshold:
    :return:
    """
    adata = adata[adata.obs[col] < upper_threshold, :]
    adata = adata[adata.obs[col] > lower_threshold, :]

    return adata


def filter_pct_fragments_in_promoters(adata, upper_threshold=0.5, lower_threshold=0.1, col="pct_fragments_in_promoters"):
    """
    filter the percentage of fragments in promoters
    :param adata:
    :param upper_threshold:
    :param lower_threshold:
    :return:
    """
    adata = adata[adata.obs[col] < upper_threshold, :]
    adata = adata[adata.obs[col] > lower_threshold, :]

    return adata


def filter_n_fragments_in_promoters(adata, upper_threshold=100000, lower_threshold=10000, col="n_fragments_in_promoters"):
    """
    filter the number of fragments in promoters
    :param adata:
    :param upper_threshold:
    :param lower_threshold:
    :return:
    """
    adata = adata[adata.obs[col] < upper_threshold, :]
    adata = adata[adata.obs[col] > lower_threshold, :]

    return adata


def filter_n_fragments(adata, upper_threshold=1000000, lower_threshold=100000, col="TN"):
    """
    filter the number of total fragments per barcode
    :param adata:
    :param upper_threshold:
    :param lower_threshold:
    :return:
    """
    adata = adata[adata.obs[col] < upper_threshold, :]
    adata = adata[adata.obs[col] > lower_threshold, :]

    return adata


def filter_chrM_fragments(adata, upper_threshold=1000, lower_threshold=0, col="UM"):
    """
    filter the number of total fragments per barcode
    :param adata:
    :param upper_threshold:
    :param lower_threshold:
    :return:
    """
    adata = adata[adata.obs[col] < upper_threshold, :]
    adata = adata[adata.obs[col] > lower_threshold, :]

    return adata


if __name__ == '__main__':

    import plotting_atac as pa
    adata = sc.read_h5ad("/home/jan/python-workspace/sc-atac/processed_data/cropped_146/assembling/anndata/cropped_146.h5ad")
    pa.plot_obs_violin(adata, ["mean_fragment_length"])
    adata = filter_mis(adata, upper_threshold=160, lower_threshold=85)
    pa.plot_obs_violin(adata, ["mean_fragment_length"])
