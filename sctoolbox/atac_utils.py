import scanpy as sc
import sctoolbox.qc_filter as qc_filter
import episcanpy as epi
import anndata
from matplotlib import pyplot as plt


def get_thresholds_atac_wrapper(adata, manual_thresholds, automatic_thresholds=True):
    """
    return the thresholds for the filtering
    :param adata:
    :param manual_thresholds:
    :param automatic_thresholds:
    :return:
    """
    if automatic_thresholds:
        keys = list(manual_thresholds.keys())
        automatic_thresholds = qc_filter.automatic_thresholds(adata, columns=keys)
        return automatic_thresholds
    else:
        # thresholds which are not set by the user are set automatically
        for key, value in manual_thresholds.items():
            if value['min'] is None or value['max'] is None:
                auto_thr = qc_filter.automatic_thresholds(adata, columns=[key])
                manual_thresholds[key] = auto_thr[key]
        return manual_thresholds


def build_legend(adata, key, value, inplace=True):
    """
    Adding info anndata.uns["legend"]
    :param adata:
    :param key:
    :param value:
    :param inplace:
    :return:
    """
    if type(adata) != anndata.AnnData:
        raise TypeError("Invalid data type. AnnData object is required.")

    m_adata = adata if inplace else adata.copy()

    if "legend" not in m_adata.uns:
        m_adata.uns["legend"] = {}
    m_adata.uns["legend"][key] = value

    if not inplace:
        return m_adata


def plot_ov_hist(adata, threshold_features=1000):
    """
    plot as overview of the adata object the coverage of the number of cells and features (peaks)
    :param adata:
    :return:
    """
    # show open features per cell
    min_features = threshold_features

    epi.pp.coverage_cells(adata, binary=True, log=False, bins=50,
                          threshold=min_features)
    # epi.pp.coverage_cells(adata, binary=True, log=10, bins=50, threshold=min_features)

    # show numbers of cells sharing features

    epi.pp.coverage_features(adata, binary=True, log=False, bins=50,
                             threshold=None)
    # epi.pp.coverage_features(adata, binary=True, log=10, bins=50, threshold=None)

    epi.pp.cal_var(adata)


def plot_obs_violin(adata, obs_cols):
    """
    plot violin plots of the obs columns
    :param adata:
    :param obs_cols:
    :return:
    """

    for col in obs_cols:
        sc.pl.violin(adata, col, show=False)

    plt.show()


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

    adata = sc.read_h5ad("/home/jan/python-workspace/sc-atac/processed_data/cropped_146/assembling/anndata/cropped_146.h5ad")
    plot_ov_hist(adata, threshold_features=1000)
    plot_obs_violin(adata, ["mean_fragment_length", "pct_fragments_in_promoters", "n_fragments_in_promoters", "n_total_fragments"])
