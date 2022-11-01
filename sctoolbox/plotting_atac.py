import episcanpy as epi
import scanpy as sc
from matplotlib import pyplot as plt

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
    #epi.pp.coverage_features(adata, binary=True, log=10, bins=50, threshold=None)

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

if __name__ == '__main__':

    adata = sc.read_h5ad("/home/jan/python-workspace/sc-atac/processed_data/cropped_146/assembling/anndata/cropped_146.h5ad")
    plot_ov_hist(adata, threshold_features=1000)
    plot_obs_violin(adata, ["mean_fragment_length", "pct_fragments_in_promoters", "n_fragments_in_promoters", "n_total_fragments"])