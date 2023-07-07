import matplotlib.pyplot as plt
import sctoolbox.utils.decorator as deco


@deco.log_anndata
def violin_HVF_distribution(adata):
    """
    plot the distribution of the HVF
    :param adata:
    :return:
    """
    # get the number of cells per highly variable feature
    hvf_var = adata.var[adata.var['highly_variable']]  # 'highly_variable' is a boolean column
    n_cells = hvf_var['n_cells_by_counts']
    n_cells.reset_index(drop=True, inplace=True)
    # violin plot
    fig, ax = plt.subplots()
    ax.violinplot(n_cells, showmeans=True, showmedians=True)
    ax.set_title('Distribution of the number of cells per highly variable feature')
    ax.set_ylabel('Number of cells')
    ax.set_xlabel('Highly variable features')
    plt.show()


@deco.log_anndata
def scatter_HVF_distribution(adata):
    """
    plot the distribution of the HVF
    :param adata:
    :return:
    """
    variabilities = adata.var[['variability_score', 'n_cells']]
    fig, ax = plt.subplots()
    ax.scatter(variabilities['n_cells'], variabilities['variability_score'])
    ax.set_title('Distribution of the number of cells and variability score per feature')
    ax.set_xlabel('Number of cells')
    ax.set_ylabel('variability score')
    plt.show()
