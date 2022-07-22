import scipy
from scipy import sparse
import numpy as np

import episcanpy as epi
import matplotlib.pyplot as plt
import sctoolbox.utilities


def tfidf_normalization(matrix, tf_type="term_frequency", idf_type="inverse_freq"):
    """ Perform TF-IDF normalization on a sparse matrix.
    The different variants of the term frequency and inverse document frequency are obtained from https://en.wikipedia.org/wiki/Tf-idf.

    Note: this function requires a lot of memory. Another option is to use the ac.pp.tfidf of the muon package.

    Parameters
    -----------
    matrix : scipy.sparse matrix
        The matrix to be normalized.
    tf_type : string, optional
        The type of term frequency to use. Can be either "raw", "term_frequency" or "log". Default: "term_frequency".
    idf_type : string, optional
        The type of inverse document frequency to use. Can be either "unary", "inverse_freq" or "inverse_freq_smooth". Default: "inverse_freq".
    """

    # t - term (peak)
    # d - document (cell)
    # N - count of corpus (total set of cells)

    # Normalize matrix to number of found peaks
    dense = matrix.todense()
    peaks_per_cell = dense.sum(axis=1)  # i.e. the length of the document(number of words)

    # Decide on which Term frequency to use:
    if tf_type == "raw":
        tf = dense
    elif tf_type == "term_frequency":
        tf = dense / peaks_per_cell     # Counts normalized to peaks (words) per cell (document)
    elif tf_type == "log":
        tf = np.log1p(dense)            # for binary documents, this scales with "raw"

    # Decide on the Inverse document frequency to use
    N = dense.shape[0]     # number of cells (number of documents)
    df = dense.sum(axis=0)  # number of cells carrying each peak (number of documents containing each word)

    if idf_type == "unary":
        idf = np.ones(dense.shape[1])  # shape is number of peaks
    elif idf_type == "inverse_freq":
        idf = np.log(N / df)    # each cell has at least one peak (each document has one word), so df is always > 0
    elif idf_type == "inverse_freq_smooth":
        idf = np.log(N / (df + 1)) + 1

    # Obtain TF_IDF
    tf_idf = np.array(tf) * np.array(idf).squeeze()
    tf_idf = sparse.csr_matrix(tf_idf)

    return(tf_idf)


def apply_svd(adata, layer=None):
    """ Singular value decomposition of anndata object.

    Parameters
    -----------
    adata : anndata.AnnData
        The anndata object to be decomposed.
    layer : string, optional
        The layer to be decomposed. If None, the layer is set to "X". Default: None.

    Returns:
    --------
    adata : anndata.AnnData
        The decomposed anndata object containing .obsm, .varm and .uns information.
    """

    if layer is None:
        mat = adata.X
    else:
        mat = adata.layers[layer]

    # SVD
    u, s, v = scipy.sparse.linalg.svds(mat, k=30, which="LM")  # find largest variance

    # u/s/v are reversed in scipy.sparse.linalg.svds:
    s = s[::-1]
    u = np.fliplr(u)
    v = np.flipud(v)

    # Visualize explained variance
    var_explained = np.round(s**2 / np.sum(s**2), decimals=3)

    adata.obsm["X_svd"] = u
    adata.varm["SVs"] = v.T
    adata.uns["svd"] = {"variance": s,
                        "variance_ratio": var_explained}

    # Hack to use the scanpy functions on SVD coordinates
    # adata.obsm["X_pca"] = adata.obsm["X_svd"]
    # adata.varm["PCs"] = adata.varm["SVs"]
    # adata.uns["pca"] = adata.uns["svd"]

    return(adata)


def get_variable_features(adata, min_score=None, show=True, inplace=True):
    """
    Get the highly variable features of anndata object. Adds the columns "highly_variable" and "variability_score" to adata.obs. If show is True, the plot is shown.

    Parameters
    -----------
    adata : anndata.AnnData
        The anndata object containing counts for variables.
    min_score : float, optional
        The minimum variability score to set as threshold. Default: None (automatic)
    show : bool
        Show plot of variability scores and thresholds. Default: True.
    inplace : bool
        If True, the anndata object is modified. Otherwise, a new anndata object is returned. Default: True.

    Returns
    --------
    If inplace is False, the function returns None
    If inplace is True, the function returns an anndata object.
    """
    sctoolbox.utilities.check_module("kneed")
    sctoolbox.utilities.check_module("statsmodels")

    from kneed import KneeLocator
    import statsmodels.api as sm

    if inplace is False:
        adata = adata.copy()

    # Calculate variability
    epi.pp.cal_var(adata, show=False)

    # Set threshold
    if min_score is None:

        # Get input data to fit
        scores = adata.var["variability_score"].sort_values(ascending=False)
        x = np.arange(len(scores))

        # Subset data to reduce computational time
        target = 10000
        step = int(len(scores) / target)
        if step > 0:
            idx_selection = np.arange(len(scores), step=step)
            scores = scores[idx_selection]
            x = x[idx_selection]

        # Smooth using lowess (prevents early finding of knees due to noise)
        scores = sm.nonparametric.lowess(scores, x, return_sorted=False, frac=0.05)

        # Find knee
        kneedle = KneeLocator(x, scores, curve="convex", direction="decreasing", online=False)
        min_score = kneedle.knee_y

    # Set "highly_variable" column in var
    adata.var["highly_variable"] = adata.var["variability_score"] >= min_score
    n_variable = adata.var["highly_variable"].sum()

    # Create plot
    if show is True:
        _, ax = plt.subplots()

        # Plot distribution of scores
        scores = adata.var["variability_score"].sort_values(ascending=False)
        x = np.arange(len(scores))
        ax.plot(x, scores)

        # Horizontal line at knee
        ax.axhline(min_score, linestyle="--", color="r")
        xlim = ax.get_xlim()
        ax.text(xlim[1], min_score, " {0:.2f}".format(min_score), fontsize=12, ha="left", va="center", color="red")

        # Vertical line at knee
        ax.axvline(n_variable, linestyle="--", color="r")
        ylim = ax.get_ylim()
        ax.text(n_variable, ylim[1], " {0:.0f}".format(n_variable), fontsize=12, ha="left", va="bottom", color="red")

        ax.set_xlabel("Ranked features")
        ax.set_ylabel("Variability score")

    # Return the copy of the adata
    if inplace is False:
        return adata
