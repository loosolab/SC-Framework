import scipy
from scipy import sparse
from scipy.sparse.linalg import svds
import numpy as np
import os
import yaml
import pandas as pd
import gzip
import re
import datetime
import copy
from glob import glob

import episcanpy as epi
import scanpy as sc
from anndata import AnnData
import matplotlib.pyplot as plt
import seaborn as sns

import sctoolbox.utilities as utils
import sctoolbox.annotation as anno
import sctoolbox.bam

#from scanpy import logging
#import pkgutil
#from typing import List, Union, Optional, Callable, Iterable


# muon package

def tfidf(data, log_tf=True, log_idf=True, log_tfidf=False, scale_factor=1e4):
    """Transform peak counts with TF-IDF (Term Frequency - Inverse Document Frequency).
    TF: peak counts are normalised by total number of counts per cell.
    DF: total number of counts for each peak.
    IDF: number of cells divided by DF.
    By default, log(TF) * log(IDF) is returned.

    Note: Function is from the muon package.

    :param anndata.AnnData data: AnnData object with peak counts.
    :param bool log_tf: Log-transform TF term, defaults to True.
    :param bool log_idf: Log-transform IDF term, defaults to True.
    :param bool log_tfidf: Log-transform TF*IDF term. Can only be used when log_tf and log_idf are False, defaults to False.
    :param int scale_factor: Scale factor to multiply the TF-IDF matrix by, defaults to 1e4.
    :raises TypeError: data must be anndata object.
    :raises AttributeError: log(TF*IDF) requires log(TF) and log(IDF) to be False.
    """
    if isinstance(data, AnnData):
        adata = data
    else:
        raise TypeError("Expected AnnData object!")

    if log_tfidf and (log_tf or log_idf):
        raise AttributeError(
            "When returning log(TF*IDF), \
            applying neither log(TF) nor log(IDF) is possible."
        )

    if sparse.issparse(adata.X):
        n_peaks = np.asarray(adata.X.sum(axis=1)).reshape(-1)
        n_peaks = sparse.dia_matrix((1.0 / n_peaks, 0), shape=(n_peaks.size, n_peaks.size))
        # This prevents making TF dense
        tf = np.dot(n_peaks, adata.X)
    else:
        n_peaks = np.asarray(adata.X.sum(axis=1)).reshape(-1, 1)
        tf = adata.X / n_peaks
    if scale_factor is not None and scale_factor != 0 and scale_factor != 1:
        tf = tf * scale_factor
    if log_tf:
        tf = np.log1p(tf)

    idf = np.asarray(adata.shape[0] / adata.X.sum(axis=0)).reshape(-1)
    if log_idf:
        idf = np.log1p(idf)

    if sparse.issparse(tf):
        idf = sparse.dia_matrix((idf, 0), shape=(idf.size, idf.size))
        tf_idf = np.dot(tf, idf)
    else:
        tf_idf = np.dot(sparse.csr_matrix(tf), sparse.csr_matrix(np.diag(idf)))

    if log_tfidf:
        tf_idf = np.log1p(tf_idf)

    adata.X = np.nan_to_num(tf_idf, 0)


def lsi(data, scale_embeddings=True, n_comps=50):
    """Run Latent Semantic Indexing.

    Note: Function is from muon package.

    :param anndata.AnnData data: AnnData object with peak counts.
    :param bool scale_embeddings: Scale embeddings to zero mean and unit variance, defaults to True.
    :param int n_comps: Number of components to calculate with SVD, defaults to 50.
    :raises TypeError: data must be anndata object.
    """
    if isinstance(data, AnnData):
        adata = data
    else:
        raise TypeError("Expected AnnData object!")

    # In an unlikely scnenario when there are less 50 features, set n_comps to that value
    n_comps = min(n_comps, adata.X.shape[1])

    #logging.info("Performing SVD")
    cell_embeddings, svalues, peaks_loadings = svds(adata.X, k=n_comps)

    # Re-order components in the descending order
    cell_embeddings = cell_embeddings[:, ::-1]
    svalues = svalues[::-1]
    peaks_loadings = peaks_loadings[::-1, :]

    if scale_embeddings:
        cell_embeddings = (cell_embeddings - cell_embeddings.mean(axis=0)) / cell_embeddings.std(
            axis=0
        )

    stdev = svalues / np.sqrt(adata.X.shape[0] - 1)

    adata.obsm["X_lsi"] = cell_embeddings
    adata.uns["lsi"] = {"stdev": stdev}
    adata.varm["LSI"] = peaks_loadings.T
    
    adata.obsm["X_pca"] = cell_embeddings
    adata.varm["PCs"] = peaks_loadings.T
    adata.uns["pca"] = {"stdev": stdev}


def atac_norm(adata, condition_col='nb_features'):
    """A function that normalizes count matrix using two methods (total and TFIDF) seperately,
    calculates PCA and UMAP and plots both UMAPs.

    :param anndata.AnnData adata: AnnData object with peak counts.
    :param str condition_col: Name of the column to use as color in the umap plot
    :return anndata.AnnData: Two AnnData objects with normalized matrices (Total and TFIDF) and UMAP.
    """
    adata_tfidf = adata.copy()
    adata_total = adata.copy()
    
    # perform tfidf and latent semantic indexing 
    print('Performing TFIDF and LSI...')
    tfidf(adata_tfidf)
    lsi(adata_tfidf)
    sc.pp.neighbors(adata_tfidf, n_neighbors=15, n_pcs=50, method='umap', metric='euclidean', use_rep='X_pca')
    sc.tl.umap(adata_tfidf, min_dist=0.1, spread=2)
    print('Done')
    
    # perform total normalization and pca
    print('Performing total normalization and PCA...')
    sc.pp.normalize_total(adata_total)
    adata_total.layers['normalised'] = adata_total.X.copy()
    epi.pp.log1p(adata_total)
    sc.pp.pca(adata_total, svd_solver='arpack', n_comps=50, use_highly_variable=False)
    sc.pp.neighbors(adata_total, n_neighbors=15, n_pcs=50, method='umap', metric='euclidean')
    sc.tl.umap(adata_total, min_dist=0.1, spread=2)
    print('Done')

    print('Plotting UMAP...')
    fig, axarr = plt.subplots(nrows=2, ncols=2, figsize=(10, 8))
    axes = axarr.flatten()
    sc.pl.pca(adata_tfidf, color=condition_col, title='TFIDF', legend_loc='none', ax=axes[0], show=False)
    sc.pl.pca(adata_total, color=condition_col, title='Total', legend_loc='right margin', ax=axes[1], show=False)
    sc.pl.umap(adata_tfidf, color=condition_col, title='TFIDF', legend_loc='none', ax=axes[2], show=False)
    sc.pl.umap(adata_total, color=condition_col, title='Total', legend_loc='right margin', ax=axes[3], show=False)

    plt.tight_layout()

    return adata_tfidf, adata_total


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

    return tf_idf


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

    return adata


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
    utils.check_module("kneed")
    utils.check_module("statsmodels")

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

# ---------------------- TOBIAS RUN -----------------------#


# from: https://github.com/yaml/pyyaml/issues/127#issuecomment-525800484
class _SpaceDumper(yaml.SafeDumper):
    # HACK: insert blank lines between top-level objects
    # inspired by https://stackoverflow.com/a/44284819/3786245
    def write_line_break(self, data=None):
        super().write_line_break(data)

        if len(self.indents) == 1:
            super().write_line_break()


def write_TOBIAS_config(out_path,
                        bams=[],
                        names=None,
                        fasta=None,
                        blacklist=None,
                        gtf=None,
                        motifs=None,
                        organism="human",
                        output="TOBIAS_output"):
    """
    Write a TOBIAS config file from input bams/fasta/blacklist etc.

    Parameters
    -----------
    out_path : string
        Path to output yaml file.
    bams : list of strings, optional
        List of paths to bam files.
    names : list of strings, optional
        List of names for the bams. If None, the names are set to the bam file names with common prefix and suffix removed. Default: None.
    fasta : string, optional
        Path to fasta file. Default: None.
    blacklist : string, optional
        Path to blacklist file. Default: None.
    gtf : string, optional
        Path to gtf file. Default: None.
    motifs : string, optional
        Path to motif file. Default: None.
    organism : string, optional
        Organism name. TOBIAS supports 'human', 'mouse' or 'zebrafish'. Default: "human".
    output : string, optional
        Output directory of the TOBIAS run. Default: "TOBIAS_output".
    """

    # Check organism input
    organism = organism.lower()
    valid_organisms = ["human", "mouse", "zebrafish"]
    if organism not in valid_organisms:
        raise ValueError(f"'{organism}' is not a valid organism. Valid organisms are: " + ", ".join(valid_organisms))

    # Remove any common prefix and suffix from names
    if names is None:
        prefix = os.path.commonprefix(bams)
        suffix = utils.longest_common_suffix(bams)
        names = [utils.remove_prefix(s, prefix) for s in bams]
        names = [utils.remove_suffix(s, suffix) for s in names]

    # Start building yaml
    data = {}
    data["data"] = {names[i]: bams[i] for i in range(len(bams))}
    data["run_info"] = {"organism": organism,
                        "blacklist": blacklist,
                        "fasta": fasta,
                        "gtf": gtf,
                        "motifs": motifs,
                        "output": output}

    # Flags for parts of pipeline to include/exclude
    data["flags"] = {"plot_comparison": True,
                     "plot_correction": True,
                     "plot_venn": True,
                     "coverage": True,
                     "wilson": True}

    # Default module parameters
    data["macs"] = "--nomodel --shift -100 --extsize 200 --broad"
    data["atacorrect"] = ""
    data["footprinting"] = ""
    data["bindetect"] = ""

    # Write dict to yaml file
    with open(out_path, 'w') as f:
        yaml.dump(data, f, Dumper=_SpaceDumper, default_flow_style=False, sort_keys=False)

    print(f"Wrote TOBIAS config yaml to '{out_path}'")


# --------------------------------------------------------------------- #
# --------------------- Insertsize distribution ----------------------- #
# --------------------------------------------------------------------- #

def add_insertsize(adata,
                   bam=None,
                   fragments=None,
                   barcode_col=None,
                   barcode_tag="CB",
                   regions=None):
    """
    Add information on insertsize to the adata object using either a .bam-file or a fragments file.
    Adds columns "insertsize_count" and "mean_insertsize" to adata.obs and a key "insertsize_distribution" to adata.uns containing the
    insertsize distribution as a pandas dataframe.

    Parameters
    ----------
    adata : anndata.AnnData
        AnnData object to add insertsize information to.
    bam : str, default None
        Path to bam file containing paired-end reads. If None, the fragments file is used instead.
    fragments : str, default None
        Path to fragments file containing fragments information. If None, the bam file is used instead.
    barcode_col : str, default None
        Column in adata.obs containing the name of the cell barcode. If barcode_col is None, it is assumed that the index of adata.obs contains the barcode.
    barcode_tag : str, default 'CB'
        Only for bamfiles: Tag used for the cell barcode for each read.
    regions : str, default None
        Only for bamfiles: A list of regions to obtain reads from, e.g. ['chr1:1-2000000']. If None, all reads in the .bam-file are used.

    Returns
    -------
    None - adata is adjusted in place.
    """

    adata_barcodes = adata.obs.index.tolist() if barcode_col is None else adata.obs[barcode_col].tolist()

    if bam is not None and fragments is not None:
        raise ValueError("Please provide either a bam file or a fragments file - not both.")

    elif bam is not None:
        table = insertsize_from_bam(bam, barcode_tag=barcode_tag, regions=regions, barcodes=adata_barcodes)

    elif fragments is not None:
        table = insertsize_from_fragments(fragments, barcodes=adata_barcodes)

    else:
        raise ValueError("Please provide either a bam file or a fragments file.")

    # Merge table to adata.obs and uns
    mean_table = table[["insertsize_count", "mean_insertsize"]]
    distribution_table = table[[c for c in table.columns if isinstance(c, int)]]

    distribution_barcodes = set(table.index)
    common = set(adata_barcodes).intersection(distribution_barcodes)
    missing = set(adata_barcodes) - distribution_barcodes

    if len(common) == 0:
        raise ValueError("No common barcodes")

    elif len(missing) > 0:
        print("WARNING: not all barcodes in adata.obs were represented in the input fragments. The values for these barcodes are set to NaN.")
        missing_table = pd.DataFrame(index=list(missing), columns=distribution_table.columns)
        distribution_table = pd.concat([distribution_table, missing_table])

    # Merge table to adata.obs and uns
    if barcode_col is None:
        adata.obs = adata.obs.merge(mean_table, left_index=True, right_index=True, how="left")
    else:
        adata.obs = adata.obs.merge(mean_table, left_on=barcode_col, right_index=True, how="left")

    adata.uns["insertsize_distribution"] = distribution_table.loc[adata_barcodes]
    adata.uns['insertsize_distribution'].columns = adata.uns['insertsize_distribution'].columns.astype(str)  # ensures correct order of barcodes in table

    print("Added insertsize information to adata.obs[[\"insertsize_count\", \"mean_insertsize\"]] and adata.uns[\"insertsize_distribution\"].")


def _check_in_list(element, alist):
    return element in alist


def _check_true(element, alist):
    return True


def insertsize_from_bam(bam,
                        barcode_tag="CB",
                        barcodes=None,
                        regions='chr1:1-2000000',
                        chunk_size=100000):
    """
    Get fragment insertsize distributions per barcode from bam file.

    Parameters
    -----------
    bam : str
        Path to bam file
    barcode_tag : str, default "CB"
        The read tag representing the barcode.
    barcodes : list, default None
        List of barcodes to include in the analysis. If None, all barcodes are included.
    regions : str or list of str, default 'chr1:1-2000000'
        Regions to include in the analysis. If None, all reads are included.
    chunk_size : int, default 500000
        Size of bp chunks to read from bam file.

    Returns
    --------
    pandas.DataFrame
        DataFrame with insertsize distributions per barcode.
    """

    # Load modules
    utils.check_module("pysam")
    import pysam

    if utils._is_notebook() is True:
        from tqdm import tqdm_notebook as tqdm
    else:
        from tqdm import tqdm

    if isinstance(regions, str):
        regions = [regions]

    # Prepare function for checking against barcodes list
    if barcodes is not None:
        barcodes = set(barcodes)
        check_in = _check_in_list
    else:
        check_in = _check_true

    # Open bamfile
    print("Opening bam file...")
    if not os.path.exists(bam + ".bai"):
        print("Bamfile has no index - trying to index with pysam...")
        pysam.index(bam)

    bam_obj = sctoolbox.bam.open_bam(bam, "rb", require_index=True)
    chromosome_lengths = dict(zip(bam_obj.references, bam_obj.lengths))

    # Create chunked genome regions:
    print(f"Creating chunks of size {chunk_size}bp...")

    if regions is None:
        regions = [f"{chrom}:0-{length}" for chrom, length in chromosome_lengths.items()]
    elif isinstance(regions, str):
        regions = [regions]

    # Create chunks from larger regions
    regions_split = []
    for region in regions:
        chromosome, start, end = re.split("[:-]", region)
        start = int(start)
        end = int(end)
        for chunk_start in range(start, end, chunk_size):
            chunk_end = chunk_start + chunk_size
            if chunk_end > end:
                chunk_end = end
            regions_split.append(f"{chromosome}:{chunk_start}-{chunk_end}")

    # Count insertsize per chunk using multiprocessing
    print(f"Counting insertsizes across {len(regions_split)} chunks...")
    count_dict = {}
    read_count = 0
    pbar = tqdm(total=len(regions_split), desc="Progress: ", unit="chunks")
    for region in regions_split:
        chrom, start, end = re.split("[:-]", region)
        for read in bam_obj.fetch(chrom, int(start), int(end)):
            read_count += 1
            try:
                barcode = read.get_tag(barcode_tag)
            except Exception:  # tag was not found
                barcode = "NA"

            # Add read to dict
            if check_in(barcode, barcodes) is True:
                size = abs(read.template_length) - 9  # length of insertion
                count_dict = add_fragment(count_dict, barcode, size)

        # Update progress
        pbar.update(1)
    pbar.close()  # close progress bar

    bam_obj.close()

    # Check if any reads were read at all
    if read_count == 0:
        raise ValueError("No reads found in bam file. Please check bamfile or adjust the 'regions' parameter to include more regions.")

    # Check if any barcodes were found
    if len(count_dict) == 0 and barcodes is not None:
        raise ValueError("No reads found in bam file for the barcodes given in 'barcodes'. Please adjust the 'barcodes' or 'barcode_tag' parameters.")

    # Convert dict to pandas dataframes
    print("Converting counts to dataframe")
    table = pd.DataFrame.from_dict(count_dict, orient="index")
    table = table[["insertsize_count", "mean_insertsize"] + sorted(table.columns[2:])]
    table["mean_insertsize"] = table["mean_insertsize"].round(2)

    print("Done getting insertsizes from bam!")

    return table


def insertsize_from_fragments(fragments, barcodes=None):
    """
    Get fragment insertsize distributions per barcode from fragments file.

    Parameters
    -----------
    fragments : str
        Path to fragments.bed(.gz) file.
    barcodes : list of str, default None
        Only collect fragment sizes for the barcodes in barcodes

    Returns
    --------
    pandas.DataFrame
        DataFrame with insertsize distributions per barcode.
    """

    # Open fragments file
    if anno._is_gz_file(fragments):
        f = gzip.open(fragments, "rt")
    else:
        f = open(fragments, "r")

    # Prepare function for checking against barcodes list
    if barcodes is not None:
        barcodes = set(barcodes)
        check_in = _check_in_list
    else:
        check_in = _check_true

    # Read fragments file and add to dict
    print("Counting fragment lengths from fragments file...")
    start_time = datetime.datetime.now()
    count_dict = {}
    for line in f:
        columns = line.rstrip().split("\t")
        start = int(columns[1])
        end = int(columns[2])
        barcode = columns[3]
        count = int(columns[4])
        size = end - start - 9  # length of insertion (-9 due to to shifted cutting of Tn5)

        # Only add fragment if check is true
        if check_in(barcode, barcodes) is True:
            count_dict = add_fragment(count_dict, barcode, size, count)

    end_time = datetime.datetime.now()
    elapsed = end_time - start_time
    f.close()
    print("Done reading file - elapsed time: {0}".format(str(elapsed).split(".")[0]))

    # Convert dict to pandas dataframe
    print("Converting counts to dataframe...")
    table = pd.DataFrame.from_dict(count_dict, orient="index")
    table = table[["insertsize_count", "mean_insertsize"] + sorted(table.columns[2:])]
    table["mean_insertsize"] = table["mean_insertsize"].round(2)

    print("Done getting insertsizes from fragments!")

    return table


def add_fragment(count_dict, barcode, size, count=1):
    """
    Add fragment of size 'size' to count_dict.

    Parameters
    -----------
    count_dict : dict
        Dictionary containing the counts per insertsize.
    barcode : str
        Barcode of the read.
    size : int
        Insertsize to add to count_dict.
    count : int, default 1
        Number of reads to add to count_dict.

    Returns
    --------
    count_dict : dict
        Updated count_dict
    """

    # Initialize if barcode is seen for the first time
    if barcode not in count_dict:
        count_dict[barcode] = {"mean_insertsize": 0, "insertsize_count": 0}

    # Add read to dict
    if size >= 0 and size <= 1000:  # do not save negative insertsize, and set a cap on the maximum insertsize to limit outlier effects

        count_dict[barcode]["insertsize_count"] += count

        # Update mean
        mu = count_dict[barcode]["mean_insertsize"]
        total_count = count_dict[barcode]["insertsize_count"]
        diff = (size - mu) / total_count
        count_dict[barcode]["mean_insertsize"] = mu + diff

        # Save to distribution
        if size not in count_dict[barcode]:  # first time size is seen
            count_dict[barcode][size] = 0
        count_dict[barcode][size] += count

    return count_dict


def plot_insertsize(adata, barcodes=None):
    """
    Plot insertsize distribution for barcodes in adata. Requires adata.uns["insertsize_distribution"] to be set.

    Parameters
    -----------
    adata : AnnData
        AnnData object containing insertsize distribution in adata.uns["insertsize_distribution"].
    barcodes : list of str, default None
        Subset of barcodes to plot information for. If None, all barcodes are used.

    Returns
    --------
    ax : matplotlib.Axes
        Axes object containing the plot.
    """

    if "insertsize_distribution" not in adata.uns:
        raise ValueError("adata.uns['insertsize_distribution'] not found!")

    insertsize_distribution = copy.deepcopy(adata.uns['insertsize_distribution'])
    insertsize_distribution.columns = insertsize_distribution.columns.astype(int)

    # Subset barcodes if a list is given
    if barcodes is not None:
        # Convert to list if only barcode is given
        if isinstance(barcodes, str):
            barcodes = [barcodes]
        table = insertsize_distribution.loc[barcodes].sum(axis=0)
    else:
        table = insertsize_distribution.sum(axis=0)

    # Plot
    ax = sns.lineplot(x=table.index, y=table.values)
    ax.set_xlabel("Insertsize (bp)")
    ax.set_ylabel("Count")

    return ax
