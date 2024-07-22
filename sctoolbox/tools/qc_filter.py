"""Tools for quality control."""
import numpy as np
import pandas as pd
import scanpy as sc
import multiprocessing as mp
import warnings
import anndata
import pkg_resources
import glob
from pathlib import Path
from sklearn.mixture import GaussianMixture
from kneed import KneeLocator
import matplotlib.pyplot as plt
import scrublet as scr
import scipy.stats as stats

from beartype import beartype
import numpy.typing as npt
from beartype.typing import Optional, Tuple, Union, Any, Literal

# toolbox functions
import sctoolbox.utils as utils
from sctoolbox.plotting.general import _save_figure
import sctoolbox.utils.decorator as deco
from sctoolbox._settings import settings
logger = settings.logger


###############################################################################
#                        PRE-CALCULATION OF QC METRICS                        #
###############################################################################

@deco.log_anndata
@beartype
def calculate_qc_metrics(adata: sc.AnnData,
                         percent_top: Optional[list[int]] = None,
                         inplace: bool = False,
                         **kwargs: Any) -> Optional[sc.AnnData]:
    """
    Calculate the qc metrics using `scanpy.pp.calculate_qc_metrics`.

    Parameters
    ----------
    adata : sc.AnnData
        Anndata object the quality metrics are added to.
    percent_top : Optional[list[int]], default None
        Which proportions of top genes to cover.
    inplace : bool, default False
        If the anndata object should be modified in place.
    **kwargs : Any
        Additional parameters forwarded to scanpy.pp.calculate_qc_metrics.

    Returns
    -------
    Optional[sc.AnnData]
        Returns anndata object with added quality metrics to .obs and .var. Returns None if `inplace=True`.

    See Also
    --------
    scanpy.pp.calculate_qc_metrics
        https://scanpy.readthedocs.io/en/stable/generated/scanpy.pp.calculate_qc_metrics.html

    Examples
    --------
    .. exec_code::

        import scanpy as sc
        import sctoolbox as sct

        adata = sc.datasets.pbmc3k()
        print("Columns in .obs before 'calculate_qc_metrics':", adata.obs.columns.tolist())
        sct.tools.qc_filter.calculate_qc_metrics(adata, inplace=True)
        print("Columns in .obs after 'calculate_qc_metrics':", adata.obs.columns.tolist())
    """

    # add metrics to copy of anndata
    if not inplace:
        adata = adata.copy()

    # remove n_genes from metrics before recalculation
    to_remove = [col for col in adata.obs.columns if col in ["n_genes", "log1p_n_genes", "n_features", "log1p_n_features"]]
    adata.obs.drop(columns=to_remove, inplace=True)

    # compute metrics
    sc.pp.calculate_qc_metrics(adata=adata, percent_top=percent_top, inplace=True, **kwargs)

    # Rename metrics
    adata.obs.rename(columns={"n_genes_by_counts": "n_genes", "log1p_n_genes_by_counts": "log1p_n_genes",
                              "n_features_by_counts": "n_features", "log1p_n_features_by_counts": "log1p_n_features"}, inplace=True)

    # return modified anndata
    if not inplace:
        return adata


@deco.log_anndata
@beartype
def predict_cell_cycle(adata: sc.AnnData,
                       species: Optional[str],
                       s_genes: Optional[str | list[str]] = None,
                       g2m_genes: Optional[str | list[str]] = None,
                       inplace: bool = True) -> Optional[sc.AnnData]:
    """
    Assign a score and a phase to each cell depending on the expression of cell cycle genes.

    Parameters
    ----------
    adata : sc.AnnData
        Anndata object to predict cell cycle on.
    species : Optional[str]
        The species of data. Available species are: human, mouse, rat and zebrafish.
        If both s_genes and g2m_genes are given, set species=None,
        otherwise species is ignored.
    s_genes : Optional[str | list[str]], default None
        If no species is given or desired species is not supported, you can provide
        a list of genes for the S-phase or a txt file containing one gene in each row.
        If only s_genes is provided and species is a supported input, the default
        g2m_genes list will be used, otherwise the function will not run.
    g2m_genes :  Optional[str | list[str]], default None
        If no species is given or desired species is not supported, you can provide
        a list of genes for the G2M-phase or a txt file containing one gene per row.
        If only g2m_genes is provided and species is a supported input, the default
        s_genes list will be used, otherwise the function will not run.
    inplace : bool, default True
        if True, add new columns to the original anndata object.

    Returns
    -------
    Optional[sc.AnnData]
        If inplace is False, return a copy of anndata object with the new column in the obs table.

    Raises
    ------
    ValueError:
        1: If s_genes or g2m_genes is not None and not of type list.
        2: If no cellcycle genes available for the given species.
        3. If given species is not supported and s_genes or g2m_genes are not given.
    """

    if not inplace:
        adata = adata.copy()

    # Check if the given s_genes/g2m_genes are lists/paths/None
    genes_dict = {"s_genes": s_genes, "g2m_genes": g2m_genes}
    for key, genes in genes_dict.items():
        if genes is not None:
            # check if s_genes is a file or list
            if isinstance(genes, str):
                if Path(genes).is_file():  # check if file exists
                    genes = utils.general.read_list_file(genes)
                else:
                    raise FileNotFoundError(f'The file {genes} was not found!')
            elif isinstance(s_genes, np.ndarray):
                genes = list(genes)
            elif not isinstance(genes, list):
                raise ValueError(f"Please provide a list of genes or a path to a list of genes to s_genes/g2m_genes! Type of {key} is {type(genes)}")

        # Save genes
        if key == "s_genes":
            s_genes = genes
        elif key == "g2m_genes":
            g2m_genes = genes

    # if two lists are given, use both and ignore species
    if s_genes is not None and g2m_genes is not None:
        species = None

    # get gene list for species
    elif species is not None:
        species = species.lower()

        # get path of directory where cell cycles gene lists are saved
        genelist_dir = pkg_resources.resource_filename("sctoolbox", "data/gene_lists/")

        # check if given species is available
        available_files = glob.glob(genelist_dir + "*_cellcycle_genes.txt")
        available_species = utils.general.clean_flanking_strings(available_files)
        if species not in available_species:
            logger.debug("Species was not found in available species!")
            logger.debug(f"genelist_dir: {genelist_dir}")
            logger.debug(f"available_files: {available_files}")
            logger.debug(f"All files in dir: {glob.glob(genelist_dir + '*')}")
            raise ValueError(f"No cellcycle genes available for species '{species}'. Available species are: {available_species}")

        # get cellcylce genes lists
        path_cellcycle_genes = genelist_dir + f"{species}_cellcycle_genes.txt"
        cell_cycle_genes = pd.read_csv(path_cellcycle_genes, header=None,
                                       sep="\t", names=['gene', 'phase']).set_index('gene')
        logger.debug(f"Read {len(cell_cycle_genes)} cell cycle genes list from file: {path_cellcycle_genes}")

        # if one list is given as input, get the other list from gene lists dir
        if s_genes is not None:
            logger.info("g2m_genes list is missing! Using default list instead")
            g2m_genes = cell_cycle_genes[cell_cycle_genes['phase'].isin(['g2m_genes'])].index.tolist()
        elif g2m_genes is not None:
            logger.info("s_genes list is missing! Using default list instead")
            s_genes = cell_cycle_genes[cell_cycle_genes['phase'].isin(['s_genes'])].index.tolist()
        else:
            s_genes = cell_cycle_genes[cell_cycle_genes['phase'].isin(['s_genes'])].index.tolist()
            g2m_genes = cell_cycle_genes[cell_cycle_genes['phase'].isin(['g2m_genes'])].index.tolist()

    else:
        raise ValueError("Please provide either a supported species or lists of genes!")

    # Scale the data before scoring
    sdata = sc.pp.scale(adata, copy=True)

    # Score the cells by s phase or g2m phase
    sc.tl.score_genes_cell_cycle(sdata, s_genes=s_genes, g2m_genes=g2m_genes)

    # add results to adata
    adata.obs['S_score'] = sdata.obs['S_score']
    adata.obs['G2M_score'] = sdata.obs['G2M_score']
    adata.obs['phase'] = sdata.obs['phase']

    if not inplace:
        return adata


@deco.log_anndata
@beartype
def estimate_doublets(adata: sc.AnnData,
                      use_native: bool = False,
                      threshold: Optional[float] = None,
                      inplace: bool = True,
                      plot: bool = True,
                      groupby: Optional[str] = None,
                      threads: int = 4,
                      fill_na: bool = True,
                      **kwargs: Any) -> Optional[sc.AnnData]:
    """
    Estimate doublet cells using scrublet.

    Adds additional columns "doublet_score" and "predicted_doublet" in adata.obs,
    as well as a "scrublet" key in adata.uns.

    Parameters
    ----------
    adata : sc.AnnData
        Anndata object to estimate doublets for.
    use_native : bool, default False
        If True, uses the native implementation of scrublet.
    threshold : float, default 0.25
        Threshold for doublet detection.
    inplace : bool, default True
        Whether to estimate doublets inplace or not.
    plot : bool, default True
        Whether to plot the doublet score distribution.
    groupby : Optional[str], default None
        Key in adata.obs to use for batching during doublet estimation. If threads > 1,
        the adata is split into separate runs across threads. Otherwise each batch is run separately.
    threads : int, default 4
        Number of threads to use.
    fill_na : bool, default True
        If True, replaces NA values returned by scrublet with 0 and False. Scrublet returns NA if it cannot calculate
        a doublet score. Keep in mind that this does not mean that it is no doublet.
        By setting this parameter true it is assmuned that it is no doublet.
    **kwargs : Any
        Additional arguments are passed to scanpy.external.pp.scrublet.

    Notes
    -----
    Groupby should be set if the adata consists of multiple samples, as this improves the doublet estimation.

    Returns
    -------
    Optional[sc.AnnData]
        If inplace is False, the function returns a copy of the adata object.
        If inplace is True, the function returns None.
    """

    if inplace is False:
        adata = adata.copy()

    # Estimate doublets
    if groupby is not None:

        all_groups = adata.obs[groupby].astype("category").cat.categories.tolist()
        if threads > 1:
            pool = mp.Pool(threads, maxtasksperchild=1)  # maxtasksperchild to avoid memory leaks

            # Run scrublet for each sub data
            logger.info("Sending {0} batches to {1} threads".format(len(all_groups), threads))
            jobs = []
            for i, sub in enumerate([adata[adata.obs[groupby] == group] for group in all_groups]):

                # Clean up adata before sending to thread
                sub.uns = {}
                sub.layers = None

                job = pool.apply_async(_run_scrublet, (sub, use_native, threshold), {"verbose": False, **kwargs})
                jobs.append(job)
            pool.close()

            utils.multiprocessing.monitor_jobs(jobs, "Scrublet per group")
            results = [job.get() for job in jobs]

        else:
            results = []
            for i, sub in enumerate([adata[adata.obs[groupby] == group] for group in all_groups]):
                logger.info("Scrublet per group: {}/{}".format(i + 1, len(all_groups)))
                res = _run_scrublet(sub, use_native=use_native, threshold=threshold, verbose=False, **kwargs)
                results.append(res)

        # Collect results for each element in tuples
        all_obs = [res[0] for res in results]
        all_uns = [res[1] for res in results]

        # Merge all simulated scores
        uns_dict = {"threshold": threshold, "doublet_scores_sim": np.array([])}
        for uns in all_uns:
            uns_dict["doublet_scores_sim"] = np.concatenate((uns_dict["doublet_scores_sim"], uns["doublet_scores_sim"]))

        # Merge all obs tables
        obs_table = pd.concat(all_obs)
        obs_table = obs_table.loc[adata.obs_names.tolist(), :]  # Sort obs to match original order

    else:
        # Run scrublet on adata
        obs_table, uns_dict = _run_scrublet(adata, threshold=threshold, **kwargs)

    # Save scores to object
    # ImplicitModificationWarning
    adata.obs["doublet_score"] = obs_table["doublet_score"]
    adata.obs["predicted_doublet"] = obs_table["predicted_doublet"]
    adata.uns["scrublet"] = uns_dict

    if fill_na:
        adata.obs[["doublet_score", "predicted_doublet"]] = (
            utils.tables.fill_na(adata.obs[["doublet_score", "predicted_doublet"]], inplace=False))

    # Check if all values in colum are of type boolean
    if adata.obs["predicted_doublet"].dtype != "bool":
        logger.warning("Could not estimate doublets for every barcode. Columns can contain NAN values.")

    # Plot the distribution of scrublet scores
    if plot is True:
        sc.external.pl.scrublet_score_distribution(adata)

    # Return adata (or None if inplace)
    if inplace is False:
        return adata


@beartype
def _run_scrublet(adata: sc.AnnData,
                  use_native: bool = False,
                  threshold: Optional[float] = None,
                  **kwargs: Any) -> Tuple[pd.DataFrame, dict[str, Union[np.ndarray, float, dict[str, float]]]]:
    """
    Thread-safe wrapper for running scrublet, which also takes care of catching any warnings.

    Parameters
    ----------
    adata : sc.AnnData
        Anndata object to estimate doublets for.
    use_native : bool, default False
        If True, uses the native implementation of scrublet.
    threshold : float, default 0.25
        Threshold for doublet detection.
    **kwargs : Any
        Additional arguments are passed to scanpy.external.pp.scrublet.

    Returns
    -------
    Tuple[pd.DataFrame, dict[str, Union[np.ndarray, float, dict[str, float]]]]
        Tuple containing .obs and .uns["scrublet"] of the adata object after scrublet.
    """

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=UserWarning, message="Received a view of an AnnData*")
        warnings.filterwarnings("ignore", category=anndata.ImplicitModificationWarning, message="Trying to modify attribute `.obs`*")  # because adata is a view

        if use_native:
            # Run scrublet with native implementation
            X = adata.X
            scrub = scr.Scrublet(X)
            doublet_scores, predicted_doublets = scrub.scrub_doublets()

            # Apply manual threshold
            if threshold:
                predicted_doublets = scrub.call_doublets(threshold=threshold)

            adata.obs["doublet_score"] = doublet_scores
            adata.obs["predicted_doublet"] = predicted_doublets

            adata.uns['scrublet'] = "scr.Scrublet(X).scrub_doublets()"

        sc.external.pp.scrublet(adata, copy=False, threshold=threshold, **kwargs)

    return (adata.obs, adata.uns["scrublet"])


@deco.log_anndata
@beartype
def predict_sex(adata: sc.AnnData,
                groupby: str,
                gene: str = "Xist",
                gene_column: Optional[str] = None,
                threshold: float = 0.3,
                plot: bool = True,
                save: Optional[str] = None,
                **kwargs: Any) -> None:
    """
    Predict sex based on expression of Xist (or another gene).

    Parameters
    ----------
    adata : sc.AnnData
        An anndata object to predict sex for.
    groupby : str
        Column in adata.obs to group by.
    gene : str, default "Xist"
        Name of a female-specific gene to use for estimating Male/Female split.
    gene_column : Optional[str], default None
        Name of the column in adata.var that contains the gene names. If not provided, adata.var.index is used.
    threshold : float, default 0.3
        Threshold for the minimum fraction of cells expressing the gene for the group to be considered "Female".
    plot : bool, default True
        Whether to plot the distribution of gene expression per group.
    save : Optional[str], default None
        If provided, the plot will be saved to this path.
    **kwargs : Any
        Additional arguments are passed to scanpy.pl.violin.

    Notes
    -----
    adata.X will be converted to numpy.ndarray if it is of type numpy.matrix.

    Returns
    -------
    None
    """

    # Normalize data before estimating expression
    logger.info("Normalizing adata")
    adata_copy = adata.copy()  # ensure that adata is not changed during normalization
    sc.pp.normalize_total(adata_copy, target_sum=None)
    sc.pp.log1p(adata_copy)

    # Get expression of gene per cell
    if gene_column is None:
        gene_names_lower = [s.lower() for s in adata_copy.var.index]
    else:
        gene_names_lower = [s.lower() for s in adata_copy.var[gene_column]]
    gene_index = [i for i, gene_name in enumerate(gene_names_lower) if gene_name == gene.lower()]
    if len(gene_index) == 0:
        logger.info("Selected gene is not present in the data. Prediction is skipped.")
        return

    # If adata.X is of type matrix convert to ndarray
    if isinstance(adata_copy.X, np.matrix):
        adata_copy.X = adata_copy.X.getA()

    # Try to flatten for adata.X np.ndarray. If not flatten for scipy sparse matrix
    try:
        adata_copy.obs["gene_expr"] = adata_copy.X[:, gene_index].flatten()
    except AttributeError:
        adata_copy.obs["gene_expr"] = adata_copy.X[:, gene_index].todense().A1

    # Estimate which samples are male/female
    logger.info("Estimating male/female per group")
    assignment = {}
    for group, table in adata_copy.obs.groupby(groupby):
        n_cells = len(table)
        n_expr = sum(table["gene_expr"] > 0)
        frac = n_expr / n_cells
        if frac >= threshold:
            assignment[group] = "Female"
        else:
            assignment[group] = "Male"

    # Add assignment to adata.obs
    df = pd.DataFrame().from_dict(assignment, orient="index")
    df.columns = ["predicted_sex"]
    if "predicted_sex" in adata.obs.columns:
        adata.obs.drop(columns=["predicted_sex"], inplace=True)
    adata.obs = adata.obs.merge(df, left_on=groupby, right_index=True, how="left")

    # Plot overview if chosen
    if plot:
        logger.info("Plotting violins")
        groups = adata.obs[groupby].unique()
        n_groups = len(groups)
        fig, axarr = plt.subplots(1, 2, sharey=True,
                                  figsize=[5 + len(groups) / 5, 4],
                                  gridspec_kw={'width_ratios': [min(4, n_groups), n_groups]})

        # Plot histogram of all values
        axarr[0].hist(adata_copy.obs["gene_expr"], bins=30, orientation="horizontal", density=True, color="grey")
        axarr[0].invert_xaxis()
        axarr[0].set_ylabel(f"Normalized {gene} expression")

        # Plot violins per group + color for female cells
        sc.pl.violin(adata_copy, keys="gene_expr", groupby=groupby, jitter=False, ax=axarr[1], show=False, order=groups, **kwargs)
        axarr[1].set_xticklabels(groups, rotation=45, ha="right")
        axarr[1].set_ylabel("")
        xlim = axarr[1].get_xlim()

        for i, group in enumerate(groups):
            if assignment[group] == "Female":
                color = "red"
                alpha = 0.3
            else:
                color = None
                alpha = 0
            axarr[1].axvspan(i - 0.5, i + 0.5, color=color, zorder=0, alpha=alpha, linewidth=0)
        axarr[1].set_xlim(xlim)
        axarr[1].set_title("Prediction of female groups")

        _save_figure(save)


###############################################################################
#                      STEP 1: FINDING AUTOMATIC CUTOFFS                      #
###############################################################################

@beartype
def gmm_threshold(data: npt.ArrayLike,
                  max_mixtures: int = 5,
                  n_std: int | float = 3,
                  plot: bool = False) -> dict[str, float]:
    """
    Get automatic min/max thresholds for input data array.

    The function will fit a gaussian mixture model, and find the threshold
    based on the mean and standard deviation of the largest mixture in the model.

    The number of mixtures aka components is estimated using the BIC criterion. 

    Parameters
    ----------
    data : npt.ArrayLike
        Array of data to find thresholds for.
    max_mixtures : int, default 5
        Maximum number of gaussian mixtures to fit.
    n_std : int | float, default 3
        Number of standard deviations from distribution mean to set as min/max thresholds.
    plot : bool, default False
        If True, will plot the distribution of BIC and the fit of the gaussian mixtures to the data.

    Returns
    -------
    dict[str, float]
        Dictionary with min and max thresholds.
    """

    # Get numpy array if input was pandas series or list
    data_type = type(data).__name__
    if data_type == "Series":
        data = data.values
    elif data_type == "list":
        data = np.array(data)

    # Attempt to reshape values
    data = data.reshape(-1, 1)

    # Fit data with gaussian mixture
    n_list = list(range(1, max_mixtures + 1))  # 1->max mixtures per model
    models = [None] * len(n_list)
    for i, n in enumerate(n_list):
        models[i] = GaussianMixture(n).fit(data)

    # Evaluate quality of models
    # AIC = [m.aic(data) for m in models]
    BIC = [m.bic(data) for m in models]

    # Choose best number of mixtures
    try:
        kn = KneeLocator(n_list, BIC, curve='convex', direction='decreasing')
        # store selected mixtures
        selected = kn.knee
        M_best = models[selected - 1]  # -1 to get index

    except Exception:
        # Knee could not be found; use the normal distribution estimated using one gaussian
        M_best = models[0]
        # store selected mixtures
        selected = 1

    # Which is the largest component? And what are the mean/variance of this distribution?
    weights = M_best.weights_
    i = np.argmax(weights)
    dist_mean = M_best.means_[i][0]
    dist_std = np.sqrt(M_best.covariances_[i][0][0])

    # Threshold estimation
    thresholds = {"min": dist_mean - dist_std * n_std,
                  "max": dist_mean + dist_std * n_std}

    # ------ Plot if chosen -------#
    if plot:

        fig, axarr = plt.subplots(1, 2, figsize=(7, 3), constrained_layout=True)
        axarr = axarr.ravel()

        # Plot distribution of BIC
        # plt.plot(n_list, AIC, color="red", label="AIC")
        axarr[0].plot(n_list, BIC, color="blue")
        axarr[0].set_xlabel("Number of mixtures")
        axarr[0].set_ylabel("BIC")

        axarr[0].axvline(selected, color="red", linestyle="--", label="Selected mixtures")

        # Plot distribution of gaussian mixtures
        min_x = min(data)
        max_x = max(data)
        x = np.linspace(min_x, max_x, 1000).reshape(-1, 1)
        logprob = M_best.score_samples(x)
        responsibilities = M_best.predict_proba(x)
        pdf = np.exp(logprob)
        pdf_individual = responsibilities * pdf[:, np.newaxis]

        axarr[1].hist(data, density=True)
        axarr[1].set_xlabel("Value")
        axarr[1].set_ylabel("Density")
        for i in range(M_best.n_components):
            w = weights[i] * 100
            axarr[1].plot(x, pdf_individual[:, i], label=f"Component {i+1} ({w:.0f}%)")

        axarr[1].axvline(thresholds["min"], color="red", linestyle="--")
        axarr[1].axvline(thresholds["max"], color="red", linestyle="--")
        axarr[1].legend(bbox_to_anchor=(1.05, 1), loc=2)  # locate legend outside of plot

    return thresholds


@beartype
def mad_treshold(data: npt.ArrayLike,
                 min_n: Union[int, float] = 3,
                 max_n: Union[int, float] = 3,
                 plot: bool = False) -> dict[str, float]:
    """
    Compute an automatic threshold using the median absolute deviation (MAD).

    The threshold is calcualted as median(data) -/+ MAD * n.

    Parameters
    ----------
    data : npt.ArrayLike
        Array of data to find thresholds for.
    min_n : int | float, default 3
        Number of MADs from distribution median to set as min thresholds.
    max_n : int | float, default 3
        Number of MADs from distribution median to set as max thresholds.
    plot : bool, default False
        If True, will plot the distribution of BIC and the fit of the gaussian mixtures to the data.

    Returns
    -------
    dict[str, float]
        Dictionary with min and max thresholds.
    """
    median = np.median(data)
    MAD = stats.median_abs_deviation(data)

    result = {"min": median - MAD * min_n,
              "max": median + MAD * max_n}

    if plot:
        _, ax = plt.subplots(figsize=(7, 3), constrained_layout=True)

        ax.hist(data, density=True)
        ax.set_xlabel("Value")
        ax.set_ylabel("Density")

        ml1 = ax.axvline(result["min"], color="red", linestyle="--")
        ml2 = ax.axvline(result["max"], color="red", linestyle="--")
        medl = ax.axvline(median, color="grey", linestyle="--")
        ax.legend((medl, ml1, ml2),
                  (f"Median ({median:.2f})", f"Min. MAD ({result['min']:.2f})", f"Max. MAD ({result['max']:.2f})"),
                  bbox_to_anchor=(1.05, 1), loc=2)  # locate legend outside of plot

    return result


@beartype
def automatic_thresholds(adata: sc.AnnData,
                         which: Literal["obs", "var"] = "obs",
                         groupby: Optional[str] = None,
                         columns: Optional[list[str]] = None,
                         FUN: callable = gmm_threshold,
                         FUN_kwargs: Any = None) -> dict[str, dict[str, Union[float, dict[str, float]]]]:
    """
    Get automatic thresholds for multiple data columns in adata.obs or adata.var.

    Parameters
    ----------
    adata : sc.AnnData
        Anndata object to find thresholds for.
    which : Literal["obs", "var"], default "obs"
        Which data to find thresholds for. Either "obs" or "var".
    groupby : Optional[str], default None
        Group rows by the column given in 'groupby' to find thresholds independently per group
    columns : Optional[list[str]], default None
        Columns to calculate automatic thresholds for. If None, will take all numeric columns.
    FUN : callable, default gmm_threshold
        A filter function. The function is expected to accept an array of values and to return a dict with thresholds: {"min": 0, "max": 1}.
        Available functions: gmm_threshold, mad_threshold.
    FUN_kwargs : Any
        Additional kwargs forwarded to the filter function.

    Returns
    -------
    dict[str, dict[str, Union[float, dict[str, float]]]]
        A dict containing thresholds for each data column,
        either grouped by groupby or directly containing "min" and "max" per column.

    Raises
    ------
    ValueError:
        If which is not set to 'obs' or 'var'
    """

    # Find out which data to find thresholds for
    if which == "obs":
        table = adata.obs
    elif which == "var":
        table = adata.var

    # Establish which columns to find thresholds for
    if columns is None:
        columns = list(table.select_dtypes(np.number).columns)

    # Check groupby
    if groupby is not None:
        if groupby not in table.columns:
            raise ValueError()

    # Get threshold per data column (and groupby if chosen)
    thresholds = {}
    for col in columns:

        if groupby is None:
            data = table[col].values
            data[np.isnan(data)] = 0
            d = FUN(data, **FUN_kwargs)
            thresholds[col] = d

        else:
            thresholds[col] = {}  # initialize to fill in per group
            for group, subtable in table.groupby(groupby):
                data = subtable[col].values
                data[np.isnan(data)] = 0
                d = FUN(data, **FUN_kwargs)
                thresholds[col][group] = d

    return thresholds


@beartype
def thresholds_as_table(threshold_dict: dict[str, dict[str, float | int | dict[str, int | float]]]) -> pd.DataFrame:
    """
    Show the threshold dictionary as a table.

    Parameters
    ----------
    threshold_dict : dict[str, dict[str, float | int | dict[str, int | float]]]
        Dictionary with thresholds.

    Returns
    -------
    pd.DataFrame
    """

    rows = []
    for column in threshold_dict:

        if "min" in threshold_dict[column] or "max" in threshold_dict[column]:
            row = [column, np.nan, threshold_dict[column].get("min", np.nan), threshold_dict[column].get("max", np.nan)]
            rows.append(row)
        else:
            for group in threshold_dict[column]:
                row = [column, group, threshold_dict[column][group].get("min", np.nan), threshold_dict[column][group].get("max", np.nan)]
                rows.append(row)

    # Assemble table
    df = pd.DataFrame(rows)
    if len(df) > 0:  # df can be empty if no valid thresholds were input

        df.columns = ["Parameter", "Group", "Minimum", "Maximum"]

        # Remove group column if no thresholds had groups
        if df["Group"].isnull().sum() == df.shape[0]:
            df.drop(columns="Group", inplace=True)

        # Remove duplicate rows
        df.drop_duplicates(inplace=True)

    return df


######################################################################################
#                     STEP 2:     DEFINE CUSTOM CUTOFFS                              #
######################################################################################

@beartype
def _validate_minmax(d: dict) -> None:
    """Validate that the dict 'd' contains the keys 'min' and 'max'."""

    allowed = set(["min", "max"])
    keys = set(d.keys())

    not_allowed = len(keys - allowed)
    if not_allowed > 0:
        raise ValueError("Keys {0} not allowed".format(not_allowed))


@beartype
def validate_threshold_dict(table: pd.DataFrame,
                            thresholds: dict[str, dict[str, int | float] | dict[str, dict[str, int | float]]],
                            groupby: Optional[str] = None) -> None:
    """
    Validate threshold dictionary.

    Thresholds can be in the format:

    .. code-block:: python

        thresholds = {"chrM_percent": {"min": 0, "max": 10},
                      "total_reads": {"min": 1000}}

    Or per group in 'groupby':

    .. code-block:: python

        thresholds = {"chrM_percent": {
                                "Sample1": {"min": 0, "max": 10},
                                "Sample2": {"max": 5}
                                },
                      "total_reads": {"min": 1000}}

    Parameters
    ----------
    table : pd.DataFrame
        Table to validate thresholds for.
    thresholds : dict[str, dict[str, int | float] | dict[str, dict[str, int | float]]]
        Dictionary of thresholds to validate.
    groupby : Optional[str], default None
        Column for grouping thresholds.

    Raises
    ------
    ValueError
        If the threshold dict is not valid.
    """

    if groupby is not None:
        groups = table[groupby]

    # Check if all columns in thresholds are available
    threshold_columns = thresholds.keys()
    not_found = [col for col in threshold_columns if col not in table.columns]
    if len(not_found) > 0:
        raise ValueError("Column(s) '{0}' given in thresholds are not found in table".format(not_found))

    # Check the format of individual column thresholds
    for col in thresholds:

        if groupby is None:  # expecting one threshold for all cells
            _validate_minmax(thresholds[col])

        else:  # Expecting either one threshold or a threshold for each sample

            for key in thresholds[col]:
                if key in groups:
                    minmax_dict = thresholds[col][key]
                    _validate_minmax(minmax_dict)
                else:  # this is a minmax threshold
                    _validate_minmax(thresholds[col])


@deco.log_anndata
@beartype
def get_thresholds_wrapper(adata: sc.AnnData,
                           manual_thresholds: dict,
                           only_automatic_thresholds: bool = True,
                           groupby: Optional[str] = None) -> dict[str, dict[str, Union[float | int, dict[str, float | int]]]]:
    """
    Get the thresholds for the filtering.

    Parameters
    ----------
    adata : sc.AnnData
        Anndata object to find QC thresholds for.
    manual_thresholds : dict[str, dict[str, Union[float, dict[str, float]]]]
        Dictionary containing manually set thresholds
    only_automatic_thresholds : bool, default True
        If True, only set automatic thresholds.
    groupby : Optional[str], default None
        Group cells by column in adata.obs.

    Returns
    -------
    dict[str, dict[str, Union[float | int, dict[str, float | int]]]]
        Dictionary containing the thresholds
    """
    manual_thresholds = get_keys(adata, manual_thresholds)

    if only_automatic_thresholds:
        keys = list(manual_thresholds.keys())
        thresholds = automatic_thresholds(adata, which="obs", columns=keys, groupby=groupby)
        return thresholds
    else:
        if groupby:
            samples = []
            current_sample = None
            for sample in adata.obs[groupby]:
                if current_sample != sample:
                    samples.append(sample)
                    current_sample = sample
        # thresholds which are not set by the user are set automatically
        for key, value in manual_thresholds.items():
            if value['min'] is None or value['max'] is None:
                auto_thr = automatic_thresholds(adata, which="obs", columns=[key], groupby=groupby)
                manual_thresholds[key] = auto_thr[key]
            else:
                if groupby:
                    thresholds = {}
                    for sample in samples:
                        thresholds[sample] = value
                else:
                    thresholds = {key: value}

                manual_thresholds[key] = thresholds

        return manual_thresholds


@beartype
def get_keys(adata: sc.AnnData,
             manual_thresholds: dict[str, Any]) -> dict[str, dict[str, Union[float | int, dict[str, float | int]]]]:
    """
    Get threshold dictionary with keys that overlap with adata.obs.columns.

    Parameters
    ----------
    adata : sc.AnnData
        Anndata object
    manual_thresholds : dict[str, Any]
        Dictionary with adata.obs colums as keys.

    Returns
    -------
    dict[str, dict[str, Union[float | int, dict[str, float | int]]]]
        Dictionary with key - adata.obs.column overlap
    """

    m_thresholds = {}
    legend = adata.obs.columns
    for key, value in manual_thresholds.items():
        if key in legend:
            m_thresholds[key] = value
        else:
            logger.info('column: ' + key + ' not found in adata.obs')

    return m_thresholds


@beartype
def get_mean_thresholds(thresholds: dict[str, Any]) -> dict[str, Any]:
    """Convert grouped thresholds to global thresholds by taking the mean across groups."""

    global_thresholds = {}
    for key, adict in thresholds.items():
        global_thresholds[key] = {}

        if "min" in adict or "max" in adict:  # already global threshold
            global_thresholds[key] = adict
        else:
            min_values = [v.get("min", None) for v in adict.values() if "min" in v]
            if len(min_values) > 0:
                global_thresholds[key]["min"] = np.mean(min_values)

            max_values = [v.get("max", None) for v in adict.values() if "max" in v]
            if len(max_values) > 0:
                global_thresholds[key]["max"] = np.mean(max_values)

    return global_thresholds


###############################################################################
#                           STEP 3: APPLYING CUTOFFS                          #
###############################################################################


@deco.log_anndata
@beartype
def apply_qc_thresholds(adata: sc.AnnData,
                        thresholds: dict[str, Any],
                        which: Literal["obs", "var"] = "obs",
                        groupby: Optional[str] = None,
                        inplace: bool = True) -> Optional[sc.AnnData]:
    """
    Apply QC thresholds to anndata object.

    Parameters
    ----------
    adata : sc.AnnData
        Anndata object to filter.
    thresholds : dict[str, Any]
        Dictionary of thresholds to apply.
    which : Literal["obs", "var"], default 'obs'
       Which table to filter on. Must be one of "obs" / "var".
    groupby : Optional[str], default None
        Column in table to group by.
    inplace : bool, default True
        Change adata inplace or return a changed copy.

    Returns
    -------
    Optional[sc.AnnData]
        Anndata object with QC thresholds applied.

    Raises
    ------
    ValueError:
        1: If the keys in thresholds do not match with the columns in adata.[which].
        2: If grouped thesholds are not found. For example do not contain min and max values.
        3: If thresholds do not contain min and max values.
    """

    table = adata.obs if which == "obs" else adata.var

    # Cells or genes? For naming in log prints
    if which == "obs":
        name = "cells"
    else:
        name = ".var features"

    # Check if all columns are found in adata
    not_found = list(set(thresholds) - set(table.columns))
    if len(not_found) > 0:
        logger.info("{0} threshold columns were not found in adata and could therefore not be applied. These columns are: {1}".format(len(not_found), not_found))
    thresholds = {k: thresholds[k] for k in thresholds if k not in not_found}

    if len(thresholds) == 0:
        raise ValueError(f"The thresholds given do not match the columns given in adata.{which}. Please adjust the 'which' parameter if needed.")

    if groupby is not None:
        groups = table[groupby].unique()

    # Check that thresholds contain min/max
    for column, d in thresholds.items():
        if 'min' not in d and 'max' not in d:
            if groupby is not None:
                keys = d.keys()
                not_found = list(set(keys) - set(groups))
                if len(not_found) > 0:
                    raise ValueError(f"{len(not_found)} groups from thresholds were not found in adata.obs[{groupby}]. These groups are: {not_found}")

            else:
                raise ValueError("Error in threshold format: Thresholds must contain min or max per column, or a threshold per group in groupby")

    # Apply thresholds
    for column, d in thresholds.items():

        # Update size of table
        table = adata.obs if which == "obs" else adata.var

        # Collect boolean array of rows to select of table
        global_threshold = False  # can be overwritten if thresholds are global
        excluded = np.array([False] * len(table))
        if groupby is not None:
            if "min" in d or "max" in d:
                global_threshold = True

            else:
                for group in d:
                    minmax_dict = d[group]

                    group_bool = table[groupby] == group

                    if "min" in minmax_dict:
                        excluded = excluded | (group_bool & (table[column] < minmax_dict["min"]))

                    if "max" in minmax_dict:
                        excluded = excluded | (group_bool & (table[column] > minmax_dict["max"]))
        else:
            global_threshold = True

        # Select using a global threshold
        if global_threshold is True:
            minmax_dict = d

            if "min" in minmax_dict:
                excluded = excluded | (table[column] < minmax_dict["min"])  # if already excluded, or if excluded by min

            if "max" in minmax_dict:
                excluded = excluded | (table[column] > minmax_dict["max"])

        # Apply filtering
        included = ~excluded

        if inplace:
            # NOTE: these are privat anndata functions so they might change without warning!
            if which == "obs":
                adata._inplace_subset_obs(included)
            else:
                adata._inplace_subset_var(included)
        else:
            if which == "obs":
                adata = adata[included]
            else:
                adata = adata[:, included]  # filter on var

        logger.info(f"Filtering based on '{column}' from {len(table)} -> {sum(included)} {name}")

    if inplace is False:
        return adata


###############################################################################
#                         STEP 4: ADDITIONAL FILTERING                        #
###############################################################################

@beartype
def _filter_object(adata: sc.AnnData,
                   filter: str | list[str],
                   which: Literal["obs", "var"] = "obs",
                   remove_bool: bool = True,
                   inplace: bool = True) -> Optional[sc.AnnData]:
    """Filter an adata object based on a filter on either obs (cells) or var (genes). Is called by filter_cells and filter_genes."""

    # Decide which element type (genes/cells) we are dealing with
    if which == "obs":
        table = adata.obs
        table_name = "adata.obs"
        element_name = "cells"
    else:
        table = adata.var
        table_name = "adata.var"
        element_name = "genes"

    n_before = len(table)

    # genes is either a string (column in .var table) or a list of genes to remove
    if isinstance(filter, str):
        if filter not in table.columns:
            raise ValueError(f"Column {filter} not found in {table_name}.columns")

        if table[filter].dtype.name != "bool":
            raise ValueError(f"Column {filter} contains values that are not of type boolean")

        boolean = table[filter].values
        if remove_bool is True:
            boolean = ~boolean

    else:
        # Check if all genes/cells are found in adata
        not_found = list(set(filter) - set(table.index))
        if len(not_found) > 0:
            logger.info(f"{len(not_found)} {element_name} were not found in adata and could therefore not be removed. These genes are: {not_found}")

        boolean = ~table.index.isin(filter)

    # Remove genes from adata
    if inplace:
        if which == "obs":
            adata._inplace_subset_obs(boolean)
        elif which == "var":
            adata._inplace_subset_var(boolean)  # boolean is the included genes
    else:
        if which == "obs":
            adata = adata[boolean]
        elif which == "var":
            adata = adata[:, boolean]

    n_after = adata.shape[0] if which == "obs" else adata.shape[1]
    filtered = n_before - n_after
    logger.info(f"Filtered out {filtered} {element_name} from adata. New number of {element_name} is: {n_after}")

    if inplace is False:
        return adata


@deco.log_anndata
@beartype
def filter_cells(adata: sc.AnnData,
                 cells: str | list[str],
                 remove_bool: bool = True,
                 inplace: bool = True) -> Optional[sc.AnnData]:
    """
    Remove cells from anndata object.

    Parameters
    ----------
    adata : sc.AnnData
        Anndata object to filter.
    cells : str | list[str]
        A column in .obs containing boolean indicators or a list of cells to remove from object .obs table.
    remove_bool : bool, default True
        Is used if genes is a column in .obs table. If True, remove cells that are True. If False, remove cells that are False.
    inplace : bool, default True
        If True, filter inplace. If False, return filtered adata object.

    Returns
    -------
    Optional[sc.AnnData]
        If inplace is False, returns the filtered Anndata object. If inplace is True, returns None.
    """

    ret = _filter_object(adata, cells, which="obs", remove_bool=remove_bool, inplace=inplace)

    return ret


@deco.log_anndata
@beartype
def filter_genes(adata: sc.AnnData,
                 genes: str | list[str],
                 remove_bool: bool = True,
                 inplace: bool = True) -> Optional[sc.AnnData]:
    """
    Remove genes from adata object.

    Parameters
    ----------
    adata : sc.AnnData
        Annotated data matrix object to filter
    genes : str | list[str]
        A column containing boolean indicators or a list of genes to remove from object .var table.
    remove_bool : bool, default True
        Is used if genes is a column in .var table. If True, remove genes that are True. If False, remove genes that are False.
    inplace : bool, default True
        If True, filter inplace. If False, return filtered adata object.

    Returns
    -------
    Optional[sc.AnnData]
        If inplace is False, returns the filtered Anndata object. If inplace is True, returns None.
    """

    ret = _filter_object(adata, genes, which="var", remove_bool=remove_bool, inplace=inplace)

    return ret
