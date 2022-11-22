import scanpy as sc
import re
import sctoolbox.qc_filter as qc_filter
import sctoolbox.creators as cr
import episcanpy as epi
import anndata as ad
from matplotlib import pyplot as plt


def assemble_from_h5ad(h5ad_files, qc_columns, column='sample', atac=True, conditions=None):
    '''
    Function to assemble multiple adata files into a single adata object with a sample column in the
    adata.obs table. This concatenates adata.obs and merges adata.uns.

    Parameters
    ----------
    h5ad_files: list of str
        list of h5ad_files
    qc_columns: dictionary
        dictionary of adata.obs columns and related thresholds
    column: str
        column name to store sample identifier
    conditions: None

    Returns
    -------

    '''

    adata_dict = {}
    counter = 0
    for h5ad_path in h5ad_files:
        counter += 1

        sample = 'sample' + str(counter)

        adata = epi.read_h5ad(h5ad_path)
        if atac:
            adata.var = adata.var.set_index('name')

        # Add information to the infoprocess
        cr.build_infor(adata, "Input_for_assembling", h5ad_path)
        cr.build_infor(adata, "Strategy", "Read from h5ad")

        print('add existing adata.obs columns to infoprocess:')
        print()
        for key, value in qc_columns.items():
            if value is not None:
                print(key + ':' + value)
                if value in adata.obs.columns:
                    build_legend(adata, key, value)
                else:
                    print('column:  ' + value + ' is not in adata.obs')

        # check if the barcode is the index otherwise set it
        barcode_index(adata)

        adata.obs = adata.obs.assign(file=h5ad_path)

        # Add conditions here

        adata_dict[sample] = adata

    adata = ad.concat(adata_dict, label=column)
    adata.uns = ad.concat(adata_dict, uns_merge='same').uns

    return adata


def get_keys(adata, manual_thresholds):
    """
    get the keys of the obs columns
    :param adata:
    :return:
    """
    m_thresholds = {}
    legend = adata.uns["legend"]
    for key, value in manual_thresholds.items():
        if key in legend:
            obs_key = legend[key]
            m_thresholds[obs_key] = value
        else:
            print('column: ' + key + ' not found in adata.obs')

    return m_thresholds


def build_default_thresholds(adata, manual_thresholds, groupby=None):
    '''
    This builds a dictionary from the manual set thresholds in the format to use it later
    Parameters
    ----------
    manual_tresholds
    groupby
    columns

    Returns
    -------

    '''
    if groupby is not None:
        samples = []
        current_sample = None
        for sample in adata.obs[groupby]:
            if current_sample != sample:
                samples.append(sample)

        thresholds = {}
        for key, value in manual_thresholds.items():
            sample_dict = {}
            for sample in samples:
                sample_dict[sample] = {key, value}
            thresholds[key] = sample_dict

    else:
        thresholds = manual_thresholds

    return thresholds


def get_thresholds_atac_wrapper(adata, manual_thresholds, only_automatic_thresholds=True, groupby=None):
    """
    return the thresholds for the filtering
    :param adata:
    :param manual_thresholds:
    :param automatic_thresholds:
    :return:
    """
    manual_thresholds = get_keys(adata, manual_thresholds)

    if only_automatic_thresholds:
        keys = list(manual_thresholds.keys())
        i = type(keys[0])
        thresholds = qc_filter.automatic_thresholds(adata, which="obs", columns=keys, groupby=groupby)
        return thresholds
    else:
        # thresholds which are not set by the user are set automatically
        for key, value in manual_thresholds.items():
            if value['min'] is None or value['max'] is None:
                i = type(key)
                auto_thr = qc_filter.automatic_thresholds(adata, which="obs",columns=[key], groupby=groupby)
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
    if type(adata) != ad.AnnData:
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


def barcode_index(adata):
    """
    check if the barcode is the index
    :param adata:
    :return:
    """
    # regex for any barcode
    regex = re.compile(r'([ATCG]{8,16})')
    # get first index element
    first_index = adata.obs.index[0]
    # check if the first index element is a barcode
    if regex.match(first_index):
        index_is_barcode = True
    else:
        index_is_barcode = False

    if not adata.obs.index.name == "barcode" and not index_is_barcode:
        # check if the barcode column is in the obs
        if 'barcode' in adata.obs.columns:
            print('setting adata.obs.index = adata.obs[barcode]')
            adata.obs = adata.obs.set_index("barcode")
    elif not adata.obs.index.name == "barcode" and index_is_barcode:
        print('setting adata.obs.index.name = barcode')
        adata.obs.index.name = 'barcode'
    else:
        print('barcodes are already the index')


if __name__ == '__main__':

    qc_columns = {}
    qc_columns['n_features_by_counts'] = None
    qc_columns['log1p_n_features_by_counts'] = None
    qc_columns['total_counts'] = None
    qc_columns['log1p_total_counts'] = None
    qc_columns['mean_insertsize'] = None
    qc_columns['n_total_fragments'] = None
    qc_columns['n_fragments_in_promoters'] = None
    qc_columns['pct_fragments_in_promoters'] = None
    qc_columns['blacklist_overlaps'] = None
    qc_columns['TN'] = 'TN'
    qc_columns['UM'] = 'UM'
    qc_columns['PP'] = 'PP'
    qc_columns['UQ'] = 'UQ'
    qc_columns['CM'] = 'CM'


    adata = assemble_from_h5ad(['/mnt/workspace/jdetlef/data/anndata/cropped_146.h5ad'], qc_columns, column='sample', conditions=None)
    #adata = epi.read_h5ad('/mnt/workspace/jdetlef/processed_data/Esophagus/assembling/anndata/Esophagus.h5ad')


    # Filter to use:
    n_features_filter = True  # True or False; filtering out cells with numbers of features not in the range defined below
    mean_insertsize_filter = True  # True or False; filtering out cells with mean insertsize not in the range defined below
    filter_pct_fp = True  # True or False; filtering out cells with promotor_enrichment not in the range defined below
    filter_n_fragments = True  # True or False; filtering out cells with promotor_enrichment not in the range defined below
    filter_chrM_fragments = True  # True or False; filtering out cells with promotor_enrichment not in the range defined below
    filter_uniquely_mapped_fragments = True  # True or False; filtering out cells with promotor_enrichment not in the range defined

    # if this is True thresholds below are ignored
    only_automatic_thresholds = True  # True or False; to use automatic thresholds

    ############################# set default values #######################################
    #
    # This will be applied to all samples the thresholds can be changed manually when plotted
    # if thresholds None they are set automatically

    # default values n_features
    min_features = None
    max_features = None

    # default mean_insertsize
    upper_threshold_mis = None
    lower_threshold_mis = None

    # default promotor enrichment
    upper_threshold_pct_fp = 0.4
    lower_threshold_pct_fp = 0.1

    # default number of fragments
    upper_thr_fragments = 200000
    lower_thr_fragments = 0

    # default number of fragments in chrM
    upper_thr_chrM_fragments = 10000
    lower_thr_chrM_fragments = 0

    # default number of uniquely mapped fragments
    upper_thr_um = 20000
    lower_thr_um = 0
    manual_thresholds = {}
    if n_features_filter:
        manual_thresholds['n_features_by_counts'] = {'min': min_features, 'max': max_features}

    if mean_insertsize_filter:
        manual_thresholds['mean_insertsize'] = {'min': lower_threshold_mis, 'max': upper_threshold_mis}

    if filter_pct_fp:
        manual_thresholds['pct_fragments_in_promoters'] = {'min': lower_threshold_pct_fp, 'max': upper_threshold_pct_fp}

    if filter_n_fragments:
        manual_thresholds['TN'] = {'min': lower_thr_fragments, 'max': upper_thr_fragments}

    if filter_chrM_fragments:
        manual_thresholds['CM'] = {'min': lower_thr_chrM_fragments, 'max': upper_thr_chrM_fragments}

    if filter_uniquely_mapped_fragments:
        manual_thresholds['UM'] = {'min': lower_thr_um, 'max': upper_thr_um}

    #adata.obs = adata.obs.fillna(0)
    thresholds = get_thresholds_atac_wrapper(adata, manual_thresholds, only_automatic_thresholds)

    print(thresholds)