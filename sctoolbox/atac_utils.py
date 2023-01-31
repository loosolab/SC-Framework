import pandas as pd
import scanpy as sc
import re
import sctoolbox.qc_filter as qc_filter
import sctoolbox.creators as cr
import sctoolbox.utilities as utils
import sctoolbox.bam as bam_utils
import episcanpy as epi
import anndata as ad
from matplotlib import pyplot as plt
import warnings


def assemble_from_h5ad(h5ad_files,
                       qc_columns,
                       merge_column='sample',
                       coordinate_cols=None,
                       set_index=True,
                       index_from=None):
    '''
    Function to assemble multiple adata files into a single adata object with a sample column in the
    adata.obs table. This concatenates adata.obs and merges adata.uns.

    Parameters
    ----------
    h5ad_files: list of str
        list of h5ad_files
    qc_columns: dictionary
        dictionary of existing adata.obs column to add to infoprocess legend
    merge_column: str
        column name to store sample identifier
    coordinate_cols: list of str
        location information of the peaks
    set_index: boolean
        True: index will be formatted and can be set by a given column
    index_from: str
        column to build the index from

    Returns
    -------

    '''

    adata_dict = {}
    counter = 0
    for h5ad_path in h5ad_files:
        counter += 1

        sample = 'sample' + str(counter)

        adata = epi.read_h5ad(h5ad_path)
        if set_index:
            format_index(adata, index_from)

        # Establish columns for coordinates
        if coordinate_cols is None:
            coordinate_cols = adata.var.columns[:3]  # first three columns are coordinates
        else:
            utils.check_columns(adata.var, coordinate_cols,
                                "coordinate_cols")  # Check that coordinate_cols are in adata.var)

        format_adata_var(adata, coordinate_cols, coordinate_cols)

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

    adata = ad.concat(adata_dict, label=merge_column)
    adata.uns = ad.concat(adata_dict, uns_merge='same').uns
    for value in adata_dict.values():
        adata.var = pd.merge(adata.var, value.var, left_index=True, right_index=True)

    # Remove name of indexes for cellxgene compatibility
    adata.obs.index.name = None
    adata.var.index.name = None

    return adata


def format_index(adata, from_column=None):
    """
    This formats the index of adata.var by the pattern ["chr", "start", "stop"]
    Parameters
    ----------
    adata: anndata.AnnData
    from_column: None or column name (str) in adata.var to be set as index

    Returns
    -------

    """
    if from_column == None:
        entry = adata.var.index[0]
        index_type = get_index_type(entry)

        if index_type == 'snapatac':
            adata.var['name'] = adata.var['name'].str.replace("b'", "")
            adata.var['name'] = adata.var['name'].str.replace("'", "")

            # split the peak column into chromosome start and end
            adata.var[['peak_chr', 'start_end']] = adata.var['name'].str.split(':', expand=True)
            adata.var[['peak_start', 'peak_end']] = adata.var['start_end'].str.split('-', expand=True)
            # set types
            adata.var['peak_chr'] = adata.var['peak_chr'].astype(str)
            adata.var['peak_start'] = adata.var['peak_start'].astype(int)
            adata.var['peak_end'] = adata.var['peak_end'].astype(int)
            # remove start_end column
            adata.var.drop('start_end', axis=1, inplace=True)

            adata.var = adata.var.set_index('name')

        elif index_type == "start_name":
            coordinate_pattern = r"(chr[0-9XYM]+)+[\_\:\-]+[0-9]+[\_\:\-]+[0-9]+"
            new_index = []
            for line in adata.var.index:
                new_index.append(re.search(coordinate_pattern, line).group(0))
            adata.var['new_index'] = new_index
            adata.var.set_index('new_index', inplace=True)

    else:
        entry = list(adata.var[from_column])[0]
        index_type = get_index_type(entry)

        if index_type == 'snapatac':
            adata.var['name'] = adata.var['name'].str.replace("b'", "")
            adata.var['name'] = adata.var['name'].str.replace("'", "")

            # split the peak column into chromosome start and end
            adata.var[['peak_chr', 'start_end']] = adata.var['name'].str.split(':', expand=True)
            adata.var[['peak_start', 'peak_end']] = adata.var['start_end'].str.split('-', expand=True)
            # set types
            adata.var['peak_chr'] = adata.var['peak_chr'].astype(str)
            adata.var['peak_start'] = adata.var['peak_start'].astype(int)
            adata.var['peak_end'] = adata.var['peak_end'].astype(int)
            # remove start_end column
            adata.var.drop('start_end', axis=1, inplace=True)

            adata.var = adata.var.set_index('name')

        elif index_type == "start_name":
            coordinate_pattern = r"(chr[0-9XYM]+)+[\_\:\-]+[0-9]+[\_\:\-]+[0-9]+"
            new_index = []
            for line in adata.var[from_column]:
                new_index.append(re.search(coordinate_pattern, line).group(0))
            adata.var['new_index'] = new_index
            adata.var.set_index('new_index', inplace=True)


def get_index_type(entry):
    """
    Check the format of the index by regex
    Parameters
    ----------
    entry

    Returns
    -------

    """

    regex_snapatac = r"^b'(chr[0-9]+)+'[\_\:\-]+[0-9]+[\_\:\-]+[0-9]+" # matches: b'chr1':12324-56757
    regex_start_name = r"^.+(chr[0-9]+)+[\_\:\-]+[0-9]+[\_\:\-]+[0-9]+" # matches: some_name-chr1:12343-76899


    if re.match(regex_snapatac, entry):
        return 'snapatac'
    if re.match(regex_start_name, entry):
        return 'start_name'


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


# def build_default_thresholds(adata, manual_thresholds, groupby=None):
#     '''
#     This builds a dictionary from the manual set thresholds in the format to use it later
#     Parameters
#     ----------
#     manual_tresholds
#     groupby
#     columns
#
#     Returns
#     -------
#
#     '''
#     if groupby is not None:
#         samples = []
#         current_sample = None
#         for sample in adata.obs[groupby]:
#             if current_sample != sample:
#                 samples.append(sample)
#                 current_sample = sample
#
#         thresholds = {}
#         for key, value in manual_thresholds.items():
#             sample_dict = {}
#             for sample in samples:
#                 sample_dict[sample] = {key: value}
#             thresholds[key] = sample_dict
#
#     else:
#         thresholds = manual_thresholds
#
#     return thresholds


def get_thresholds_atac_wrapper(adata, manual_thresholds, only_automatic_thresholds=True, groupby=None):
    """
    return the thresholds for the filtering
    :param adata: anndata.AnnData
    :param manual_thresholds:
    :param automatic_thresholds:
    :return:
    """
    manual_thresholds = get_keys(adata, manual_thresholds)

    if only_automatic_thresholds:
        keys = list(manual_thresholds.keys())
        thresholds = qc_filter.automatic_thresholds(adata, which="obs", columns=keys, groupby=groupby)
        return thresholds
    else:
        samples = []
        current_sample = None
        for sample in adata.obs[groupby]:
            if current_sample != sample:
                samples.append(sample)
                current_sample = sample
        # thresholds which are not set by the user are set automatically
        for key, value in manual_thresholds.items():
            if value['min'] is None or value['max'] is None:
                auto_thr = qc_filter.automatic_thresholds(adata, which="obs", columns=[key], groupby=groupby)
                manual_thresholds[key] = auto_thr[key]
            else:
                thresholds = {}
                for sample in samples:
                    thresholds[sample] = {key: value}

                manual_thresholds[key] = thresholds

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


def validate_regions(adata, coordinate_columns):
    """ Checks if the regions in adata.var are valid.

    Parameters
    -----------
    adata : AnnData
        AnnData object containing the regions to be checked.
    coordinate_columns : list of str
        List of length 3 for column names in adata.var containing chr, start, end coordinates. """

    # Test whether the first three columns are in the right format
    chr, start, end = coordinate_columns

    # Test if coordinate columns are in adata.var
    utils.check_columns(adata.var, coordinate_columns, "adata.var")

    # Test whether the first three columns are in the right format
    for _, line in adata.var.to_dict(orient="index").items():
        valid = False

        if isinstance(line[chr], str) and isinstance(line[start], int) and isinstance(line[end], int):
            if line[start] <= line[end]:  # start must be smaller than end
                valid = True  # if all tests passed, the line is valid

        if valid is False:
            raise ValueError("The region {0}:{1}-{2} is not a valid genome region. Please check the format of columns: {3}".format(line[chr], line[start], line[end], coordinate_columns))


def format_adata_var(adata,
                     coordinate_columns=None,
                     columns_added=["chr", "start", "end"]):
    '''
    Formats the index of adata.var and adds peak_chr, peak_start, peak_end columns to adata.var if needed.
    If coordinate_columns are given, the function will check if these columns already contain the information needed. If the coordinate_columns are in the correct format, nothing will be done.
    If the coordinate_columns are invalid (or coordinate_columns is not given) the index is checked for the following format:
    "*[_:-]start[_:-]stop"

    If the index can be formatted, the formatted columns (columns_added) will be added.
    If the index cannot be formatted, an error will be raised.

    :param adata: AnnData
        The anndata object containing features to annotate.
    :param coordinate_columns: list of str or None
        List of length 3 for column names in adata.var containing chr, start, end coordinates to check.
        If None, the index will be formatted.
    :param columns_added: list of str
        List of length 3 for column names in adata.var containing chr, start, end coordinates to add.
    '''

    # Test whether the first three columns are in the right format
    format_index = True
    print(coordinate_columns)
    if coordinate_columns is not None:
        try:
            validate_regions(adata, coordinate_columns)
            format_index = False
        except KeyError:
            print("The coordinate columns are not found in adata.var. Trying to format the index.")
        except ValueError:
            print("The regions in adata.var are not in the correct format. Trying to format the index.")

    # Format index if needed
    if format_index:
        print("formatting adata.var index to coordinate columns:")
        regex = r'[^_:\-]+[\_\:\-]+[0-9]+[\_\:\-]+[0-9]+'  # matches chr_start_end / chr-start-end / chr:start-end and variations

        # Prepare lists to insert
        peak_chr_list = []
        peak_start_list = []
        peak_end_list = []

        names = adata.var.index
        for name in names:
            if re.match(regex, name):  # test if name can be split by regex

                # split the name into chr, start, end
                split_name = re.split(r'[\_\:\-]', name)
                peak_chr_list.append(split_name[0])
                peak_start_list.append(int(split_name[1]))
                peak_end_list.append(int(split_name[2]))

            else:
                raise ValueError("Index does not match the format *_start_stop or *:start-stop. Please check your index.")

        adata.var.drop(columns_added, axis=1,
                       errors='ignore', inplace=True)

        adata.var.insert(0, columns_added[2], peak_end_list)
        adata.var.insert(0, columns_added[1], peak_start_list)
        adata.var.insert(0, columns_added[0], peak_chr_list)

        # Check whether the newly added columns are in the right format
        validate_regions(adata, columns_added)


def bam_adata_ov(adata, bamfile, cb_col):
    """
    Check if adata.obs barcodes existing in a column of a bamfile
    Parameters
    ----------
    adata: anndata.AnnData
        adata object where adata.obs is stored
    bamfile: str
        path of the bamfile to investigate
    cb_col: str
        bamfile column to extract the barcodes from

    Returns
    -------

    """

    bam_obj = bam_utils.open_bam(bamfile, "rb")

    sample = []
    counter = 0
    iterations = 1000
    for read in bam_obj:
        tag = read.get_tag(cb_col)
        sample.append(tag)
        if counter == iterations:
            break
        counter += 1

    barcodes_df = pd.DataFrame(adata.obs.index)
    count_table = barcodes_df.isin(sample)
    hits = count_table.sum()
    hitrate = hits[0] / iterations

    return hitrate


def check_barcode_tag(adata, bamfile, cb_col):
    """
    Check for the possibilty that the wrong barcode is used
    Parameters
    ----------
    adata: anndata.AnnData
        adata object where adata.obs is stored
    bamfile: str
        path of the bamfile to investigate
    cb_col: str
        bamfile column to extract the barcodes from

    Returns
    -------

    """
    hitrate = bam_adata_ov(adata, bamfile, cb_col)

    if hitrate <= 0.05:
        warnings.warn('Less than 5% of the barcodes from the bamfile found in the .obs table. \n'
                      'Consider if you are using the wrong column for cb-tag or bamfile. \n'
                      'The following process can take several hours')
    elif hitrate == 0:
        warnings.warn('None of the barcodes from the bamfile found in the .obs table. \n'
                      'Consider if you are using the wrong column cb-tag or bamfile. \n'
                      'The following process can take several hours')

    if hitrate > 0.05:
        print('Barcode tag: OK')


if __name__ == '__main__':


    # violin_HVF_distribution(adata)
    # scatter_HVF_distribution(adata)

    # qc_columns = {}
    # qc_columns['n_features_by_counts'] = None
    # qc_columns['log1p_n_features_by_counts'] = None
    # qc_columns['total_counts'] = None
    # qc_columns['log1p_total_counts'] = None
    # qc_columns['mean_insertsize'] = None
    # qc_columns['n_total_fragments'] = None
    # qc_columns['n_fragments_in_promoters'] = None
    # qc_columns['pct_fragments_in_promoters'] = None
    # qc_columns['blacklist_overlaps'] = None
    # qc_columns['TN'] = 'TN'
    # qc_columns['UM'] = 'UM'
    # qc_columns['PP'] = 'PP'
    # qc_columns['UQ'] = 'UQ'
    # qc_columns['CM'] = 'CM'
    #
    # adata = assemble_from_h5ad(['/mnt/workspace/jdetlef/data/anndata/Esophagus.h5ad'],
    #                    qc_columns=qc_columns,
    #                    merge_column='sample',
    #                    coordinate_cols=None,
    #                    set_index=True,
    #                    index_from='name')
    # adata = assemble_from_h5ad(['/mnt/agnerds/PROJECTS/extern/ext442_scATAC_Glaser_11_22/preprocessing_output/data/all_annotated_peaks.h5ad'], qc_columns, coordinate_cols=['peak_chr', 'peak_start', 'peak_end'], column='sample')
    #adata = epi.read_h5ad('/mnt/workspace/jdetlef/processed_data/Esophagus/assembling/anndata/Esophagus.h5ad')
    #adata = epi.read_h5ad('/mnt/workspace/jdetlef/loosolab_sc_rna_framework/tests/data/atac/mm10_atac.h5ad')
    # bamfile = '/mnt/workspace/jdetlef/data/bamfiles/sorted_Esophagus.bam'
    #bamfile = '/mnt/workspace/jdetlef/loosolab_sc_rna_framework/tests/data/atac/homo_sapiens_liver.bam'

#    check_barcode_tag(adata, bamfile, cb_col='CB')
    #
    #
    adata = epi.read_h5ad('/mnt/workspace/jdetlef/ext_ana/processed/all/assembling/anndata/all.h5ad')
    # Filter to use:
    n_features_filter = True  # True or False; filtering out cells with numbers of features not in the range defined below
    mean_insertsize_filter = True  # True or False; filtering out cells with mean insertsize not in the range defined below
    filter_pct_fp = True  # True or False; filtering out cells with promotor_enrichment not in the range defined below
    filter_n_fragments = True  # True or False; filtering out cells with promotor_enrichment not in the range defined below
    filter_chrM_fragments = True  # True or False; filtering out cells with promotor_enrichment not in the range defined below
    filter_uniquely_mapped_fragments = True  # True or False; filtering out cells with promotor_enrichment not in the range defined

    # if this is True thresholds below are ignored
    only_automatic_thresholds = False  # True or False; to use automatic thresholds

    ############################# set default values #######################################
    #
    # This will be applied to all samples the thresholds can be changed manually when plotted
    # if thresholds None they are set automatically

    # default values n_features
    min_features = 10
    max_features = 100

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


    auto_thr = qc_filter.automatic_thresholds(adata, which="obs", columns=['n_features_by_counts', 'mean_insertsize'], groupby="Sample")
    # default_thresholds = build_default_thresholds(adata, manual_thresholds, groupby="Sample")
    thresholds = get_thresholds_atac_wrapper(adata, manual_thresholds, only_automatic_thresholds, groupby="Sample")

    print(thresholds)
