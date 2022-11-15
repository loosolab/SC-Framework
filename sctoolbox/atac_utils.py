import scanpy as sc
import re
import sctoolbox.qc_filter as qc_filter
import sctoolbox.creators as cr
import episcanpy as epi
import anndata as ad
from matplotlib import pyplot as plt

def assemble_from_h5ad(h5ad_files, qc_columns, column='sample', conditions=None):

    adata_dict = {}
    counter = 0
    for h5ad_path in h5ad_files:
        counter += 1

        sample = 'sample' + str(counter)

        adata = epi.read_h5ad(h5ad_path)

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

        #adata.obs = adata.obs.assign(sample=sample)
        adata.obs = adata.obs.assign(file=h5ad_path)

        # Add conditions
        
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


def get_thresholds_atac_wrapper(adata, manual_thresholds, automatic_thresholds=True):
    """
    return the thresholds for the filtering
    :param adata:
    :param manual_thresholds:
    :param automatic_thresholds:
    :return:
    """
    manual_thresholds = get_keys(adata, manual_thresholds)

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

    pass
