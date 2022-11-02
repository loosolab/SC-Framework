import atac as atac
import episcanpy as epi
import sctoolbox.atac_tree as sub_tree
import sctoolbox.qc_filter as qc_filter

# bam = '/home/jan/python-workspace/sc-atac/data/bamfiles/sorted_cropped_146.bam'
# adata = epi.read_h5ad('/home/jan/python-workspace/sc-atac/data/anndata/cropped_146.h5ad')
# adata.obs = adata.obs.set_index('barcode')
#
# adata = atac.add_insertsize(adata, bam=bam)
#
# print('done')

#Path related settings (these should be the same as for the previous notebook)
output_dir = '/home/jan/python-workspace/sc-atac/processed_data'
test = 'cropped_146'


# make an instance of the class
tree = sub_tree.ATAC_tree()
# set processing/output directory
tree.processing_dir = output_dir
# set sample/experiment..
tree.run = test

# load the data
assembling_output = tree.assembled_anndata
adata = epi.read_h5ad(assembling_output)

# if this is True thresholds below are ignored
automatic_thresholds = True # True or False; to use automatic thresholds
#

n_features_filter = False # True or False; filtering out cells with numbers of features not in the range defined below
# default values n_features
min_features = 5
max_features = 10000

mean_insertsize_filter = True # True or False; filtering out cells with mean insertsize not in the range defined below
# default mean_insertsize
upper_threshold_mis=None
lower_threshold_mis=None

filter_pct_fp=True # True or False; filtering out cells with promotor_enrichment not in the range defined below
# default promotor enrichment
upper_threshold_pct_fp=0.4
lower_threshold_pct_fp=0.1

filter_n_fragments=True # True or False; filtering out cells with promotor_enrichment not in the range defined below
# default number of fragments
upper_thr_fragments=1000000
lower_thr_fragments=2

filter_chrM_fragments=True # True or False; filtering out cells with promotor_enrichment not in the range defined below
# default number of fragments in chrM
upper_thr_chrM_fragments=10000
lower_thr_chrM_fragments=0

filter_uniquely_mapped_fragments=True # True or False; filtering out cells with promotor_enrichment not in the range defined below
# default number of uniquely mapped fragments
upper_thr_um=1000000
lower_thr_um=0

manual_thresholds = {}
if n_features_filter:
    manual_thresholds['features'] = {'min' : min_features, 'max' : max_features}

if mean_insertsize_filter:
    manual_thresholds['mean_insertsize'] = {'min' : lower_threshold_mis, 'max' : upper_threshold_mis}

if filter_pct_fp:
    manual_thresholds['pct_fragments_in_promoters'] = {'min' : lower_threshold_pct_fp, 'max' : upper_threshold_pct_fp}

if filter_n_fragments:
    manual_thresholds['TN'] = {'min' : lower_thr_fragments, 'max' : upper_thr_fragments}

if filter_chrM_fragments:
    manual_thresholds['CM'] = {'min' : lower_thr_chrM_fragments, 'max' : upper_thr_chrM_fragments}

if filter_uniquely_mapped_fragments:
    manual_thresholds['UM'] = {'min' : lower_thr_um, 'max' : upper_thr_um}

for key, value in manual_thresholds.items():
    if value['min'] is None or value['max'] is None:
        auto_thr = qc_filter.automatic_thresholds(adata, columns=[key])
        manual_thresholds[key] = auto_thr[key]
# get keys of manual_thresholds
keys = list(manual_thresholds.keys())
if automatic_thresholds:
    keys = list(manual_thresholds.keys())
    automatic_thresholds = qc_filter.automatic_thresholds(adata, columns=keys)
    qc_filter.apply_qc_thresholds(adata, automatic_thresholds)
else:
    qc_filter.apply_qc_thresholds(adata, manual_thresholds)



print(automatic_thresholds)