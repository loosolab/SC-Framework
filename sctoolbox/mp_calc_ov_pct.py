import pandas as pd
from collections import Counter
import multiprocessing as mp
import sctoolbox.utilities as utils

class MPOverlapPct():


    def __init__(self):

        self.merged_dict = None

    def calc_pct(self,
                 overlap_file,
                 fragments_file,
                 barcodes,
                 adata,
                 regions_name='list',
                 n_threads=8):

        # check if there was an overlap
        if not overlap_file:
            print("There was no overlap!")
            return

        # get unique barcodes from adata.obs
        barcodes = set(barcodes)

        # make columns names that will be added to adata.obs
        col_total_fragments = 'n_total_fragments'
        # if no name is given or None, set default name
        if not regions_name:
            col_n_fragments_in_list = 'n_fragments_in_list'
            col_pct_fragments = 'pct_fragments_in_list'
        else:
            col_n_fragments_in_list = 'n_fragments_in_' + regions_name
            col_pct_fragments = 'pct_fragments_in_' + regions_name

        # calculating percentage
        print('Calculating percentage...')
        # read overlap file as dataframe
        ov_fragments = pd.read_csv(overlap_file, sep="\t", header=None, chunksize=1000000)
        merged_ov_dict = self.mp_counter(ov_fragments, barcodes=barcodes, column=col_n_fragments_in_list, n_threads=n_threads)
        fragments = pd.read_csv(fragments_file, sep="\t", header=None, chunksize=1000000)
        merged_fl_dict = self.mp_counter(fragments, barcodes=barcodes, column=col_total_fragments, n_threads=n_threads)

        # add to adata.obs
        adata.obs[col_n_fragments_in_list] = adata.obs.index.map(merged_ov_dict).fillna(0)
        adata.obs[col_total_fragments] = adata.obs.index.map(merged_fl_dict).fillna(0)

        # calc pct
        adata.obs[col_pct_fragments] = adata.obs[col_n_fragments_in_list] / adata.obs[col_total_fragments]

        return adata


    def get_barcodes_sum(self, df, barcodes, col_name):
        # drop columns we dont need
        df.drop(df.iloc[:, 5:], axis=1, inplace=True)
        df.columns = ['chr', 'start', 'end', 'barcode', col_name]
        # remove barcodes not found in adata.obs
        df = df.loc[df['barcode'].isin(barcodes)]
        # drop chr start end columns
        df.drop(['chr', 'start', 'end'], axis=1, inplace=True)
        # get the sum of reads counts in each cell barcode
        df = df.groupby('barcode').sum()

        count_dict = df[col_name].to_dict()

        return count_dict


    def log_result(self, result):
        if self.merged_dict:
            self.merged_dict = dict(Counter(self.merged_dict) + Counter(result))
            # print('merging')
        else:
            self.merged_dict = result


    def mp_counter(self, fragments, barcodes, column, n_threads=8):

        pool = mp.Pool(n_threads, maxtasksperchild=48)
        jobs = []
        for chunk in fragments:
            job = pool.apply_async(self.get_barcodes_sum, args=(chunk, barcodes, column), callback=self.log_result)
            jobs.append(job)
        utils.monitor_jobs(jobs, description="Progress")
        pool.close()
        pool.join()
        # reset settings
        returns = self.merged_dict
        self.merged_dict = None

        return returns


if __name__ == '__main__':

    import episcanpy as epi

    fragments_file = '/mnt/workspace/jdetlef/data/bamfiles/sorted_Esophagus_fragments_sorted.bed'
    overlap_file = '/mnt/workspace/jdetlef/data/bamfiles/sorted_Esophagus_fragments_sorted_homo_sapiens.104.promoters2000.gtf_sorted_overlap.bed'
    h5ad_file = '/mnt/workspace/jdetlef/data/anndata/Esophagus.h5ad'

    adata = epi.read_h5ad(h5ad_file)
    barcodes = adata.obs['barcode']
   # barcodes = set(barcodes)
    instance = MPOverlapPct()

    adata = instance.calc_pct(overlap_file, fragments_file, barcodes, adata, regions_name='list', n_threads=8)

    print("Done")