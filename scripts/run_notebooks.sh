cd notebooks
papermill --log-output -k sctoolbox 1_assembling_anndata.ipynb out.ipynb
papermill --log-output -k sctoolbox 2_QC_filtering.ipynb out.ipynb
papermill --log-output -k sctoolbox 3_batch.ipynb out.ipynb
papermill --log-output -k sctoolbox 4_clustering.ipynb out.ipynb
