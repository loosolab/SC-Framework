cd notebooks
papermill --log-output -k sctoolbox --log-level DEBUG 1_assembling_anndata.ipynb out.ipynb
papermill --log-output -k sctoolbox --log-level DEBUG 2_QC_filtering.ipynb out.ipynb
papermill --log-output -k sctoolbox --log-level DEBUG 3_batch.ipynb out.ipynb
papermill --log-output -k sctoolbox --log-level DEBUG 4_clustering.ipynb out.ipynb
