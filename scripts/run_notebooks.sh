cd notebooks
papermill --log-output 1_assembling_anndata.ipynb out.ipynb
papermill --log-output 2_QC_filtering.ipynb out.ipynb
papermill --log-output 3_batch.ipynb out.ipynb
papermill --log-output 4_clustering.ipynb out.ipynb
