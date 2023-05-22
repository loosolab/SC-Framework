cd rna-notebooks
papermill --log-output -k sctoolbox --log-level INFO 01_assembling_anndata.ipynb out.ipynb
papermill --log-output -k sctoolbox --log-level INFO 02_QC_filtering.ipynb out.ipynb
papermill --log-output -k sctoolbox --log-level INFO 03_batch.ipynb out.ipynb
papermill --log-output -k sctoolbox --log-level INFO 04_clustering.ipynb out.ipynb
