cd rna-notebooks
papermill --log-output -k sctoolbox --log-level DEBUG 01_assembling_anndata.ipynb out.ipynb
echo "finished running notebook 1"
papermill --log-output -k sctoolbox --log-level DEBUG 02_QC_filtering.ipynb out.ipynb
papermill --log-output -k sctoolbox --log-level DEBUG 03_batch.ipynb out.ipynb
papermill --log-output -k sctoolbox --log-level DEBUG 04_clustering.ipynb out.ipynb
