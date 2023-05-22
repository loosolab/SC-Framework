#!/usr/bin/env bash

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
echo $SCRIPT_DIR

# Run RNA notebooks
cd $SCRIPT_DIR/../rna-notebooks
papermill -k sctoolbox --log-level INFO 01_assembling_anndata.ipynb out.ipynb
#papermill --log-output -k sctoolbox --log-level INFO 01_assembling_anndata.ipynb out.ipynb
papermill -k sctoolbox --log-level INFO 02_QC_filtering.ipynb out.ipynb
papermill -k sctoolbox --log-level INFO 03_batch.ipynb out.ipynb
papermill -k sctoolbox --log-level INFO 04_clustering.ipynb out.ipynb
