# loosolab_SC_RNA_framework

A python framework for single cell analysis. It provides a plethora of functions for conducting common analysis tasks and respective visualization. It also includes a number of jupyter notebooks to further streamline the analysis process, making it easy to follow and reproduce analysis results.

# Workflow

![](image/scRNAseq.png)

# Installation

1. Download the repository. This will download the repository to your current folder.
```
git clone https://gitlab.gwdg.de/loosolab/software/loosolab_sc_rna_framework.git
```
2. Change working directory to repository.
```
cd loosolab_sc_rna_framework
```
3. Install analysis environment. Speed up installation by replacing `conda` with `mamba` (has to be installed).
```
conda env create -f sctoolbox_env.yml
```
4. Activate the environment.
```
conda activate sctoolbox
```
5. Register the environment as a jupyter kernel.
```
python -n ipykernel install --user --name sctoolbox --display-name "sctoolbox"
```

# Usage
1. Open your notebook and set the `sctoolbox` kernel

2. Open the first notebook (`1_assembling_anndata.ipynb`) and follow the instructions of the first two cells.

3. Example files to run the notebooks are available here:
```
/mnt/agnerds/loosolab_SC_RNA_framework/examples/assembling_10_velocity
```

4. The marker genes are stored here:
```
/mnt/agnerds/loosolab_SC_RNA_framework/marker_genes
```

# Notebooks
The main parts of the analysis workflow are provided as jupyter notebooks. They can be found in the `notebooks` directory.

## Notebook 1 (1_assembling_anndata.ipynb)

Assembly the 10X anndata to run velocity analysis, convert from Seurat to anndata object, assembly the 10X anndata object from public dataset.

## Notebook 2 (2_QC_filtering.ipynb)

QC and filtering steps.
