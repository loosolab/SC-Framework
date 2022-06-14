# loosolab_SC_RNA_framework

Here we are developing a set of notebooks for automation of scRNA-Seq of looso's lab.

# Workflow

![](image/scRNAseq.png)

## Setup utilities

Create and activate a conda environment:
```
$ conda create -n scRNAseq python=3.7
$ conda activate scRNAseq
```
Move to your workspace before installation. NOTE.: Install the scRNAseq automator in your workspace is crucial for a proper execution:
```
$ cd /mnt/workspace/YOUR_WORKSPACE
$ mkdir scRNAseq_autom
$ cd scRNAseq_autom
```

Clone and install the tools with:
```
$ git clone https://gitlab.gwdg.de/loosolab/software/loosolab_sc_rna_framework.git
$ pip install .
```

# Usage
Open the notebook 1 (1_assembling_anndata.ipynb) and follow the instructions in the first two cells.

Example files to run the notebooks are available here $/mnt/agnerds/loosolab_SC_RNA_framework/examples

The marker genes are stored in $/mnt/agnerds/loosolab_SC_RNA_framework/marker_genes

# Notebooks
The directory "nooteboks" has the Jupyter notebooks here developed.

**Notebook 1 (1_assembling_anndata.ipynb):**

	Assembly the 10X anndata to run velocity analysis, convert from Seurat to anndata object, assembly the 10X anndata object from public dataset.

**Notebook 2 (2_QC_filtering.ipynb):**

	QC and filtering steps.
