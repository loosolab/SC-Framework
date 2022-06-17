# loosolab_SC_RNA_framework

Here we are developing a set of notebooks for automation of scRNA-Seq of looso's lab.

# Workflow

![](image/scRNAseq.png)

## Installing

Create and activate a conda environment:
```
$ conda create -n scRNAseq_autom python=3.7 ipykernel
$ conda activate scRNAseq_autom
```
Register the environment in the kernel
```
$ python -n ipykernel install --user --name scRNAseq_autom --display-name "scRNAseq_autom"
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
$ cd loosolab_sc_rna_framework
$ git checkout dev
$ pip install .
```
Copy the notebooks to another directory
```
$ cp notebooks/* /home/notebooks 
```

# Usage
1- Open your notebook and set the scRNAseq_autom kernel

2- Open the notebook 1 (1_assembling_anndata.ipynb) and follow the instructions in the first two cells.

3- Example files to run the notebooks are available here
```
$/mnt/agnerds/loosolab_SC_RNA_framework/examples/assembling_10_velocity
```

4- The marker genes are stored here
```
$/mnt/agnerds/loosolab_SC_RNA_framework/marker_genes
```

# Notebooks
The directory "nooteboks" has the Jupyter notebooks here developed.

**Notebook 1 (1_assembling_anndata.ipynb):**

	Assembly the 10X anndata to run velocity analysis, convert from Seurat to anndata object, assembly the 10X anndata object from public dataset.

**Notebook 2 (2_QC_filtering.ipynb):**

	QC and filtering steps.
