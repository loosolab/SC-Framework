# loosolab_SC_RNA_framework

Here we are developing a set of notebooks for automation of scRNA-Seq of looso's lab.

# Workflow
The main workflow below is also presented in the powerpoint file.

![](image/scRNAseq_2x_2_.png)

## Setup utilities

To setup the custom single cell utilities, clone and install the tools with:
```
$ git clone https://gitlab.gwdg.de/loosolab/software/loosolab_sc_rna_framework.git
$ cd loosolab_sc_rna_framework
$ pip install .
```

Then, the tools are available for loading into the notebooks e.g. as:
```
> import sctoolbox.plotting
> sctoolbox.plotting.search_umap_parameters(adata, (...))
```

or directly as:
```
> from sctoolbox.plotting import search_umap_parameters
> search_umap_parameters(adata, (...))
```

# Usage
Open the notebook 1 (1_assembling_anndata.ipynb) and follow the instructions in the first two cells.

Example files to run the notebooks are available here $/mnt/agnerds/loosolab_SC_RNA_framework/examples


# Notebooks
The directory "nooteboks" has the Jupyter notebooks here developed.

**Notebook 1 (1_assembling_anndata.ipynb):**
	Assembly the 10X anndata to run velocity analysis, convert from Seurat to anndata object, assembly the 10X anndata object from public dataset.

**Notebook 2 (2_QC_filtering.ipynb):**
	QC and filtering steps.
