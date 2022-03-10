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


# Notebooks
The directory "nooteboks" has the Jupyter notebooks under development.

Example files to run the notebooks are available here $/mnt/agnerds/loosolab_SC_RNA_framework/raw_data

# Modules instruction
The modules present in the notebook folder must be in the folder where your notebooks are located in your VM.
