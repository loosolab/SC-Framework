[![Static Badge](https://img.shields.io/badge/Zenodo-10.5281%2Fzenodo.16913120-blue)](https://doi.org/10.5281/zenodo.16913120)
[![Static Badge](https://img.shields.io/badge/bioRxiv-10.1101%2F2025.11.11.687874-%23bd2736)](https://doi.org/10.1101/2025.11.11.687874)
[![Release](https://gitlab.gwdg.de/loosolab/software/sc_framework/-/badges/release.svg)](https://github.com/loosolab/SC-Framework/releases)
![Coverage](https://gitlab.gwdg.de/loosolab/software/sc_framework/badges/main/coverage.svg?key_text=coverage&key_width=70)
![Pipeline](https://gitlab.gwdg.de/loosolab/software/sc_framework/badges/main/pipeline.svg?ignore_skipped=true)
[![Static Badge](https://img.shields.io/badge/Software-green?label=FAIR)](https://fairsoftwarechecklist.net/v0.2?f=31&a=32113&i=32311&r=133)
[![PyPI Version](https://img.shields.io/pypi/v/SC-Framework.svg?style=plastic)](https://pypi.org/project/SC-Framework/)
[![PyPI download month](https://img.shields.io/pypi/dm/SC-Framework.svg?style=plastic)](https://pypi.python.org/pypi/SC-Framework/)


# SC Framework

A python framework for single cell analysis. It provides a plethora of functions for conducting common analysis tasks and respective visualization. It also includes a number of jupyter notebooks to further streamline the analysis process, making it easy to follow and reproduce analysis results.

![](image/sc_framework_overview.png)

# Readthedocs
The SC framework is accompanied by an extensive documentation where detailed information regarding available notebooks, functions and a multitude of examples can be found. It can be accessed using the following link:

https://loosolab.pages.gwdg.de/software/sc_framework/

# Installation
## 1. Environment & Package installation
1. Download the repository. This will download the repository to your current folder.
```
git clone https://gitlab.gwdg.de/loosolab/software/sc_framework.git
```
2. Change the working directory to the newly created repository directory.
```
cd sc_framework
```
3. Install analysis environment. Note: using `mamba` is faster than `conda`, but this requires mamba to be installed.
```
mamba env create -f sctoolbox_env.yml
```
4. Activate the environment.
```
conda activate sctoolbox
```
5. Install the sctoolbox framework into the environment.
```
pip install .[all]
```
or
```
pip install SC-Framework[all]
```

## 2. Jupyter setup
Follow these steps if you want to run any of the provided jupyter notebooks.

1. If "jupyter-notebook" command is not available at this point: install notebook package.
```
pip install notebook
```
2. Register the environment as a jupyter kernel.
```
python -m ipykernel install --user --name sctoolbox --display-name "sctoolbox"
```

## 3. Git setup (for developers)
If you want to push changes to notebooks, you need to add the custom .gitconfig to the local .git config-file in order to enable clearing of notebook outputs:
```
git config --replace-all include.path "../.gitconfig"
```
**Make sure to activate the `sctoolbox` environment before staging the notebook file.**

# Analysis
## Idioms
### 1. Notebook Structure
All notebooks follow the same general template and rules:
- Notebooks typically begin with loading the anndata object and a cell for user inputs.
- Cells with a blue background require user input.
- Colorless cells are considered static and therefore shouldn't be changed. They are also locked and cannot be changed.
- The last step of a notebook is to save the analyzed anndata as a `.h5ad` file to be used by subsequent analysis steps (notebooks).

### 2. Module settings
The SC framework provides a settings class (`SctoolboxConfig`).
- `SctoolboxConfig` is used to set non-analysis options like output paths, number of threads, file prefixes, logging level. 
- The settings can be changed using the above mentioned class or by loading a config file (`sctoolbox.utils.settings_from_config`).

### 3. Logging
The framework provides two types of logging: 
1. **Traditional logging** written to a log file. This includes messages, warnings and errors that occur during the execution of functions. 
2. The second is **function logging**. This type of logging is added to the anndata object (`adata.uns["sctoolbox"]["log"]`). Whenever a function works on an anndata object (usually when receiving an anndata through a parameter), general information about the function call is stored inside the anndata object (name of the executed function, parameters, start time, who executed it, etc.).

The function log can be accessed using `sctoolbox.utils.get_parameter_table(adata)`

## Getting Started
Once the environment is set up and everything is installed, the analysis can be started using the provided jupyter notebooks. This can be done in a few steps:

1. Select the notebooks that fit to your data type (for example: scRNA or scATAC data). The notebooks are located in the following directories found in the root directory of the repository:
    - scRNA: `rna_analysis/`
    - scATAC: `atac_analysis/`

2. Copy the folder of the step above to your preferred analysis path.
```
cp -r rna_analysis/ </my/groubreaking/analysis/>
```

3. (optional) Some notebooks are data type independent and are located in `general_notebooks/`. These can be copied to the same directory as the other analysis notebooks. E.g.:
```
cp general_notebooks/pseudotime_analysis.ipynb </my/groubreaking/analysis/rna_analysis/notebooks/>
```

4. Access the notebooks in the directory and run them perform your analysis (general notebooks should be run last).

## Folder structure
While going through the analysis notebooks, a folder structure is created to store all the results and intermediate files (figures, `.h5ad` files, tables etc.). The default structure is created in the `*_analysis` directory that contains the notebooks. It is independent of data type.

```
└── *_analysis
    ├── adatas
    │   └── *.h5ad
    ├── figures
    │   ├── 02_QC
    │   │   ├── *.png
    │   │   └── *.pdf
    │   ├── 03_batch_correction
    │   │   ├── *.png
    │   │   └── *.pdf
    │   ...
    ├── logs
    │   └── *.txt
    ├── notebooks
    │   ├── *.ipynb
    │   └── config.yml
    └── tables
        ├── 02_QC
        │   ├── *.xlsx
        │   └── *.tsv
        ├── 03_batch_correction
        │   ├── *.xlsx
        │   └── *.tsv
        ...
```

The `*_analysis` directory contains up to five subdirectories. In the beginning, there is only the `notebooks` directory it contains all of the analysis notebooks and a `config.yml`. The `config.yml` holds general settings (e.g. paths) for each notebook. It is loaded earlier in the execution of a notebook and can be adjusted as needed. The rest of the subdirectories are created during the execution of the notebooks as they are needed. `adatas/` contains intermediate `.h5ad` files created at the end of each notebook. `figures/` contains all of the plots created during analysis. `logs/` contains log-files and `tables/` stores additional result tables. The directories `figures/` and `tables/` are divided into one directory per notebook.

# FAQ
### Q: I have an old/ already started analysis. How do I find out what was done or who was responsible?

**A:** The function logging which contains this information, can be accessed using `sctoolbox.utils.get_parameter_table(adata)`. For more information see [here](https://loosolab.pages.gwdg.de/software/sc_framework/API/utils.html#sctoolbox.utils.decorator.get_parameter_table).

### Q: My `.h5ad` file is already pre-analyzed. I want to skip some of the analysis notebooks. What do I do?

**A:** You should always start with the assembly notebook (the first notebook), which ensures a proper output structure. Afterwards, go ahead with the notebook you want to run.

### Q: I have encountered a bug, I have a feature request, there is something I need help with or want to discuss.

**A:** We are always happy to help. If you encounter something that needs attention open an issue with a detailed explanation and if possible a small code example. Thank you!
