Introduction
============

.. image:: image/sc_framework_overview.png
   :width: 800

The SC-Framework provides a comprehensive environment for single-cell data analysis. Starting from quantified data, it enables users to load, preprocess, visualize, and analyze their datasets. Core functionalities are implemented as a Python package and organized into structured workflows within Jupyter notebooks. The framework adheres to the `FAIR principles <https://doi.org/10.1038/s41597-022-01710-x>`_ (Findable, Accessible, Interoperable, Reusable) and includes extensive testing to ensure high quality, reproducibility, and usability.

**See our** `video series <https://youtube.com/playlist?list=PLTA07KFyG53ImxJKpoiPO9XiU_I5anMKp&si=hSFQx0CSiakG3Aj7>`_ **on YouTube.**

.. youtube:: H_0DNURMOvM
   :align: center
   :width: 100%

The package
-----------

The main functionalities of the SC-Framework are implemented as a Python package available on `GitHub <https://github.com/loosolab/SC-Framework>`_ and `PyPI <https://pypi.org/project/SC-Framework/>`_. The package includes comprehensive :doc:`API documentation <API/index>`, featuring code examples, expected outputs and relevant resources where applicable.

The package is split into three submodules:

1. ``utils``
~~~~~~~~~~~~

General-purpose utility functions to support analysis workflows, including input/output operations, logging and table formatting.

2. ``tools``
~~~~~~~~~~~~

Core analysis functions such as dimension reduction, normalization, marker identification and peak annotation.

3. ``plotting``
~~~~~~~~~~~~~~~

Visualization functions designed to generate plots and figures from results produced by the ``tools`` submodule.

The Jupyter notebooks
---------------------

The SC-Framework includes Jupyter notebooks that organize individual functions into structured analysis workflows. Different notebooks are designed for specific workflow sections, such as quality control or annotation. The framework provides core notebooks—that manage essential steps required in every analysis, such as quality control—and downstream notebooks tailored to address specific research questions (see the top image).
The SC-Framework offers notebooks for various types of analysis:

 - Transcriptome analysis (`RNA <https://github.com/loosolab/SC-Framework/tree/main/rna_analysis/notebooks>`_)
 - Chromatin accessibility analysis (`ATAC <https://github.com/loosolab/SC-Framework/tree/main/atac_analysis/notebooks>`_)
 - Data type-agnostic notebooks that can be applied to any dataset (`general <https://github.com/loosolab/SC-Framework/tree/main/general_notebooks>`_)

.. figure:: image/analysis_progress.png
   :alt: Typicall analysis workflow with iterative analysis to find optimal parameters and branching into concurrent runs.
   :width: 400

   A typical analysis workflow involves iterative steps, where individual cells or entire notebooks may be repeated to identify optimal parameters. The analysis may also branch into concurrent runs to explore different hypotheses or investigative paths.

Single-cell analysis often requires iterative exploration of the dataset to identify optimal parameter combinations. Moreover, ongoing analysis may necessitate splitting the workflow into multiple concurrent branches, for example to analyze a specific subset of cell types while continuing the analysis on the full dataset. The SC-Framework is designed to accommodate both needs by enabling the re-execution of individual cells or entire notebooks within a workflow (see image above). This flexibility supports dynamic, adaptive analysis pipelines tailored to evolving research questions.

Analysis directory structure
----------------------------

.. figure:: image/result_structure.png
   :alt: The folder structure of an analysis run. The folder creation is on-demand. The main folders store ``.h5ad`` files (adata), visualization (figures), logging information (logs) and tables (tables). The folders are created next to the ``notebooks`` folder that contains the analysis notebooks.
   :width: 400

   The folder structure created during an analysis run.

Running the Jupyter notebooks automatically creates an output structure. Directories are created on-demand based on the ``config.yaml`` found in the notebooks folder (adjust the config-file to your needs).

::

    run/
    ├── adatas/
    │   ├── anndata_1.h5ad
    │   └── anndata_annotated.h5ad
    ├── figures/
    │   └── 02_QC/
    │       └── cell_filtering.png
    ├── logs/
    │   └── 02_log.txt
    ├── notebooks/
    │   ├── 02_QC_filtering.ipynb
    │   └── config.yaml
    ├── report/
    └── tables/
        └── 0A1_receptor_ligand/
            └── rl_interaction_table.tsv

**Description:**

- ``run/`` - Contains everything related to the analysis run
- ``adatas/`` - The individual AnnData-files in ```.h5ad`` format. Created at the end of each notebook
- ``figures/`` - The images created during the analysis
- ``logs`` - Logging files for each notebook
- ``notebooks`` - Contains the analysis notebooks and a ``config.yaml`` that defines the folder-structure
- ``report`` - Files used to render an analysis report PowerPoint
- ``tables`` - Tables created during the analysis
