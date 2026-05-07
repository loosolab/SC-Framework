Analysis notebooks
==================

This section explains the design phyilosophy of the analysis notebooks.

The SC-Framework provides a collection of analysis notebooks either dedicated to a specific data type or data type agnostic, meaning it can be used with any data. Currently, there are three locations at which notebooks can be found:

- RNA_
- ATAC_
- general_ (data type independent notebooks. Move to the RNA/ATAC notebook folder before use)

.. _RNA: https://github.com/loosolab/SC-Framework/tree/main/rna_analysis/notebooks
.. _ATAC: https://github.com/loosolab/SC-Framework/tree/main/atac_analysis/notebooks
.. _general: https://github.com/loosolab/SC-Framework/tree/main/general_notebooks

Key aspects
-----------

General
~~~~~~~

- The notebooks are divided into **core notebooks**, i.e. notebooks that are required for analysis, and **downstream notebooks**, i.e. optional follow up notebooks to investigate specific topics. The **core notebooks** may be skipped if the data is pre-analyzed, except the *assembly notebook* to ensure proper data integration and handling.
- Each notebook represents a fundamental step in the single-cell analysis workflow, encapsulating key procedures such as data preprocessing, quality control, clustering, and marker identification.
- The ``config.yaml`` next to the notebooks defines the folder structure that is automatically created during analysis.
- Notebooks in ``general_notebooks/`` need to be moved to the respective analysis location before use (``*_analysis/notebooks/``).
- Notebooks should be executed in the order defined by their prefix, e.g., "01", "02", ...
- Notebooks with a character in their prefix may run in any order unless the prefix is followed by a number, e.g., "0A1" → "0A2" but "0B" before or after these two.
- The ``99-report.ipynb`` notebook creates an analysis report and should be run last.

In-file design
~~~~~~~~~~~~~~

All notebooks follow the same scheme. The notebook starts with a short introduction describing the aim of the notebook, followed by the imports (code loading to setup the following steps). Then a colored cell asks for general parameters related to the analysis. This is followed by code-blocks interspaced by descriptions to support and inputs required during the analysis (see the figure below).


.. image:: image/notebook_structure.png
   :width: 800

- Cells with blue background indicate user interaction.
- White cells are not intended for change and thus locked (unlock with `Runtools <https://jupyter-contrib-nbextensions.readthedocs.io/en/latest/nbextensions/runtools/readme.html>`_).
- The first cell in a notebook is hidden and runs on initialization. It colors the cells and checks if the package and notebook versions match.
- Input and output ``.h5ad`` files are stored in the ``adata/`` folder unless changed in the ``config.yaml``.

Available notebooks
~~~~~~~~~~~~~~~~~~~
.. toctree::
   :maxdepth: 2

   rna-notebooks/index
   atac-notebooks/index
   general-notebooks/index
