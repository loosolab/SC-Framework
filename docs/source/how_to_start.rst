How to start
============

The following sections describe the steps required to setup and conduct single cell analysis.

Installation
------------

The SC-Framework utilizes Jupyter notebooks and Conda environments to streamline the analysis and simplify the installation. Please ensure the following tools are available before proceeding.

**Requirements:**

- `Jupyter Notebook Server <https://jupyter.org/>`_
- `Mamba <https://mamba.readthedocs.io/en/latest/index.html>`_ (or Conda)

1. Environment
~~~~~~~~~~~~~~

First, download and create the environment, which is later used to do the analysis.

.. code-block:: bash

  curl -L -o sctoolbox_env.yml "https://raw.githubusercontent.com/loosolab/SC-Framework/main/sctoolbox_env.yml"
  mamba env create -f sctoolbox_env.yml
  mamba activate sctoolbox

2. Package
~~~~~~~~~~

Next, install the `SC-Framework package <https://pypi.org/project/SC-Framework/>`_ (aka sctoolbox), into the environment.

.. code-block:: bash

  pip install SC-Framework[all]

1. Register environment as kernel
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

With all the tools installed the next step is to allow using them in Jupyter. To do this, register the Conda environment as a Jupyter kernel.

.. code-block:: bash

  python -m ipykernel install --user --name <conda_env_name> --display-name "sctoolbox"

Analysis
--------

The SC-Framework provides a series of notebooks to streamline analysis. The notebooks can be found in the repository they are sorted into RNA-, ATAC-related and general notebooks, notebooks that can be applied to any data.

- RNA_
- ATAC_
- general_ (data type independent notebooks. Move to the RNA/ATAC notebook folder before use)

.. _RNA: https://github.com/loosolab/SC-Framework/tree/main/rna_analysis/notebooks
.. _ATAC: https://github.com/loosolab/SC-Framework/tree/main/atac_analysis/notebooks
.. _general: https://github.com/loosolab/SC-Framework/tree/main/general_notebooks

Notebook setup
~~~~~~~~~~~~~~

Download and copy the respective notebook folder (see above) to your preferred location. The general notebooks can be copied in the same folder if required. Running the notebooks will automatically create a structure to store input/output files. The structure is defined via the ``config.yml``, next to the notebooks, and can be edited to your preference.

.. note::
   Always start with the **assembly notebook** to ensure seamless data integration. Then skip to the analysis notebook of interest and load the data produced by the assembly notebook.

.. figure:: image/analysis_structure.png
   :alt: Exemplary dataflow of a analysis spanning multiple runs.
   :width: 200

   The recommended analysis structure, each analysis run, e.g. analysis on a subset of cell types, is contained in a separated folder, with the initial input taken from e.g. another run.

We recommend to structure the analysis to support multiple analysis runs that may occur during investigation. Further consider adding a ``notes.md`` or similar to document the intention of each analysis run.

::

    sc-framework-analysis/
    ├── run_1/
    │   ├── notebooks/
    │   └── notes.md
    ├── run_2/
    │   ├── notebooks/
    │   └── notes.md
    └── run_3/
        ├── notebooks/
        └── notes.md

# TODO CLI tool


Docker
------

The SC-Framework provides a `Docker <https://www.docker.com/>`_ image for each release. This creates a static environment for long-term reproducibility and allows to take full advantage of single cell analysis in a HPC context. Each image contains all necessary software to conduct a full single cell analysis.

The images can be found in our GitLab `Container registry <https://gitlab.gwdg.de/loosolab/software/sc_framework/container_registry/>`_, where each image is tagged for its respective SC-Framework version.

**Download the image:**

.. code-block:: bash

  docker pull docker.gitlab.gwdg.de/loosolab/software/sc_framework:latest

# TODO explain how to start and run notebooks with the container
