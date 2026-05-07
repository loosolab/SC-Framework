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

3. Register environment as kernel
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

Download and copy the respective notebook folder (see above or check the :ref:`notebook-downloader-cli`) to your preferred location. The general notebooks can be copied in the same folder if required. Running the notebooks will automatically create a structure to store input/output files. The structure is defined via the ``config.yml``, next to the notebooks, and can be edited to your preference.

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
    │   └── *_analysis/
    │       ├── notebooks/
    │       └── notes.md
    ├── run_2/
    │   └── *_analysis/
    │       ├── notebooks/
    │       └── notes.md
    └── run_3/
        └── *_analysis/
            ├── notebooks/
            └── notes.md

.. _notebook-downloader-cli:

Notebook downloader CLI
^^^^^^^^^^^^^^^^^^^^^^^

The SC-Framework includes a command line interface, installed with the package, to automatically download the analysis notebooks and create the structure shown above.

.. code-block:: bash

  usage: SCF-analysis-setup [-h] -p PATH -n NAME -m {rna,atac} [-g] [-t TOKEN] [-r REFERENCE] [-v]

  Download and prepare analysis notebooks from the SC-Framework repository (https://github.com/loosolab/SC-Framework).

  options:
    -h, --help            show this help message and exit
    -p PATH, --path PATH  Path to the output directory.
    -n NAME, --name NAME  The name of the analysis. Will create a directory with this name in the given 'path'.
    -m {rna,atac}, --method {rna,atac}
                          The method matching your data type.
    -g, --exclude_general
                          Whether to download the general notebooks.
    -t TOKEN, --token TOKEN
                          A GitHub access token. Useful to circumvent API throttling.
    -r REFERENCE, --reference REFERENCE
                          Download notebooks of a specific version. Either a branch name, version tag or commit SHA. Will download the latest version from main on default.
    -v, --version         Show the SC-Framework version.

For example, a RNA analysis can be setup with:

.. note::
  The path given with ``-p`` must be created before running the command.

.. code-block:: bash

  SCF-analysis-setup -p </path/to/experiment/> -n initial_analysis -m rna


.. hint::
  Use a `GitHub token <https://docs.github.com/en/authentication/keeping-your-account-and-data-secure/managing-your-personal-access-tokens#creating-a-personal-access-token-classic>`_ to avoid rate limitations.

Docker
------

The SC-Framework provides a `Docker <https://www.docker.com/>`_ image for each release. This creates a static environment for long-term reproducibility and allows to take full advantage of single cell analysis in a HPC context. Each image contains all necessary software to conduct a full single cell analysis.

The images can be found in our GitLab `Container registry <https://gitlab.gwdg.de/loosolab/software/sc_framework/container_registry/>`_, where each image is tagged for its respective SC-Framework version.

**Download the image:**

.. code-block:: bash

  docker pull docker.gitlab.gwdg.de/loosolab/software/sc_framework:latest


**Run the image as a container:**

.. code-block:: bash

  docker docker run --rm -d -p 8123:8080 --volume /path/on/your/machine:/data docker.gitlab.gwdg.de/loosolab/software/sc_framework:latest bash -c "jupyter-notebook --allow-root --ip=0.0.0.0 --notebook-dir=/data --no-browser --NotebookApp.token='' --debug --port 8080"

This will start a docker container with a Jupyter notebook server that has the SC-Framework analysis environment.

.. table::
  :widths: 20 80
  :class: table-responsive

  ======================================== =================================================================================================================================
  Parameter                                Description                                                                                                                      
  ======================================== =================================================================================================================================
  ``--rm``                                 Automatically deletes the container when it stops.                                                                               
  ``-d``                                   Start the container in the background.                                                                                           
  ``-p 8123:8080``                         Forward container port `8080` to `8123` on your machine. This allows to access the Jupyter server with ``http://<your_ip>:8123``.
  ``--volume /path/on/your/machine:/data`` Makes a directory on your machine accessible from within the container.                                                          
  ======================================== =================================================================================================================================

.. note::
  ``--volume`` does not work with certain filesystems or can be restricted in HPC environments (see `here <https://docs.docker.com/engine/storage/bind-mounts/#configure-bind-propagation>`_). In such a case the directory will show up as empty. A workaround is to move your directory to a location that is supported e.g. the home directory (``/home/<user>/``).

The final part of the command (``bash -c "jupyter-notebook ..."``) takes care of starting the Jupyter server.

Now you can **access your container and analyse** notebooks that are located within the directory given via the ``-v`` parameter. Access the Jupyter server from your browser with ``http://<your_ip>:8123``.

**Find and stop the running container:**

The existing containers can be listed with

.. code-block:: bash

  docker container ls -a

and stopped with

.. code-block:: bash

  docker container stop <container_name>

