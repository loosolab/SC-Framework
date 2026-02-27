Welcome to the documentation of the single cell framework!
==========================================================

.. image:: image/sc_framework_overview.png
   :width: 800

.. image:: https://img.shields.io/badge/Zenodo-10.5281%2Fzenodo.16913120-blue
   :target: https://doi.org/10.5281/zenodo.11065517
   :alt: Zenodo DOI 10.5281/zenodo.11065517

.. image:: https://img.shields.io/badge/bioRxiv-10.1101%2F2025.11.11.687874-%23bd2736
   :target: https://doi.org/10.1101/2025.11.11.687874
   :alt: bioRxiv DOI 10.1101/2025.11.11.687874

.. image:: https://gitlab.gwdg.de/loosolab/software/sc_framework/-/badges/release.svg
   :target: https://github.com/loosolab/SC-Framework/releases
   :alt: SC-Framework Release

.. image:: https://gitlab.gwdg.de/loosolab/software/sc_framework/badges/main/coverage.svg?key_text=coverage&key_width=70
   :alt: Test coverage

.. image:: https://gitlab.gwdg.de/loosolab/software/sc_framework/badges/main/pipeline.svg?ignore_skipped=true
   :alt: Pipeline status

.. image:: https://img.shields.io/badge/Software-green?label=FAIR
   :target: https://fairsoftwarechecklist.net/v0.2?f=31&a=32113&i=32311&r=133
   :alt: FAIR checklist

.. image:: https://img.shields.io/pypi/v/SC-Framework.svg?style=plastic
   :target: https://pypi.org/project/SC-Framework/
   :alt: PyPI Version

.. image:: https://img.shields.io/pypi/dm/SC-Framework.svg?style=plastic
   :target: https://pypi.python.org/pypi/SC-Framework/
   :alt: PyPI download month

**Visit the SC-Framework on** `GitHub <https://github.com/loosolab/SC-Framework>`_ **!**

SC-Framework
------------

The SC-Framework provides a full environment for single cell analysis. It is split into a python package, aimed to conduct common tasks and visualization, and Jupyter notebooks to streamline the individual analysis steps. This in combination with extensive documentation, tutorials, logging, etc. provides a FAIR framework to analyze single cell data in a rapid, comprehensible and reproducible manner.

Quickstart
----------

**Prerequesites: A** `Jupyter Notebook Server <https://jupyter.org/>`_ **and** `Mamba <https://mamba.readthedocs.io/en/latest/index.html>`_ **(or Conda).**

Create and activate the SC-Framework environment (downloaded from our GitHub repository).

.. code-block:: bash

  curl -L -o sctoolbox_env.yml "https://raw.githubusercontent.com/loosolab/SC-Framework/main/sctoolbox_env.yml"
  mamba env create -f sctoolbox_env.yml
  mamba activate sctoolbox

Install the package from PyPI. We recommend to install into a conda environment.

.. code-block:: bash

  pip install SC-Framework[all]

Setup a jupyter kernel

.. code-block:: bash

  python -m ipykernel install --user --name <conda_env_name> --display-name "sctoolbox"

Open the notebooks matching your data, e.g., `rna`, and start your analysis. The notebooks can be found in the respective folders:

- RNA_
- ATAC_
- general_ (data type independent notebooks. Move to the RNA/ATAC notebook folder before use)

.. _RNA: https://github.com/loosolab/SC-Framework/tree/main/rna_analysis/notebooks
.. _ATAC: https://github.com/loosolab/SC-Framework/tree/main/atac_analysis/notebooks
.. _general: https://github.com/loosolab/SC-Framework/tree/main/general_notebooks

Other links
-----------

Interested in our group and our projects? :)

- `Homepage <https://github.molgen.mpg.de/pages/loosolab/www/index.html>`_
- `GitHub <https://github.com/loosolab>`_
- `CPI Repository <https://bioinformatics.mpi-bn.mpg.de/bcu-repository-bn/home>`_
- `Bulk Pipelines <https://loosolab.pages.gwdg.de/software/bulk_pipelines/>`_
- `Bluesky <https://web-cdn.bsky.app/profile/did:plc:lsrycjv2xlemztqd5ayi3jho>`_
- `X <https://x.com/loosolab>`_
- `YouTube <https://www.youtube.com/@loosolab>`_
- `Contact <https://github.molgen.mpg.de/pages/loosolab/www/contact/>`_

.. toctree::
   :maxdepth: 2
   :caption: Contents:
   :hidden:

   introduction
   tutorials/index
   rna-notebooks/index
   atac-notebooks/index
   general-notebooks/index
   API/index
   CHANGES
   cite
