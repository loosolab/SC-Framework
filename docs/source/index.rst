Welcome to the documentation of the single cell framework!
==========================================================

.. image:: image/sc_framework_overview.png
   :width: 800

[![Static Badge](https://img.shields.io/badge/Zenodo-10.5281%2Fzenodo.16913120-blue)](https://doi.org/10.5281/zenodo.16913120)
[![Static Badge](https://img.shields.io/badge/bioRxiv-10.1101%2F2025.11.11.687874-%23bd2736)](https://doi.org/10.1101/2025.11.11.687874)
[![Release](https://gitlab.gwdg.de/loosolab/software/sc_framework/-/badges/release.svg)](https://github.com/loosolab/SC-Framework/releases)
![Coverage](https://gitlab.gwdg.de/loosolab/software/sc_framework/badges/main/coverage.svg?key_text=coverage&key_width=70)
![Pipeline](https://gitlab.gwdg.de/loosolab/software/sc_framework/badges/main/pipeline.svg?ignore_skipped=true)
[![Static Badge](https://img.shields.io/badge/Software-green?label=FAIR)](https://fairsoftwarechecklist.net/v0.2?f=31&a=32113&i=32311&r=133)
[![PyPI Version](https://img.shields.io/pypi/v/SC-Framework.svg?style=plastic)](https://pypi.org/project/SC-Framework/)
[![PyPI download month](https://img.shields.io/pypi/dm/SC-Framework.svg?style=plastic)](https://pypi.python.org/pypi/SC-Framework/)

**Visit the SC-Framework on `GitHub <https://github.com/loosolab/SC-Framework>`_!**

SC-Framework
------------
# TODO add a brief introduction

Quickstart
----------

Install the package from PyPI. We recommend to install into a conda environment.
``pip install SC-Framework[all]``

Setup a jupyter kernel
``python -m ipykernel install --user --name <conda_env_name> --display-name "sctoolbox"``

Load the analysis notebooks matching your data, e.g., `rna`.
``TODO sctoolbox.creators.setup_experiment``

Open the notebooks and start your analysis.

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

   tutorials/index
   rna-notebooks/index
   atac-notebooks/index
   general-notebooks/index
   API/index
   CHANGES
