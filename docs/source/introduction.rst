Introduction
============

.. image:: image/sc_framework_overview.png
   :width: 800

The SC-Framework provides a comprehensive environment for single-cell data analysis. Starting from quantified data, it enables users to load, preprocess, visualize, and analyze their datasets. Core functionalities are implemented as a Python package and organized into structured workflows within Jupyter notebooks. The framework adheres to the `FAIR principles <https://doi.org/10.1038/s41597-022-01710-x>`_ (Findable, Accessible, Interoperable, Reusable) and includes extensive testing to ensure high quality, reproducibility, and usability.

**See our** `video series <https://youtube.com/playlist?list=PLTA07KFyG53ImxJKpoiPO9XiU_I5anMKp&si=hSFQx0CSiakG3Aj7>`_ **on YouTube.**

<iframe width="560" height="315" src="https://www.youtube.com/embed/videoseries?si=7g1uGaqruajYDGCo&amp;list=PLTA07KFyG53ImxJKpoiPO9XiU_I5anMKp" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share" referrerpolicy="strict-origin-when-cross-origin" allowfullscreen></iframe>

The package
-----------

The main functionalities of the SC-Framework are implemented as a Python package available on `GitHub <https://github.com/loosolab/SC-Framework>`_ and `PyPI <https://pypi.org/project/SC-Framework/>`_. The package includes comprehensive :doc:`API documentation <API/index>`, featuring code examples, expected outputs and relevant resources where applicable.

The package is split into three submodules:

1. ``utils``

General-purpose utility functions to support analysis workflows, including input/output operations, logging and table formatting.

2. ``tools``

Core analysis functions such as dimension reduction, normalization, marker identification and peak annotation.

3. ``plotting``

Visualization functions designed to generate plots and figures from results produced by the ``tools`` submodule.

The Jupyter notebooks
---------------------

The SC-Framework contains Jupyter notebook, which organize individual functions into analysis workflows. Different notebooks apply to different workflow sections such as quality control or annotation. 

TODO
- explain core vs downstream
- rna and atac
- general
- concept of repeating steps and notebooks (fig from paper; runs)

Analysis directory structure
----------------------------

TODO
- draw output structure in ascii (is there a better way?)
