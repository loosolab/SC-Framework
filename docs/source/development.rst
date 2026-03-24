Development
===========

This section is intended for developers. It contains information about the recommended development setup, design choices and contribution requirements.

**The development takes place on** `GitLab <https://gitlab.gwdg.de/loosolab/software/sc_framework>`_ **!**

**Want to implement something new?** Always check existing functions. The SC-Frameworks package (sctoolbox) contains a lot of functions maybe you are lucky and someone already implemented your desired functionality.

Git
---

Clear notebook output
~~~~~~~~~~~~~~~~~~~~~

We require notebooks to not contain executed code, i.e. output results. Removing them is tedious, which is why we provide a `.gitconfig` to automatically clear all outputs before a notebook is commited. If you want to push changes to notebooks, you need to add the custom `.gitconfig`, provided in this repository, to the local Git installation in order to enable clearing of notebook outputs:

.. code-block:: bash

  git config --replace-all include.path "../.gitconfig"

.. note::
  Make sure to activate the sctoolbox environment before staging the notebook file.

Instead, you may **delete the outputs manually** from within the notebook. This can be done using the menu on the top of the notebook:

1. `Cell -> All Outputs -> Clear` to delete execution results
2. `Cell -> Execution Timings -> Clear (all)` to delete the runtime of each cell
3. Save the notebook to ensure all edits are commited.

Notebooks
---------

This is a collection of rules that should be followed when changing or adding a notebook.

Location
~~~~~~~~

Notebooks are currently stored at three locations:

- `rna_analysis/notebooks <https://gitlab.gwdg.de/loosolab/software/sc_framework/-/tree/460f6d3d0ab44ac24f4b2df9dab372a6b64bd56a/rna_analysis/notebooks>`_
- `atac_analysis/notebooks <https://gitlab.gwdg.de/loosolab/software/sc_framework/-/tree/460f6d3d0ab44ac24f4b2df9dab372a6b64bd56a/atac_analysis/notebooks>`_
- `general_notebooks <https://gitlab.gwdg.de/loosolab/software/sc_framework/-/tree/460f6d3d0ab44ac24f4b2df9dab372a6b64bd56a/general_notebooks>`_

Which shows the inteded data type the notebook should be used with. General notebooks are data type agnostic, meaning, they can be used with any type of data. New notebooks should be appropriately placed within this structure.

.. note::
  General notebooks should be moved to the specific data type directory, e.g. `rna_analysis/notebooks` before use.

Configuration
~~~~~~~~~~~~~

Global parameters for each notebook are defined through a `config.yaml` stored next to the notebooks. This file contains a section for each notebook primarily defining in- and output paths.

.. code-block::

  "pseudotime_analysis":
      adata_input_dir: "../adatas/"
      adata_output_dir: "../adatas/"
      figure_dir: "../figures/pseudotime/"
      log_file: "../logs/pseudotime_analysis_log.txt"
      overwrite_log: True
      report_dir: "../report/pseudotime_analysis/"

The above shows the *pseudotime notebook* parameters, which are forwarded to the `sctoolbox.settings`. The `config.yaml` is loaded from within the notebook as follows:

.. code-block:: python

  sctoolbox.settings.settings_from_config("config.yaml", key="pseudotime_analysis")

Kernel
~~~~~~

All notebooks should use the `sctoolbox` kernel per default. This ensures proper code execution without requiring intervention by the user.

Formatting
~~~~~~~~~~

Here are the formatting rules for the notebooks.

Notebook sections
^^^^^^^^^^^^^^^^^
Each notebook has to be separated into sections separated by markdown headers.
   
We allow 4 levels of headers:

.. code-block::

  # Level 1
  ## Level 2
  ### Level 3
  #### Level 4
  
Each subsection of the notebook needs to be at level 2, with subsections of level 2 being level 3 and so on.  
Each subsection and its subsections have to be numbered:

.. code-block::

  # Notebook Title
  ## 1 - Subsection 1
  ### 1.1 - Sub-Subsection 1
  #### 1.1.1 - Sub-Sub-Subsection
  ### 1.2 - Sub-Subsection 2
  ## 2 - Subsection 2

If a subsection at level 2 has a subsection it has to be underlined with a line of 2px thickness:

.. code-block::

  <hr style="border:2px solid black"> </hr>

If a subsection at level 3 has a subsection it has to be underlined with a line of 1px thickness:

.. code-block::

  <hr style="border:1px solid black"> </hr>

The only exception to this rule is the first subsection.

Fixed cells
^^^^^^^^^^^

- The first cell of the notebook has to be a **hidden init cell**. (See input cells for more info)
- The second cell of the notebook contains only the underlined title.
    - The title of the notebooks needs to be at level 1 with no other header being allowed to be at this level.
    - The title of the notebook is underlined with a line of 2px thickness:

      .. code-block::

        <hr style="border:2px solid black"> </hr>

- The first subsection of a notebook needs to be a description:

  .. code-block::

    ## 1 - Description

Separation lines
^^^^^^^^^^^^^^^^

After each section of levels 2 and 3, a separation line has to be inserted as a markdown cell:

.. code-block::

  ___

- This cell should not contain any other text.
- In level 4 the sections are **not** separated by a line.

Input cells
^^^^^^^^^^^

- Each input cell has to be colored blue (`PowderBlue`).
- Before each input cell a markdown cell containing the following has to be placed:

  .. code-block::

    <h1><center>⬐ Fill in input data here ⬎</center></h1>

- After each input cell a separation line (Markdown cell) has to be placed:

  .. code-block::

    ___

Locked cells
^^^^^^^^^^^^

Each cell has to be locked using the `runtools` nbextension except for the input cells. See `here <https://jupyter-contrib-nbextensions.readthedocs.io/en/latest/nbextensions/runtools/readme.html>`_ for a description on how to use `runtools`.

Mark important information cells
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Important information should be marked with a special HTML `div`-tag. This will color the Markdown cell red. Make sure there is a space before and after the text.

.. code-block::

  <div class="alert alert-block alert-danger">

      your text...

  </div>

Cells
~~~~~

Commonly used cells within the analysis notebooks.

First cell
^^^^^^^^^^

This is a hidden initialization cell, meaning it will run when the notebook is opened. Below is an example for a first cell that may be copied.

.. code-block:: python

    from sctoolbox.utils.jupyter import bgcolor, _compare_version

    # change the background of input cells
    bgcolor("PowderBlue", select=[2, 4, 7])

    nb_name = "pseudotime_analysis.ipynb"

    _compare_version(nb_name)

The contents of this cell typical include:

- the `bgcolor` function defining which cells to highlight
- the name of the file to allow if the version of the analysis notebooks matches the sctoolbox package version

The Nbextension are likely pre-installed. See `here <https://github.com/Jupyter-contrib/jupyter_nbextensions_configurator?tab=readme-ov-file#usage>`_ for an explanation on how to enable them. You may need to refresh the notebook page after activating an extension.

1. Initialization cell:
    Requires the `Initialization cells` Nbextension.

    Go to the top menu and click `View -> Cell Toolbar -> Initialization Cell`. A checkbox will appear above each cell. Mark the ones of cells that should run on initialization. Click `View -> Cell Toolbar -> Initialization Cell` again to hide the checkbox and save the notebook.

2. Hide cells:
    Requires the `Runtools` Nbextension.

    See `here <https://jupyter-contrib-nbextensions.readthedocs.io/en/latest/nbextensions/runtools/readme.html>`_ for a description on how to hide cell inputs.

    1. Mark the respective cell
    2. Hide the cell input
    3. Unmark the cell (The cell will be hidden, you have the keep close track on the button presses.)

Setup cell
^^^^^^^^^^^

This cell loads all the required packages, sets the notebook related settings and shows the versions of important packages.

.. code-block:: python

    import pandas as pd
    import scanpy as sc
    from pathlib import Path

    import sctoolbox.utils as utils
    import sctoolbox.tools as tools
    import sctoolbox.plotting as pl
    from sctoolbox import settings

    settings.settings_from_config("config.yaml", key="03")

    # Set additional options for figures
    sc.set_figure_params(vector_friendly=True, dpi_save=600, scanpy=False)

    with pd.option_context("display.max.rows", None, "display.max_colwidth", None):
        display(utils.general.get_version_report(report="versions.yml"))

Data overview
^^^^^^^^^^^^^

Having an overview of the data is crucial for e.g. selecting the correct clustering. This cell shows the `AnnData.obs` (cell related info) and `AnnData.var` (gene related info) without truncating columns. May be edited depending on use case, e.g. only show `AnnData.var`.

.. code-block:: python

    with pd.option_context("display.max.rows", 5, "display.max.columns", None):
        display(adata)
        display(adata.obs)
        display(adata.var)

Can also be combined with file loading:

.. code-block:: python

    adata = utils.adata.load_h5ad("anndata_2.h5ad")

    with pd.option_context("display.max.rows", 5, "display.max.columns", None):
        display(adata)
        display(adata.obs)
        display(adata.var)

Final cells
^^^^^^^^^^^

The final cells in each notebook 

- save the `AnnData` object as an `.h5ad` file so it can be used in following analysis
- and close the logging file.

.. code-block:: python

    # Saving the data
    adata_output = "anndata_1.h5ad"
    utils.adata.save_h5ad(adata, adata_output)

.. code-block:: python

    sctoolbox.settings.close_logfile()

Descriptions
~~~~~~~~~~~~

# TODO add something about how descriptions in the notebooks should look

Plotting
~~~~~~~~

# TODO how should plotting be done?

- always as pdf?
- numbered output files
- mention report saving
- mention the internal plot save function
