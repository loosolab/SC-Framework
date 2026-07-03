Development
===========

This section is intended for developers. It contains information about the recommended development setup, design choices and contribution requirements.

**The development takes place on** `GitLab <https://gitlab.gwdg.de/loosolab/software/sc_framework>`_ **!**

**Want to implement something new?** Always check existing functions. The SC-Frameworks package (sctoolbox) contains a lot of functions maybe you are lucky and someone already implemented your desired functionality.

Setup
-----

The recommended development setup is the same conda environment used to run the analysis notebooks, extended with the optional dependencies required for testing, linting, spellchecking and building the documentation.

Environment and package
~~~~~~~~~~~~~~~~~~~~~~~

First clone the repository, create the ``sctoolbox`` conda environment and install the package into it. Using `mamba <https://mamba.readthedocs.io/>`_ is faster than ``conda`` but requires mamba to be installed.

.. code-block:: bash

  # clone and enter the repository
  git clone https://gitlab.gwdg.de/loosolab/software/sc_framework.git
  cd sc_framework

  # create and activate the environment
  mamba env create -f sctoolbox_env.yml
  conda activate sctoolbox

  # install sctoolbox with all optional dependencies in editable mode
  pip install -e .[all]

The ``[all]`` extra pulls in every optional dependency group (see the ``[project.optional-dependencies]`` section of ``pyproject.toml``). Install only a subset, e.g. ``pip install -e .[atac]``, if you do not need everything. The ``-e``/``--editable`` flag installs the package in *editable* mode so your changes to ``src/sctoolbox`` take effect without reinstalling.

Development dependencies
~~~~~~~~~~~~~~~~~~~~~~~~

The tooling required to test and lint the project is declared as `dependency groups <https://packaging.python.org/en/latest/specifications/dependency-groups/>`_ in the ``[dependency-groups]`` section of ``pyproject.toml``. Unlike the optional dependencies above, these groups are **not** part of the distributed package and are installed separately with pip's ``--group`` flag (requires ``pip >= 25.1``):

.. list-table::
  :header-rows: 1
  :widths: 20 30 50

  * - Group
    - Packages
    - Purpose
  * - ``test``
    - pytest, pytest-mock, pytest-cov, pytest-html, openpyxl
    - Run the unit tests and produce coverage/HTML reports.
  * - ``lint``
    - ruff
    - Lint and check docstrings (the binding test command starts with ``ruff check``).
  * - ``spellcheck``
    - codespell[toml]
    - Spellcheck the code and documentation.
  * - ``docs``
    - sphinx, sphinx-rtd-theme, sphinx-exec-code, nbsphinx, ...
    - Build this documentation locally.

Install the groups you need into the activated ``sctoolbox`` environment:

.. code-block:: bash

  # everything needed to develop, test and lint
  pip install -e .[all] --group test --group lint --group spellcheck

  # add the docs toolchain if you want to build the documentation
  pip install --group docs

The ``--group`` flag can be combined with a normal install target, so the package and its development dependencies can be installed in a single command. After installation you can run the test suite, the linter and the spellchecker:

.. code-block:: bash

  ruff check         # lint + docstring checks
  pytest             # unit tests with coverage
  codespell          # spellcheck (uses the config in pyproject.toml)

One test dependency is **not** part of these groups: `scar <https://github.com/Novartis/scar.git>`_ is installed from git rather than PyPI and is deliberately left out of ``[all]`` and ``sctoolbox_env.yml`` because of its size. The CI test job installs it to exercise the scar-related functions. Install it if you want to run those tests locally:

.. code-block:: bash

  pip install git+https://github.com/Novartis/scar.git

Verifying the environment
~~~~~~~~~~~~~~~~~~~~~~~~~~~

To confirm an environment carries everything the workflow needs, run ``scripts/check_dev_env.py`` inside it. It reads the required packages from the ``[dependency-groups]`` table of ``pyproject.toml`` and reports anything missing or version-mismatched, alongside an editable ``sctoolbox`` install, the ``scar`` test dependency, and — for the docs build — the system ``pandoc`` binary:

.. code-block:: bash

  python scripts/check_dev_env.py                  # full dev environment
  python scripts/check_dev_env.py --scope package  # just the package test/lint/spellcheck tooling

A zero exit status means the environment is complete; otherwise the report lists the gaps and the command to install them.

Git
---

Clear notebook output
~~~~~~~~~~~~~~~~~~~~~

We require notebooks to not contain executed code, i.e. output results. Removing them is tedious, which is why we provide a `.gitconfig` to automatically clear all outputs before a notebook is committed. If you want to push changes to notebooks, you need to add the custom `.gitconfig`, provided in this repository, to the local Git installation in order to enable clearing of notebook outputs:

.. code-block:: bash

  git config --replace-all include.path "../.gitconfig"

.. note::
  Make sure to activate the sctoolbox environment before staging the notebook file.

Instead, you may **delete the outputs manually** from within the notebook. This can be done using the menu on the top of the notebook:

1. `Cell -> All Outputs -> Clear` to delete execution results
2. `Cell -> Execution Timings -> Clear (all)` to delete the runtime of each cell
3. Save the notebook to ensure all edits are committed.

Notebooks
---------

This is a collection of rules that should be followed when changing or adding a notebook.

.. note::
  Keep the code within notebooks minimal. Consider moving large code-blocks into functions and add them to the sctoolbox package.

Location
~~~~~~~~

Notebooks are currently stored at three locations:

- `rna_analysis/notebooks <https://gitlab.gwdg.de/loosolab/software/sc_framework/-/tree/460f6d3d0ab44ac24f4b2df9dab372a6b64bd56a/rna_analysis/notebooks>`_
- `atac_analysis/notebooks <https://gitlab.gwdg.de/loosolab/software/sc_framework/-/tree/460f6d3d0ab44ac24f4b2df9dab372a6b64bd56a/atac_analysis/notebooks>`_
- `general_notebooks <https://gitlab.gwdg.de/loosolab/software/sc_framework/-/tree/460f6d3d0ab44ac24f4b2df9dab372a6b64bd56a/general_notebooks>`_

A notebook is either *data-specific*, meaning it is designed to work with a specific type of data (e.g. RNA, ATAC) or *data-agnostic*, meaning it may be used with any type of data. This is reflected by the notebooks location and should be considered when developing/adding a new notebook.

General notebooks are stored in a separate directory to avoid maintaining multiple copies of the same notebook. However, they are designed to run in the same environment as the *data-specific* notebooks and should therefore be copied to the *data-specific* notebook directory before use.

.. note::
  General notebooks should be moved to a *data-specific* directory, e.g. `rna_analysis/notebooks` before use.

Configuration
~~~~~~~~~~~~~

A `config.yaml` file, located in the same directory as the notebooks (e.g., `rna_analysis/notebooks/`), provides the means for configuration. It contains a **section for each notebook** and defines global parameters, primarily related to input and output paths, that may be edited by the user.

.. code-block::

  "pseudotime_analysis":
      adata_input_dir: "../adatas/"
      adata_output_dir: "../adatas/"
      figure_dir: "../figures/pseudotime/"
      log_file: "../logs/pseudotime_analysis_log.txt"
      overwrite_log: True
      report_dir: "../report/pseudotime_analysis/"

The example above shows the *pseudotime notebook* section, which defines the parameters for the notebook of the same name. These parameters are forwarded to the :class:`sctoolbox.settings <sctoolbox.SctoolboxConfig>` module. The `config.yaml` is loaded at the beginning of each notebook as follows:

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

- The first cell of the notebook has to be a **hidden init cell**. (See :ref:`input-cells` for more info)
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

.. _input-cells:

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

This is a **hidden initialization cell**, meaning it will run when the notebook is opened. Below is an example for a first cell that may be copied.

.. code-block:: python

    from sctoolbox.utils.jupyter import bgcolor, _compare_version

    # change the background of input cells
    bgcolor("PowderBlue", select=[2, 4, 7])

    nb_name = "pseudotime_analysis.ipynb"

    _compare_version(nb_name)

The contents of this cell typical include:

- the `bgcolor` function defining which cells to highlight
- the name of the file to allow checking if the version of the analysis notebooks matches to the sctoolbox package version

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

All notebooks should contain texts describing the current steps to explain the analysis process and help the user in interpretation and decision making.

- The first text in a notebook should describe the general aim of the notebook.
- Parameters should be described using tables. The tables should contain at least parameter name, description, options (e.g. "values between 0-1") and default value.
- Add a rule of thumb when possible. Sentences like "Higher values are better but result in longer runtime." are really valuable in decision making.
- Keep it concise, explain relevant concepts not each step in an algorithm. Ask yourself: "Is this information relevant to get an optimal analysis result?"
- Add links to information that is nice to know but not strictly needed to progress with the analysis.
- Add text to help interpret plots.
- If necessary, add the requirements to run this notebook, e.g., "Requires clustered data.".

.. note::
  Add sources to packages, best practise, relevant papers or anything else that might help the user to further inform themselves.

.. _output_section:

Outputs
~~~~~~~

Output files, such as plots and tables should be saved with a number prefix indicating the order in which they were created. Plots should be saved in PDF format to allow easy editing.

Use the sctoolbox internal saving functions for example:

- :func:`sctoolbox.plotting.general._save_figure`
- :func:`sctoolbox.utils.tables.write_excel`

Testing
~~~~~~~

The SC-Framework has a CI/CD pipeline which runs all analysis notebooks to ensure robust analysis and interfaces. However, this requires the notebooks to be setup to run with the test data located next to the notebooks. The notebooks run in the correct analysis order (specified through the prefix) so you can use files that will be created during the analysis preceding the respective notebook.

.. note::
  General notebooks, i.e. notebooks located in `general_notebooks`, must be copied to the respective `*_analysis` directory. Please add them to the respective CI/CD job (``notebooks-RNA`` or ``notebooks-ATAC``) in the `.gitlab-ci.yml`.

Naming
~~~~~~

The prefix in the filename, e.g. ``01_assembling_anndata.ipynb``, indicates the order in which notebooks should run. For example, the typical run order of an RNA analysis is:

1. ``01_assembling_anndata.ipynb``
2. ``02_QC_filtering.ipynb``
3. ``03_normalization_batch_correction.ipynb``
4. ``04_clustering.ipynb``

Afterwards, notebooks without prefix (`general_notebooks`) or with a letter-based prefix (e.g. ``0B_velocity_analysis.ipynb``) may be run in any order. Letter-based prefixes that end with a number should run in the order given by the number at the end. For example, first ``0A1_ligand_receptor.ipynb``, then ``0A2_ligand_receptor_hub_genes.ipynb``. The ``99-report.ipynb`` notebook should run last as indicated by its prefix.

Package
-------

`sctoolbox` is the python package that provides a lot of the functionality used within the notebooks. Its code is located in the ``src`` directory. The `sctoolbox` is structured into three main parts each containing a number of submodules.

Structure
~~~~~~~~~

Adding a new submodule also requires to add the new name to the ``__init__.py`` on the same level.

plotting
^^^^^^^^

Contains visualization functions. New plotting functions should normally be added here.

tools
^^^^^

Functions related to a specific tool or topic. Anything regarding integration of a new tool may be added here.

utils
^^^^^

General utility functions. Usually functions that may be used at several places throughout the package.

Extending the package
~~~~~~~~~~~~~~~~~~~~~

New functions should be added next to functions of similar content to keep the package clean an concise. Check above for the general package structure. Each of the general modules, shown as directories in the repository, contain python scripts aka submodules. These submodules can be interpreted as subcategories. A new function should be added to a submodule with a similar topic.

A new file may be created if the function does not fit to any of the existing submodules. A new file must be registered in the ``__init__.py`` that can be found in the same directory as the newly created file. To register add the name of the new file (without the extension) to the ``__all__`` variable within the ``__init__.py``.

.. code-block:: python

  # define what is exported in this module
  __all__ = [
      ...
      "clustering",
      "embedding",
      "<new_file>"
  ]


Functions should utilize the :class:`sctoolbox.settings <sctoolbox.SctoolboxConfig>`, e.g. to manage the default number of threads or filepaths. A parameter may be implemented to overwrite this behavior. Also see the :ref:`output_section` section.

See below for considerations regarding the actual code like style or robustness.

Testing
~~~~~~~

The `sctoolbox` contains unit-tests, utilizing `Pytest <https://docs.pytest.org/en/stable/>`_, to ensure robust code. The code coverage is the percentage of how many lines of code are tested. A coverage `> 90%` is considered great. Therefore, we aim to test as much as possible. Especially, new functions should be tested from the beginning to avoid later problems.

Tests are located within the ``tests`` directory. The structure within the directory follows the structure of the package (``src/sctoolbox``) for convenience.

Decorators
~~~~~~~~~~

Python `decorators <https://realpython.com/primer-on-python-decorators/>`_ are a way to do something before and/or after calling a function. You should be aware of two decorators used with many of the `sctoolbox` functions:

Beartype
^^^^^^^^

`Beartype <https://beartype.readthedocs.io/en/latest/>`_ is a package that checks parameter types and ensure that they match the type hint. For example it would raise an error if a parameter expects an integer but receives a string. Adding the beartype decorator enables these parameter checks:

.. code-block:: python

  from beartype import beartype

  @beartype
  def my_new_function():
      pass

log_anndata
^^^^^^^^^^^

This decorator is implemented in the `sctoolbox`. It logs calls to the functions it decorates and adds them to an AnnData object, which must be provided through one of the parameters of the decorated function. The log can be viewed with :func:`sctoolbox.utils.decorator.get_parameter_table`.

Add this decorator to a function that receives an AnnData object. Not every function must be logged, only do this for **top-level/important** functions.

.. code-block:: python

  import sctoolbox.utils.decorator as deco

  @deco.log_anndata
  @beartype
  def estimate_doublets_amulet(...):
      ...

.. note::
  The order of decorators is important! ``@beartype`` should always be directly above the function.

Doc-string
~~~~~~~~~~

Documentation is important for usability, that is why the SC-Framework requires doc-strings for every function. Doc-strings should follow the `numpy-style <https://numpydoc.readthedocs.io/en/latest/format.html>`_ and are enforced by the `Ruff <https://docs.astral.sh/ruff/>`_ linter.

Example code and results
^^^^^^^^^^^^^^^^^^^^^^^^

Functions may carry an ``Examples`` section that shows how they are called and renders the expected output in the :doc:`API-reference </API/index>`. Two Sphinx directives are available:

- ``.. plot::`` runs the example and embeds the resulting figure. Use it for **plotting functions**.
- ``.. exec_code::`` runs the example and embeds its textual output. Use it for **non-plotting code**.

Examples are best practice but never mandatory:

- **Plotting functions** are **highly encouraged** to add a ``.. plot::`` example. Showing the produced figure is what makes the API reference useful for these functions.
- **All other public functions** are **encouraged** to add an ``.. exec_code::`` example where running the function produces a result worth showing (a table, a printed summary). Skip it for functions whose output is not illustrative.

To do so, add an example section to the doc-string of the respective function:

.. code-block:: python

  def some_function():
      """
      Some function

      Parameters
      ----------
      [some parameters]

      Returns
      -------
      [return value]

      Examples
      --------
      .. plot:: <- for plotting functions
          :context: close-figs

          plotting_function(input)

      .. exec_code:: <- for other code
          import some_code
          some_code()
      """

Always pass ``:context: close-figs`` to a ``.. plot::`` directive. The ``:context:`` keeps the shared namespace (see below) alive across examples, and ``close-figs`` closes the previous figure so each example renders only its own plot.

Where does the data come from?
""""""""""""""""""""""""""""""

Each module page runs a short setup script **once at the top**, before any of that page's examples. These scripts live in the repository and provide the imports and input data the examples rely on (so the setup itself is not repeated in every example):

- ``docs/source/plot_pre_code.py`` — runs before the examples on the ``plotting`` page.
- ``docs/source/utils_pre_code.py`` — runs before the examples on the ``utils`` page.

If your new example needs input data that the relevant pre-code script does not yet prepare (e.g. a new fixture, a different ``.obs`` column), **extend that script** so the variable exists for everyone. Keep additions minimal and reuse the existing ``adata`` where possible.

What not to do!
"""""""""""""""

Please do not overwrite any variables in your example code! All examples on a module page share a single namespace and run after each other in order — the pre-code script first, then every function's example.

For example, the main example input is stored in the variable ``adata``. If you would overwrite it with something else all the following code that uses the input variable is likely to fail.

Checking that an example renders
""""""""""""""""""""""""""""""""

The example directives only execute when the documentation is built; ``ruff`` and ``pytest`` do not run them. The authoritative check is the full Sphinx build (``make -C docs html``), which CI runs on the ``dev`` pipeline — this is what executes ``.. plot::`` examples and renders their figures.

For a quick local check while iterating, build a single page with the ``dummy`` builder. It reads and validates the page without producing HTML, which avoids the notebook-finalisation step that makes a single-page ``html`` build fail:

.. code-block:: bash

  # validate the utils page and run its .. exec_code:: examples
  sphinx-build -b dummy docs/source /tmp/scdocs docs/source/API/utils.rst

What this does and does not catch:

- ``.. exec_code::`` examples **are executed** — the build exits non-zero if one raises. This is a fast, offline smoke-test for examples on non-plotting pages.
- ``.. plot::`` examples are **only parsed, not executed**, by the ``dummy`` builder (and the plotting pre-code even downloads data over the network). Their execution is verified by the full ``make -C docs html`` build on CI, not by this local check.

The build also prints ``toctree`` / cross-reference warnings about the rest of the documentation that was not built — those are expected for a single-page build and do not fail it.

Deprecation
~~~~~~~~~~~

Continued development creates changes in design and structure of the code. As a result, functions, classes, etc. may become outdated and should be removed. We use the `deprecation package <https://pypi.org/project/deprecation/>`_ to give users a grace period before the respective code is deleted.

- A deprecated function should be marked for **removal in two minor** versions.
- Add ``fail_if_not_removed`` to the functions tests.

.. code-block:: python

  import deprecation
  from sctoolbox import __version__

  @deprecation.deprecated(deprecated_in="0.15.0", removed_in="0.17.0",
                          current_version=__version__,
                          details="Use function xy instead")
  def deprecated_function():
      pass

.. code-block:: python

  import deprecation

  @deprecation.fail_if_not_removed
  def test_deprecated_function():
    pass

Changelog
~~~~~~~~~

The repository contains a `CHANGES.md` file that lists the changes that are made with each version of the SC-Framework. An entry must be added whenever changes are made to either the ``sctoolbox`` package or the notebooks. This is enforced by the CI/CD pipeline. Once released, the documentation will be updated and the changes are displayed in :doc:`CHANGES`.

The changelog is in `markdown-format <https://en.wikipedia.org/wiki/Markdown>`_. A new version can be added by creating a new section at the top of the file:

.. code-block::

  # Changelog

  ## 0.xy.z (in progress)
  - implemented a new feature (#<issue number>)
  ...

  ### Changes to notebooks
  - added a new notebook (#<issue number>)
  ...

- ``(in progress)`` will be changed to the release date e.g. ``## 0.14.2 (24-11-2025)``.
- If possible, **add an issue number** to the end of the entry.
- The ``Changes to notebooks`` section can be skipped if there are no changes to the notebooks.
- Keep it short and link information e.g. an issue for details.
