Frequently asked questions
==========================

This section explains common questions, design phyilosophies and idioms.

FAQ
---
**Q: I have an old/ already started analysis. How do I find out what was done or who was responsible?**

**A:** The function logging contains information on which function was run and by whom. It can be accessed using :func:`sctoolbox.utils.decorator.get_parameter_table`.

**Q: My** ``.h5ad`` **file is already pre-analyzed. I want to skip some of the analysis notebooks. What do I do?**

**A:** You should always start with the assembly notebook (the first notebook), which ensures a proper output structure. Afterwards, go ahead with the notebook you want to run.

**Q: I have encountered a bug, I have a feature request, there is something I need help with or want to discuss.**

**A:** We are always happy to help. If you encounter something that needs attention open an issue with a detailed explanation and if possible a small code example. Thank you!

Idioms
------

1. Notebook Structure
~~~~~~~~~~~~~~~~~~~~~
All notebooks follow the same general template and rules:

- Notebooks typically begin with loading the anndata object and a cell for user inputs.
- Cells with a blue background require user input.
- Colorless cells are considered static and therefore shouldn't be changed. They are also locked and cannot be changed.
- The last step of a notebook is to save the analyzed anndata as a ``.h5ad`` file to be used by subsequent analysis steps (notebooks).

2. Module settings
~~~~~~~~~~~~~~~~~~
The SC framework provides a settings class (:class:`SctoolboxConfig`).

- :class:`SctoolboxConfig` is used to set non-analysis options like output paths, number of threads, file prefixes, logging level. 
- The settings can be changed using the above mentioned class or by loading a config file (:func:`sctoolbox.utils.settings_from_config`).

3. Logging
~~~~~~~~~~
The framework provides two types of logging:

1. **Traditional logging** written to a log file. This includes messages, warnings and errors that occur during the execution of functions. 
2. The second is **function logging**. This type of logging is added to the anndata object (``adata.uns["sctoolbox"]["log"]``). Whenever a function works on an anndata object (usually when receiving an anndata through a parameter), general information about the function call is stored inside the anndata object (name of the executed function, parameters, start time, who executed it, etc.).

The function log can be accessed using :func:`sctoolbox.utils.decorator.get_parameter_table`.
