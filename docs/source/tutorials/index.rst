Tutorials
---------

The SC-Framework features a collection of analysis Jupyter notebooks, which can be found in our `GitHub repository<https://github.com/loosolab/SC-Framework>`_. Method specific notebooks, e.g. RNA related, are found within the “_analysis” folder, while notebooks usable with any data are found in the “general_notebooks” folder. An analysis starts with copying the “analysis” to the desired location and opening the first notebook (assembly). Depending on the circumstances (e.g. whether the data is pre-analyzed) the other notebooks are either run in the order indicated by the number prefix in their name or skipped. Notebooks without a prefix (general) or with a prefix that contains a character (e.g. “0A1…”) are considered downstream and to be run after the numbered notebooks. While downstream notebooks don’t follow an overarching order of execution, notebooks with the same character in their prefix must be run in order, e.g. “0A1\_…” -> “0A2\_…”. General notebooks should be copied into the respective analysis folder before their execution.

Follow the links below to see exemplary multi-notebook analysis workflows. 

**scRNA-seq Tutorial**

.. toctree::
   :maxdepth: 2

   scRNA-seq_Analysis/index

**scATAC-seq Tutorial**

.. toctree::
   :maxdepth: 2

   scATAC-seq_Analysis/index
