# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#

import os
import sys
import glob
import json
import shutil

sys.path.insert(0, os.path.abspath('.'))
import build_api


sys.path.insert(0, os.path.abspath('../..'))

# -- Project information -----------------------------------------------------

project = 'SC FRAMEWORK'
copyright = '2024, Loosolab'
author = 'Loosolab'

# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = ['matplotlib.sphinxext.plot_directive',  # for plot examples in docs
              'sphinx_exec_code',   # for code examples in docs
              'sphinx.ext.autodoc',
              'sphinx.ext.autosummary',
              'sphinx.ext.napoleon',
              'sphinx.ext.viewcode',
              'sphinx.ext.intersphinx',
              "nbsphinx",
              "nbsphinx_link"
              ]

napoleon_numpy_docstring = True
autodoc_member_order = 'bysource'

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

# Mock modules imported within sctoolbox - prevents failures of documentation build
autodoc_mock_imports = []

# ---- Automatic documentation generation -------------------------------------

build_api.main()  # generate API documentation using custom script

# --- Create nblink files for notebooks ----------------------------------------

# Remove all previous .nblink files
links = glob.glob("*notebooks/*.nblink")
for link in links:
    os.remove(link)

# Create nblinks for current notebooks
notebooks = glob.glob("../../*_analysis/notebooks/*.ipynb") # captures both rna-notebooks, atac-notebooks etc.
notebooks.extend(glob.glob("../../*_notebooks/*.ipynb")) # capture general_notebooks
for f in notebooks:

    if "rna_analysis" in f:
        notebook_folder = "rna-notebooks/"
    elif "atac_analysis" in f:
        notebook_folder = "atac-notebooks/"
    elif "general_notebooks" in f:
        notebook_folder = "general-notebooks/"
    else:
        raise ValueError("Did not recoginze notebook type.")

    os.makedirs(notebook_folder, exist_ok=True)  # create folder if it doesn't exist

    f_name = os.path.basename(f).replace(".ipynb", "")

    d = {"path": "../" + f}
    with open(notebook_folder + f_name + ".nblink", 'w') as fp:
        json.dump(d, fp)

nbsphinx_execute = 'never'

# -- Options for automatic plots in docs -------------------------------------

# Copy data from test data folder to docs folder
# os.makedirs("source/API", exist_ok=True)

# copy folder
shutil.copytree("../../tests/data", "API/data", dirs_exist_ok=True)  # dirs_exist_true only important when testing locally
# copy folder
shutil.copytree("../../image", "image", dirs_exist_ok=True)

plot_include_source = True
plot_html_show_source_link = False
plot_formats = [("png", 90)]
plot_html_show_formats = False
plot_pre_code = open("plot_pre_code.py").read()
utils_pre_code = open("utils_pre_code.py").read()

plot_rcparams = {'savefig.bbox': 'tight'}  # make sure plots are not cut off in the docs
plot_apply_rcparams = True                 # if context option is used

# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
# html_theme = 'alabaster'
html_theme = 'sphinx_rtd_theme'

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']
html_css_files = [
    'css/custom.css',
]

html_js_files = [
    'js/custom.js',
]
