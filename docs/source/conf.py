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

sys.path.insert(0, os.path.abspath('../..'))

# -- Project information -----------------------------------------------------

project = 'SC FRAMEWORK'
copyright = '2022, Loosolab'
author = 'Loosolab'


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = ['matplotlib.sphinxext.plot_directive',
              'sphinx.ext.autodoc',
              'sphinx.ext.autosummary',
              'sphinx.ext.napoleon',
              'sphinx.ext.viewcode',
              'sphinx.ext.intersphinx',
              "nbsphinx",
              "nbsphinx_link",
              ]

napoleon_numpy_docstring = True
autodoc_member_order = 'bysource'

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

# Mock modules imported within sctoolbox - prevents failures of documentation build
# modules are needed to build the example figures in the documentation, so they should not be mocked
autodoc_mock_imports = []  # 'uropa', 'anndata', 'numpy', 'matplotlib',
                           # 'glob', 'sklearn', 'seaborn',
                           # 'qnorm', 'pylab', 'episcanpy']

# --- Create nblink files for notebooks ----------------------------------------

# Remove all previous .nblink files
links = glob.glob("notebooks/*.nblink")
for link in links:
    os.remove(link)

# Create nblinks for current notebooks
notebooks = glob.glob("../../notebooks/*.ipynb")
for f in notebooks:
    f_name = os.path.basename(f).replace(".ipynb", "")

    d = {"path": "../" + f} 
    with open("notebooks/" + f_name + ".nblink", 'w') as fp:
        json.dump(d, fp)

nbsphinx_execute = 'never'

# -- Options for automatic plots in docs -------------------------------------

plot_include_source = True
plot_html_show_source_link = False
plot_formats = [("png", 90)]
plot_html_show_formats = False

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
