"""Sctoolbox a collection of single cell analysis functions."""

from setuptools import setup
from setuptools import find_namespace_packages
import re
import os
import glob

import sys
import subprocess

# Install these dependencies before everything else
# do this here (not in conda) to make this package buildable without prior steps
pre_deps = [
    "cmake>=3.18",  # fixes ERROR: Failed to build installable wheels for some pyproject.toml based projects (louvain)
    "setuptools_scm>=9.0.3"  # due to https://github.com/pypa/setuptools-scm/issues/938
]

# get a clean bash path (not altered by the pip build process)
# pip installs each package using separate build environment. So the pre-dependencies would be only available during sctoolbox installation.
# This is circumvented by installing the pre-dependencies globally. Which is similar to doing "pip install <pre-deps>" before "pip install sctoolbox".
# TODO only works on Linux based systems
clean_path = subprocess.check_output(["echo $PATH"], shell=True).decode("utf-8").strip()

for dep in pre_deps:
    subprocess.run([sys.executable, "-m", "pip", "install", dep], env={'PATH': clean_path}, check=True)

# Module requirements
extras_require = {"converter": ['rpy2', 'anndata2ri'],
                  "atac": ['uropa',
                           'pybedtools>=0.9.1',  # https://github.com/daler/pybedtools/issues/384
                           'pygenometracks>=3.8',
                           'peakqc',
                           'tobias>=0.17.2'],  # to avoid installation error
                  "interactive": ['click'],
                  "batch_correction": ['bbknn', 'harmonypy', 'scanorama'],
                  # pycirclize https://github.com/moshi4/pyCirclize/issues/75
                  "receptor_ligand": ['scikit-learn', 'igraph', 'pycirclize>=1.7.1', 'liana', 'mudata>=0.3.1', 'networkx>=3.5'],  # anndata>=10.9 requires mudata>=0.3.1; networkx>=3.5 for numpy 2 support
                  "velocity": ['scvelo @ git+https://github.com/rwiegan/scvelo.git'],  # install from fork until this is merged: https://github.com/theislab/scvelo/pull/1308
                  "pseudotime": ["scFates"],  # omit scFates due to version conflict https://github.com/LouisFaure/scFates/issues/50
                  "gsea": ["gseapy"],
                  "deseq2": ["pydeseq2>=0.5.2"],  # https://github.com/owkin/PyDESeq2/issues/242
                  "scar": ["scar @ git+https://github.com/Novartis/scar.git"]
                  }

# contains all requirements aka a full installation
extras_require["all"] = list(dict.fromkeys([item for sublist in extras_require.values() for item in sublist if item != "scFates"]))  # flatten list of all requirements; omit scFates, see line 39
# contains the requirements needed to run notebooks 1-4 (atac & rna); skipping rarely used dependencies (e.g. scar)
extras_require["core"] = sum([value for key, value in extras_require.items() if key in ["converter", "atac", "interactive", "batch_correction"]], start=[])
# contains dependencies needed for the downstream notebooks (general & notebooks after 4)
extras_require["downstream"] = sum([value for key, value in extras_require.items() if key in ["receptor_ligand", "velocity", "gsea", "deseq2"]], start=[])  # , "pseudotime" ; see line 39 for more info


def find_version(f: str) -> str:
    """
    Get package version from version file.

    Parameters
    ----------
    f : str
        Path to version file.

    Returns
    -------
    str
        Version string.

    Raises
    ------
    RuntimeError
        If version string is missing.
    """
    version_file = open(f).read()
    version_match = re.search(r"^__version__ = ['\"]([^'\"]*)['\"]", version_file, re.M)
    if version_match:
        return version_match.group(1)
    else:
        raise RuntimeError("Unable to find version string.")


# Find all packages in sctoolbox
packages = find_namespace_packages("sctoolbox")
packages = ["sctoolbox." + package for package in packages]  # add sctoolbox. prefix to all packages

# Find top level modules in sctoolbox
modules = glob.glob("sctoolbox/*.py")
modules = [m.replace("/", ".")[:-3] for m in modules if not m.endswith("__init__.py")]  # omit file ending and adjust path to import format e.g. sctoolbox._modules


# Readme from git
def readme():
    """Collect the readme file content."""
    with open('README.md') as f:
        return f.read()


setup(
    # This is the name of the package important in PyPI, important during installation "pip install SC-Framework".
    # The package is imported using "import sctoolbox" defined by the folder name.
    name='SC-Framework',
    description='Custom modules for single cell analysis',
    long_description=readme(),
    long_description_content_type='text/markdown',
    version=find_version(os.path.join("sctoolbox", "_version.py")),
    license='MIT',
    packages=packages,
    py_modules=modules,
    python_requires='>=3.9,<3.13',  # dict type hints as we use it require python 3.9; 3.13 not supported by cmake # https://github.com/python-cmake-buildsystem/python-cmake-buildsystem/issues/350
    install_requires=[
        'pysam',
        'matplotlib',
        'matplotlib_venn',
        'scanpy[louvain,leiden]>=1.11',  # 'colorbar_loc' not available before 1.9; fix run_rank_genes error 1.11; also install community detection (louvain & leiden)
        'anndata>=0.8',  # anndata 0.7 is not upward compatible
        'numba>=0.57.0rc1',  # minimum version supporting python>=3.10, but 0.57 fails with "cannot import name 'quicksort' from 'numba.misc'" for scrublet
        'numpy>=2',
        'kneed',
        'qnorm',
        'plotly',
        'scipy>=1.14',
        'statsmodels @ git+https://github.com/statsmodels/statsmodels',  # remove once statsmodels 0.15 is released
        'tqdm',
        'pandas>1.5.3',  # https://gitlab.gwdg.de/loosolab/software/sc_framework/-/issues/200
        'seaborn>0.12',
        'ipympl',
        'ipywidgets>=8.0.0',  # needed to show labels in interactive accordion widgets
        'scrublet',
        'IPython',
        'openpyxl',
        'apybiomart',
        'requests',
        'python-gitlab',
        'psutil',
        'deprecation',
        'pyyaml',
        'beartype>=0.18.2',  # Version 0.18.0 is not working properly
        'packaging',
        'throttler',
        'upsetplot',
        'pptreport>=1.1.4',  # to fix placeholder not found error
        'boto3',
        'Jinja2'
    ],
    include_package_data=True,
    extras_require=extras_require
)
