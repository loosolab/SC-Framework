"""Sctoolbox a collection of single cell analysis functions."""

from setuptools import setup
from setuptools import find_namespace_packages
import re
import os
import glob

# Module requirements
extras_require = {"converter": ['rpy2', 'anndata2ri'],
                  "atac": ['pyyaml', 'episcanpy', 'uropa', 'pybedtools', 'pygenometracks>=3.8', 'peakqc'],
                  "interactive": ['click'],
                  "batch_correction": ['bbknn', 'harmonypy', 'scanorama'],
                  "receptor_ligand": ['scikit-learn', 'igraph', 'pycirclize', 'liana', 'mudata>=0.3.1'],  # anndata>=10.9 requires mudata>=0.3.1
                  "velocity": ['scvelo @ git+https://github.com/theislab/scvelo.git'],
                  "pseudotime": ["scFates"],
                  "gsea": ["gseapy==1.1.2"],  # Version 1.1.3 currently does not work properly with our pinned matplotlib version. Could also be a bug by gseapy.
                  "deseq2": ["pydeseq2>=0.4.11"],
                  "scar": ["scar @ git+https://github.com/Novartis/scar.git"]
                  }

extras_require["all"] = list(dict.fromkeys([item for sublist in extras_require.values() for item in sublist]))  # flatten list of all requirements


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

setup(
    name='sctoolbox',
    description='Custom modules for single cell analysis',
    version=find_version(os.path.join("sctoolbox", "_version.py")),
    license='MIT',
    packages=packages,
    py_modules=modules,
    python_requires='>=3.9',  # dict type hints as we use it require python 3.9
    install_requires=[
        'cmake>=3.18',  # fixes ERROR: Failed to build installable wheels for some pyproject.toml based projects (gseapy, louvain)
        'pysam',
        'matplotlib<3.9.0',
        'matplotlib_venn',
        'scanpy>=1.10.2',  # 'colorbar_loc' not available before 1.9
        'anndata>=0.8',  # anndata 0.7 is not upward compatible
        'numba>=0.57.0rc1',  # minimum version supporting python>=3.10, but 0.57 fails with "cannot import name 'quicksort' from 'numba.misc'" for scrublet
        'numpy',
        'kneed',
        'qnorm',
        'plotly',
        'scipy>=1.14',
        'statsmodels',
        'tqdm',
        'pandas>1.5.3',  # https://gitlab.gwdg.de/loosolab/software/sc_framework/-/issues/200
        'seaborn>0.12',
        'ipympl',
        'ipywidgets>=8.0.0',  # needed to show labels in interactive accordion widgets
        'scrublet',
        'leidenalg',
        'louvain',
        'IPython',
        'openpyxl',
        'apybiomart',
        'requests',
        'python-gitlab',
        'psutil',
        'python-gitlab',
        'deprecation',
        'pyyaml',
        'beartype>=0.18.2',  # Version 0.18.0 is not working properly
        'pybedtools>=0.9.1',  # https://github.com/daler/pybedtools/issues/384
        'packaging',
        'throttler',
        'upsetplot'
    ],
    include_package_data=True,
    extras_require=extras_require
)
