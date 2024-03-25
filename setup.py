"""Sctoolbox a collection of single cell analysis functions."""

from setuptools import setup
from setuptools import find_namespace_packages
import re
import os
import glob

# Module requirements
extras_require = {"converter": ['rpy2', 'anndata2ri'],
                  "atac": ['pyyaml', 'episcanpy', 'uropa', 'pybedtools', 'pygenometracks'],
                  "interactive": ['click'],
                  "batch_correction": ['bbknn', 'harmonypy', 'scanorama'],
                  "receptor_ligand": ['scikit-learn<=1.2.2', 'igraph'],  # bbknn requires sk-learn <= 1.2
                  "velocity": ['scvelo'],
                  "pseudotime": ["fa2 @ git+https://github.com/AminAlam/forceatlas2.git"],  # fa2 is abandoned we should replace it soon! (see #212)
                  # Diffexpr is currently restricted to a specific commit to avoid dependency issues with the latest version
                  "deseq2": ["rpy2", "diffexp @ git+https://github.com/wckdouglas/diffexpr.git@0bc0ba5e42712bfc2be17971aa838bcd7b27a785#egg=diffexp"]  # rpy2 must be installed before diffexpr
                  }

extras_require["all"] = list(dict.fromkeys([item for sublist in extras_require.values() for item in sublist]))  # flatten list of all requirements


def find_version(f: str) -> str:
    """
    Get package version from file.

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
    python_requires='>=3.9,<3.11',  # dict type hints as we use it require python 3.9; pybedtools is not compatible with python 3.11
    install_requires=[
        'pysam',
        'matplotlib',
        'matplotlib_venn',
        'scanpy>=1.9',  # 'colorbar_loc' not available before 1.9
        'anndata>=0.8',  # anndata 0.7 is not upward compatible
        'numba>=0.57.0rc1',  # minimum version supporting python>=3.10, but 0.57 fails with "cannot import name 'quicksort' from 'numba.misc'" for scrublet
        'numpy',
        'kneed',
        'qnorm',
        'plotly',
        'scipy',
        'statsmodels',
        'tqdm',
        'pandas<=1.5.3',  # https://gitlab.gwdg.de/loosolab/software/sc_framework/-/issues/200
        'seaborn<0.12',  # statannotations 0.6.0 requires seaborn<0.12
        'ipympl',
        'ipywidgets<=7.7.5',  # later versions cause problems in some cases for interactive plots
        'scrublet',
        'leidenalg',
        'louvain',
        'IPython',
        'openpyxl',
        'apybiomart',
        'requests',
        'ratelimiter',
        'python-gitlab',
        'psutil',
        'pyyaml',
        'deprecation',
        'beartype',
        'pybedtools',
        'packaging'
    ],
    include_package_data=True,
    extras_require=extras_require
)
