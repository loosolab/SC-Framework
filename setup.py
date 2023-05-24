from setuptools import setup
from setuptools import find_namespace_packages
import re
import os
import glob

# Module requirements
extras_require = {"converter": ['rpy2', 'anndata2ri'],
                  "atac": ['episcanpy', 'pyyaml', 'uropa', 'ipywidgets', 'sinto', 'pybedtools'],
                  "interactive": ['click'],
                  "batch_correction": ['bbknn', 'harmonypy', 'scanorama'],
                  "receptor_lignad": ['scikit-learn', 'igraph'],

                  # Diffexpr is currently restricted to a specific commit to avoid dependency issues with the latest version
                  "deseq2": ["rpy2", "diffexp @ git+https://github.com/wckdouglas/diffexpr.git@0bc0ba5e42712bfc2be17971aa838bcd7b27a785#egg=diffexp"]  # rpy2 must be installed before diffexpr
                  }

extras_require["all"] = list(dict.fromkeys([item for sublist in extras_require.values() for item in sublist]))  # flatten list of all requirements


# Find version for package
def find_version(f):
    version_file = open(f).read()
    version_match = re.search(r"^__version__ = ['\"]([^'\"]*)['\"]", version_file, re.M)
    if version_match:
        return version_match.group(1)
    else:
        raise RuntimeError("Unable to find version string.")


# Find all packages in sctoolbox
packages = find_namespace_packages("sctoolbox")
packages = [package for package in packages if not package.startswith("data")]  # remove data package as it is included in manifest
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
    python_requires='>=3,<3.11',  # pybedtools is not compatible with python 3.11
    install_requires=[
        'pysam',
        'matplotlib',
        'matplotlib_venn',
        'scanpy>=1.9',  # 'colorbar_loc' not available before 1.9
        'numba>=0.57.0rc1',  # minimum version supporting python>=3.10
        'numpy',
        'kneed',
        'qnorm',
        'plotly',
        'scipy',
        'statsmodels',
        'tqdm',
        'pandas',
        'seaborn',
        'ipympl',
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
    ],
    include_package_data=True,
    extras_require=extras_require
)
