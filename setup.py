from setuptools import setup
from setuptools import find_namespace_packages

# Module requirements
extras_require = {"converter": ['rpy2', 'anndata2ri'],
                  "atac": ['pysam', 'episcanpy', 'pyyaml', 'psutil', 'uropa', 'ipywidgets', 'sinto', 'pybedtools'],
                  "interactive": ['click'],
                  "batch_correction": ['bbknn', 'harmonypy', 'scanorama'],
                  "receptor_lignad": ['scikit-learn', 'igraph'],

                  # Diffexpr is currently restricted to a specific commit to avoid dependency issues with the latest version
                  "deseq2": ["rpy2", "diffexp @ git+https://github.com/wckdouglas/diffexpr.git@0bc0ba5e42712bfc2be17971aa838bcd7b27a785#egg=diffexp"]  # rpy2 must be installed before diffexpr
                  }

extras_require["all"] = list(dict.fromkeys([item for sublist in extras_require.values() for item in sublist]))  # flatten list of all requirements

setup(
    name='sc-toolbox',
    description='Custom modules for single cell analysis',
    license='MIT',
    packages=find_namespace_packages(),
    python_requires='>=3,<3.11',  # pybedtools is not compatible with python 3.11
    install_requires=[
        'matplotlib',
        'scanpy>=1.9',  # 'colorbar_loc' not available before 1.9
        'numba>=0.57.0rc1',  # minimum version supporting python>=3.10
        'numpy',
        'kneed',
        'fitter',
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
        'ratelimiter',
        'python-gitlab'
    ],
    include_package_data=True,
    extras_require=extras_require
)
