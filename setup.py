from setuptools import setup

# Module requirements
extras_require = {"converter": ['rpy2', 'anndata2ri'],
                  "atac": ['pysam', 'episcanpy', 'pyyaml', 'psutil', 'uropa', 'ipywidgets'],
                  "interactive": ['click'],
                  "batch_correction": ['bbknn', 'mnnpy', 'harmonypy', 'scanorama'],
                  "receptor_lignad": ['scikit-learn', 'igraph'],

                  # Diffexpr is currently restricted to a specific commit to avoid dependency issues with the latest version
                  "deseq2": ["rpy2", "diffexp @ git+https://github.com/wckdouglas/diffexpr.git@0bc0ba5e42712bfc2be17971aa838bcd7b27a785#egg=diffexp"]  # rpy2 must be installed before diffexpr
                  }

extras_require["all"] = list(set(sum(extras_require.values(), [])))  # flatten list of all requirements

setup(
    name='sc-toolbox',
    description='Custom modules for single cell analysis',
    license='MIT',
    packages=['sctoolbox'],
    python_requires='>=3',
    install_requires=[
        'matplotlib',
        'scanpy>=1.9',  # 'colorbar_loc' not available before 1.9
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
        'gitlab'
    ],
    include_package_data=True,
    extras_require=extras_require
)
