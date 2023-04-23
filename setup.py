from setuptools import setup

# Module requirements
converter = ['rpy2==3.4.5', 'anndata2ri']
atac = ['pysam', 'episcanpy', 'pyyaml', 'psutil', 'uropa', 'ipywidgets']
interactive = ['click']
batch_correction = ['bbknn', 'mnnpy', 'harmonypy', 'scanorama']
all = converter + atac + interactive + batch_correction

setup(
    name='sc-toolbox',
    description='Custom modules for single cell analysis',
    license='MIT',
    packages=['sctoolbox'],
    python_requires='>=3',
    install_requires=[
        'matplotlib',
        'scanpy>=1.9',  # 'colorbar_loc' not available before 1.9
        'numba>=0.57.0rc1',  # minimum version supporting python>=3.10
        'numpy',
        'scanpy',
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
        'IPython'
    ],
    include_package_data=True,
    extras_require={
        'all': all,
        'converter': converter,
        'atac': atac,
        'interactive': interactive,
        'batch_correction': batch_correction
    }
)
