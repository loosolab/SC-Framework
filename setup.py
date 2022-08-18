from setuptools import setup

# Module requirements
converter = ['rpy2==3.4.5', 'anndata2ri']
atac = ['pysam', 'episcanpy', 'pyyaml', 'ipywidgets']
interactive = ['click', 'IPython']
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
        'numpy',
        'scanpy',
        'kneed',
        'fitter',
        'qnorm',
        'scipy',
        'statsmodels',
        'tqdm',
        'pandas',
        'seaborn'
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
