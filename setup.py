from setuptools import setup

# Module requirements
converter = ['rpy2==3.4.5', 'anndata2ri']
atac = ['pysam', 'episcanpy', 'pyyaml', 'ipywidgets']
batch_correction = ['bbknn', 'mnnpy', 'harmonypy', 'scanorama']
all = converter + atac + batch_correction


setup(name='sc-toolbox',
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
			'tqdm'
		],
		include_package_data=True,
		extras_require={
			'all': all,
			'converter': converter,
			'atac': atac,
			'batch_correction': batch_correction
		}
)
