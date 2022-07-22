from setuptools import setup

setup(name='sc-toolbox',
		description='Custom modules for single cell analysis',
		license='MIT',
		packages=['sctoolbox'],
		python_requires='>=3',
		install_requires=[
			'matplotlib',
			'numpy',
			'scanpy',
			'rpy2',
			'anndata2ri',
			'kneed',
			'fitter',
			'qnorm',
			'pysam'
		],
		)
