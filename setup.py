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
			'kneed',
			'fitter',
			'qnorm'
		],
		extras_require={
			'converter': ['rpy2==3.4.5', 'anndata2ri'],
			'bam': ['pysam']
		}
)
