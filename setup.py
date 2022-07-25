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
			'receptor-ligand': ['sklearn', 'igraph', 'cairocffi'],  # cairocffi needed for cairo backend
			'converter': ['rpy2==3.4.5', 'anndata2ri'],
			'bam': ['pysam']
		}
)
