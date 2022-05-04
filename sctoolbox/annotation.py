import pandas as pd

def add_cellxgene_annotation(adata, csv):
	""" Add columns from cellxgene annotation to the adata .obs table.
	
	Parameters
	------------
	adata : anndata object
		The adata object to add annotations to.
	csv : str
		Path to the annotation file from cellxgene containing cell annotation.
		
	Returns
	--------
	None - the annotation is added to adata in place.
	"""
	
	anno_table = pd.read_csv(csv, sep=",", comment='#')
	anno_table.set_index("index", inplace=True)
	anno_name = anno_table.columns[-1]
	adata.obs.loc[anno_table.index, anno_name] = anno_table[anno_name].astype('category')
	