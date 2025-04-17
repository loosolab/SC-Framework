# Import the plotting module
import sctoolbox.plotting as pl
from sctoolbox import tools

# Load example dataset
import numpy as np
np.random.seed(42)
import scanpy as sc

adata = sc.datasets.pbmc68k_reduced()
adata.obs["condition"] = np.random.choice(["C1", "C2", "C3"], size=adata.shape[0])
tools.marker_genes.run_rank_genes(adata, "louvain")
tools.gsea.gene_set_enrichment(adata,
                               marker_key="rank_genes_louvain_filtered",
                               organism="human",
                               method="prerank",
                               inplace=True)
