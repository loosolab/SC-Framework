# Import the plotting module
import sctoolbox.plotting as pl
import sctoolbox.utilities as utils

# Load example dataset
import numpy as np
np.random.seed(42)
import scanpy as sc

adata = sc.datasets.pbmc68k_reduced()
adata.obs["condition"] = np.random.choice(["C1", "C2", "C3"], size=adata.shape[0])
