# Import the plotting module
import sctoolbox.plotting as pl
import sctoolbox.tools.receptor_ligand as rl

# Load example dataset
import numpy as np
import scanpy as sc
np.random.seed(42)


adata = sc.datasets.pbmc68k_reduced()
adata.obs["condition"] = np.random.choice(["C1", "C2", "C3"], size=adata.shape[0])
adata.obs["timepoint"] = np.random.choice(["Day0", "Day3", "Day7"], size=adata.shape[0])

# Setup receptor-ligand database
rl.download_db(
    adata=adata,
    db_path="celltalkdb",
    ligand_column="ligand",
    receptor_column="receptor",
    inplace=True
)

# Calculate the interaction table
rl.calculate_interaction_table(
    adata=adata,
    cluster_column='louvain',
    inplace=True
)

# Get top interactions from the dataset
interactions = rl.get_interactions(adata, min_perc=5)
top_receptors = interactions['receptor_gene'].value_counts().head(5).index.tolist()
top_ligands = interactions['ligand_gene'].value_counts().head(5).index.tolist()

# Select interesting interactions to track
interaction_pairs = []
for i, (_, row) in enumerate(interactions.sort_values('interaction_score', ascending=False).iterrows()):
    # Get top 3 interactions
    if i >= 3:
        break
    interaction_pairs.append((
        row['receptor_gene'],
        row['receptor_cluster'],
        row['ligand_gene'],
        row['ligand_cluster']
    ))

# Calculate differences between conditions
adata_diff = rl.calculate_condition_differences(
    adata=adata,
    condition_columns=['condition'],
    cluster_column='louvain',
    min_perc=None,
    condition_filters={'condition': ['C1', 'C2']},
    inplace=False
)

# Calculate time-dependent differences
time_diff = rl.calculate_condition_differences(
    adata=adata,
    condition_columns=['timepoint', 'condition'],
    cluster_column='louvain',
    min_perc=5,
    condition_filters={
        'condition': ['C1'],
        'timepoint': ['Day0', 'Day3', 'Day7']
    },
    time_column='timepoint',
    time_order=['Day0', 'Day3', 'Day7'],
    inplace=False
)
