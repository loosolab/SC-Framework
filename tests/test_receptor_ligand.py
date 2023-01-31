import pytest
import os
import pandas as pd
import numpy as np
import scanpy as sc

import sctoolbox.receptor_ligand as rl


# ------------------------------ FIXTURES -------------------------------- #

@pytest.fixture
def adata():
    """ Load and returns an anndata object. """
    f = os.path.join(os.path.dirname(__file__), 'data', "adata.h5ad")

    return sc.read_h5ad(f)


@pytest.fixture
def db_file():
    """ Path to receptor-ligand database """
    return os.path.join(os.path.dirname(__file__), 'data', 'receptor-ligand', 'mouse_lr_pair.tsv')


@pytest.fixture
def adata_db(adata, db_file):
    """ Adds interaction db to adata """
    return rl.download_db(adata=adata,
                          db_path=db_file,
                          ligand_column='ligand_gene_symbol',
                          receptor_column='receptor_gene_symbol',
                          inplace=False,
                          overwrite=False)


@pytest.fixture
def adata_inter(adata_db):
    """ Adds interaction scores to adata """
    return rl.calculate_interaction_table(adata=adata_db,
                                          cluster_column="louvain",
                                          gene_index=None,
                                          normalize=1000,
                                          inplace=False,
                                          overwrite=False)


# ------------------------------ TESTS -------------------------------- #

# ----- test setup functions ----- #

def test_download_db(adata, db_file):
    """ Assert rl database is added into anndata"""
    obj = adata.copy()

    # adata does not have database
    assert "receptor-ligand" not in obj.uns

    # add database
    rl.download_db(adata=adata,
                   db_path=db_file,
                   ligand_column='ligand_gene_symbol',
                   receptor_column='receptor_gene_symbol',
                   inplace=True,
                   overwrite=False)

    # adata contains database
    assert "receptor-ligand" in obj.uns
    assert "database" in obj.uns["receptor-ligand"]


def test_interaction_table(adata_db):
    """ Assert interaction are computed/ added into anndata """
    obj = adata_db.copy()

    # adata has db but no scores
    assert "receptor-ligand" in obj.uns
    assert "database" in obj.uns["receptor-ligand"]
    assert "interactions" not in obj.uns["receptor-ligand"]

    # compute rl scores
    rl.calculate_interaction_table(adata=obj,
                                   cluster_column="louvain",
                                   gene_index=None,
                                   normalize=1000,
                                   inplace=True,
                                   overwrite=False)

    # adata contains scores
    assert "interactions" in obj.uns["receptor-ligand"]


# ----- test helpers ----- #

def test_get_interactions(adata_inter):
    """ Assert that interactions can be received """
    interactions_table = rl.get_interactions(adata_inter)

    # output is a pandas table
    assert isinstance(interactions_table, pd.DataFrame)


def test_check_interactions(adata, adata_db, adata_inter):
    """ Assert that interaction test is properly checked """
    # raise error without rl info
    with pytest.raises(ValueError):
        rl._check_interactions(adata)

    # raise error with incomplete rl info
    with pytest.raises(ValueError):
        rl._check_interactions(adata_db)

    # accept
    rl._check_interactions(adata_inter)


# ----- test plotting ----- #

def test_violin(adata_inter):
    """ Violin plot is functional """
    plot = rl.interaction_violin_plot(adata_inter,
                                      min_perc=0,
                                      output=None,
                                      figsize=(5, 30),
                                      dpi=100)

    assert isinstance(plot, np.ndarray)


def test_hairball(adata_inter):
    """ Hairball network plot is functional """
    plot = rl.hairball(adata_inter,
                       min_perc=0,
                       interaction_score=0,
                       interaction_perc=90,
                       output=None,
                       title=None,
                       color_min=0,
                       color_max=None,
                       restrict_to=[],
                       show_count=True)

    assert isinstance(plot, np.ndarray)


def test_connectionPlot(adata_inter):
    """ Test if connectionPlot is working """
    plot = rl.connectionPlot(adata=adata_inter,
                             restrict_to=None,
                             figsize=(5, 10),
                             dpi=100,
                             connection_alpha="interaction_score",
                             output=None,
                             title=None,
                             receptor_cluster_col="receptor_cluster",
                             receptor_col="receptor_gene",
                             receptor_hue="receptor_score",
                             receptor_size="receptor_percent",
                             ligand_cluster_col="ligand_cluster",
                             ligand_col="ligand_gene",
                             ligand_hue="ligand_score",
                             ligand_size="ligand_percent",
                             filter="receptor_score > 0 & ligand_score > 0 & interaction_score > 0",
                             lw_multiplier=2,
                             wspace=0.4,
                             line_colors="rainbow")

    assert isinstance(plot, np.ndarray)
