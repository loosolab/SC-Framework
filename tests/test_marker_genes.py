import pytest
import os
import scanpy as sc
import numpy as np
import sctoolbox.marker_genes
from importlib.resources import files


# ---------------------------- FIXTURES -------------------------------- #

@pytest.fixture
def adata():

    h5ad = os.path.join(os.path.dirname(__file__), 'data', 'adata.h5ad')
    adata = sc.read_h5ad(h5ad)

    adata.obs["condition"] = np.random.choice(["C1", "C2", "C3"], size=adata.shape[0])

    return adata


# ------------------------------ TESTS --------------------------------- #

def test_get_chromosome_genes():
    """ Test if get_chromosome_genes get the right genes from the gtf """

    gtf = os.path.join(os.path.dirname(__file__), 'data', 'genes.gtf')

    with pytest.raises(Exception):
        sctoolbox.marker_genes.get_chromosome_genes(gtf, "NA")

    genes_chr1 = sctoolbox.marker_genes.get_chromosome_genes(gtf, "chr1")
    genes_chr11 = sctoolbox.marker_genes.get_chromosome_genes(gtf, "chr11")

    assert genes_chr1 == ["DDX11L1", "WASH7P", "MIR6859-1"]
    assert genes_chr11 == ["DGAT2"]


@pytest.mark.parametrize("species, gene_column", [("mouse", None),
                                                  ("unicorn", "gene"),
                                                  (None, None)])
def test_label_genes(adata, species, gene_column):
    """ Test of genes are labeled in adata.var """

    if species is None:
        with pytest.raises(ValueError):
            sctoolbox.marker_genes.label_genes(adata, species=species)  # no species given, and it cannot be found in infoprocess

    else:
        sctoolbox.marker_genes.label_genes(adata, gene_column=gene_column, species=species)

        added_columns = ["is_ribo", "is_mito", "cellcycle", "is_gender"]
        missing = set(added_columns) - set(adata.var.columns)  # test that columns were added

        if species == "mouse":
            assert len(missing) == 0
        else:
            assert "is_mito" in adata.var.columns and "is_ribo" in adata.var.columns  # is_gender and cellcycle are not added


def test_get_rank_genes_tables(adata):
    """ test if rank gene tables are created and saved to excel file """

    sc.tl.rank_genes_groups(adata, groupby="condition")

    tables = sctoolbox.marker_genes.get_rank_genes_tables(adata, out_group_fractions=True, save_excel="rank_genes.xlsx")

    assert len(tables) == 3
    assert os.path.exists("rank_genes.xlsx")


def test_mask_rank_genes(adata):
    """ Test if genes are masked in adata.uns['rank_genes_groups'] """

    sc.tl.rank_genes_groups(adata, groupby="condition")

    genes = adata.var.index.tolist()[:10]
    sctoolbox.marker_genes.mask_rank_genes(adata, genes)
    tables = sctoolbox.marker_genes.get_rank_genes_tables(adata)

    for key in tables:
        table_names = tables[key]["names"].tolist()
        assert len(set(genes) - set(table_names)) == len(genes)  # all genes are masked


@pytest.mark.parametrize("score_name", ["test1", "test2"])
def test_score_genes(adata, score_name):
    """ Test if cell cycle is predicted and added to adata.obs """

    # set gene names as index instead of ensemble ids
    adata.var.reset_index(inplace=True)
    adata.var.set_index('gene', inplace=True)
    adata.var_names_make_unique()

    # test scoring genes with a list
    if score_name == "test1":
        gene_set = adata.var.index.to_list()[:100]
        sctoolbox.marker_genes.score_genes(adata, gene_set, score_name=score_name)

        assert score_name in adata.obs.columns

    # test scoring genes with a list in a file
    elif score_name == "test2":
        gene_set = os.path.join(os.path.dirname(__name__.split('.')[0]), 'data/gene_lists', 'human_mito_genes.txt')
        #genelist_dir = files(__name__.split('.')[0]).joinpath("data/gene_lists/")
        #gene_set = str(genelist_dir / "human_mito_genes.txt")
        sctoolbox.marker_genes.score_genes(adata, gene_set, score_name=score_name)

        assert score_name in adata.obs.columns
