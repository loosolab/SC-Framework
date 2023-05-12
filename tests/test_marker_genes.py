import pytest
import os
import scanpy as sc
import numpy as np
import pandas as pd
import tempfile
import sctoolbox.marker_genes


# ---------------------------- FIXTURES -------------------------------- #

@pytest.fixture
def adata():

    h5ad = os.path.join(os.path.dirname(__file__), 'data', 'adata.h5ad')
    adata = sc.read_h5ad(h5ad)

    adata.obs["condition"] = np.random.choice(["C1", "C2", "C3"], size=adata.shape[0])

    return adata


@pytest.fixture
def adata_cc(adata):
    """ Prep adata for cell cycle test. """
    # set gene names as index instead of ensemble ids
    adata.var.reset_index(inplace=True)
    adata.var['gene'] = adata.var['gene'].astype('str')
    adata.var.set_index('gene', inplace=True)
    adata.var_names_make_unique()

    return adata


@pytest.fixture
def cc_table():
    return pd.read_csv(os.path.join(os.path.dirname(__file__), '../sctoolbox/data/', 'mouse_cellcycle_genes.txt'), sep="\t", header=None)


@pytest.fixture
def s_genes(cc_table):
    return cc_table[cc_table.loc[:, 1] == "s_genes"][0]


@pytest.fixture
def g2m_genes(cc_table):
    return cc_table[cc_table.loc[:, 1] == "g2m_genes"][0]


@pytest.fixture
def g2m_file(g2m_genes):
    """ Write a tmp file, which is deleted after usage. """
    g2m_genes += "\n"
    
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = os.path.join(tmpdir, "g2m_genes.txt")

        with open(tmp, "w") as f:
            f.writelines(g2m_genes)

        yield tmp


@pytest.fixture
def s_file(s_genes):
    """ Write a tmp file, which is deleted after usage. """
    s_genes += "\n"
    
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = os.path.join(tmpdir, "s_genes.txt")

        with open(tmp, "w") as f:
            f.writelines(s_genes)

        yield tmp


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


@pytest.mark.parametrize(
    "species, s_genes, g2m_genes, inplace",
    [
        # ("mouse", None, None, False),  # can not test on species as no cell cycle genes are present in testdata
        (None, "s_file", "g2m_file", True),
        (None, "s_genes", "g2m_genes", True),
        ("unicorn", None, None, False)
    ],
    indirect=["s_genes", "g2m_genes"]
)
def test_predict_cell_cycle(adata_cc, species, s_genes, g2m_genes, inplace):
    """ Test if cell cycle is predicted and added to adata.obs """
    expected_columns = ["S_score", "G2M_score", "phase"]

    assert not any(c in adata_cc.obs.columns for c in expected_columns)

    if species == "unicorn":
        with pytest.raises(ValueError):
            sctoolbox.marker_genes.predict_cell_cycle(adata_cc, species=species)
            return

    out = sctoolbox.marker_genes.predict_cell_cycle(adata_cc, species=species, s_genes=s_genes, g2m_genes=g2m_genes, inplace=inplace)

    if inplace:
        assert out is None
        assert all(c in adata_cc.obs.columns for c in expected_columns)
    else:
        assert not any(c in adata_cc.obs.columns for c in expected_columns)
        assert all(c in out.obs.columns for c in expected_columns)
