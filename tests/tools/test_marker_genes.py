"""Test gene marker functions."""

import pytest
import os
import scanpy as sc
import numpy as np
import pandas as pd
import sctoolbox.tools.marker_genes as mg


# ---------------------------- FIXTURES -------------------------------- #

@pytest.fixture
def adata():
    """Create testing adata."""

    np.random.seed(1)  # set seed for reproducibility

    h5ad = os.path.join(os.path.dirname(__file__), '../data', 'adata.h5ad')
    adata = sc.read_h5ad(h5ad)

    sample_names = ["C1_1", "C1_2", "C2_1", "C2_2", "C3_1", "C3_2"]
    adata.obs["samples"] = np.random.choice(sample_names, size=adata.shape[0])
    adata.obs["condition"] = adata.obs["samples"].str.split("_", expand=True)[0]
    adata.obs["condition-col"] = adata.obs["condition"]
    adata.obs["long_condition"] = ["Tooooooooo_loooooong_duplicate_1"] * (len(adata.obs) // 2) + ["Tooooooooo_loooooong_duplicate_2"] * (len(adata.obs) - len(adata.obs) // 2)

    # Raw counts for DESeq2
    adata.layers["raw"] = adata.layers["spliced"] + adata.layers["unspliced"]

    return adata


@pytest.fixture
def adata_score(adata):
    """Prepare adata for scoring/ cell cycle test."""

    # set gene names as index instead of ensemble ids
    adata.var.reset_index(inplace=True)
    adata.var['gene'] = adata.var['gene'].astype('str')
    adata.var.set_index('gene', inplace=True)
    adata.var_names_make_unique()

    return adata


@pytest.fixture
def gene_set(adata_score):
    """Return subset of adata genes."""
    return adata_score.var.index.to_list()[:50]


# ------------------------------ TESTS --------------------------------- #


def test_get_chromosome_genes():
    """Test if get_chromosome_genes get the right genes from the gtf."""

    gtf = os.path.join(os.path.dirname(__file__), '../data', 'genes.gtf')

    with pytest.raises(Exception):
        mg.get_chromosome_genes(gtf, "NA")

    genes_chr1 = mg.get_chromosome_genes(gtf, "chr1")
    genes_chr11 = mg.get_chromosome_genes(gtf, "chr11")

    assert genes_chr1 == ["DDX11L1", "WASH7P", "MIR6859-1"]
    assert genes_chr11 == ["DGAT2"]


@pytest.mark.parametrize("m_genes, r_genes, g_genes", [("internal", None, None),
                                                       (None, "internal", None),
                                                       (None, None, "internal")])
def test_label_genes_failure(adata, m_genes, r_genes, g_genes):
    """Test label_genes fails on incorrect parameter combination."""
    with pytest.raises(ValueError):
        mg.label_genes(adata, species=None, m_genes=m_genes, r_genes=r_genes, g_genes=g_genes)


@pytest.mark.parametrize("species, m_genes, r_genes, g_genes", [("mouse", "internal", "internal", "internal"),
                                                                (None, ["mt-Tf", "mt-Rnr1", "mt-Tv"], None, None),
                                                                (None, None, str(mg._GENELIST_LOC / "mouse_ribo_genes.txt"), None),])
def test_label_genes(adata, species, m_genes, r_genes, g_genes):
    """Test label_genes success."""
    added_columns = ["is_ribo", "is_mito", "is_gender"]
    assert not any(c in added_columns for c in adata.var.columns)

    mg.label_genes(adata, species=species, m_genes=m_genes, r_genes=r_genes, g_genes=g_genes)

    if species == "mouse":
        missing = set(added_columns) - set(adata.var.columns)  # test that columns were added
        assert len(missing) == 0
    else:
        assert "is_mito" in adata.var.columns and "is_ribo" in adata.var.columns  # is_gender is not added


def test_add_gene_expression(adata):
    """Test add_gene_expression success and failure."""
    gene = adata.var.index[0]

    # success
    assert f"{gene}_values" not in adata.obs.columns
    mg.add_gene_expression(adata=adata, gene=gene)
    assert f"{gene}_values" in adata.obs.columns

    # failure
    with pytest.raises(ValueError):
        mg.add_gene_expression(adata=adata, gene="INVALID")


def test_get_rank_genes_tables(adata):
    """Test if rank gene tables are created and saved to excel file."""

    sc.tl.rank_genes_groups(adata, groupby="condition")

    tables = mg.get_rank_genes_tables(adata, out_group_fractions=True, save_excel="rank_genes.xlsx")

    assert len(tables) == 3
    assert os.path.exists("rank_genes.xlsx")

    os.remove("rank_genes.xlsx")


@pytest.mark.parametrize("alt_name", [{}, {"Tooooooooo_loooooong_duplicate_1": "Grp_1", "Tooooooooo_loooooong_duplicate_2": "Grp_2"}])
def test_get_rank_genes_tables_duplicates(adata, alt_name):
    """Test the handling of duplicated group names during excel write."""
    sc.tl.rank_genes_groups(adata, groupby="long_condition")

    if alt_name == {}:
        with pytest.raises(ValueError):
            _ = mg.get_rank_genes_tables(adata, out_group_fractions=True, save_excel="rank_genes.xlsx", alt_name=alt_name)
    else:
        _ = mg.get_rank_genes_tables(adata, out_group_fractions=True, save_excel="rank_genes.xlsx", alt_name=alt_name)

        os.remove("rank_genes.xlsx")


@pytest.mark.parametrize("kwargs", [{"var_columns": ["invalid", "columns"]}])  # save_excel must be str
def test_get_rank_genes_tables_errors(adata, kwargs):
    """Test if get_rank_gene_tables raises errors."""

    sc.tl.rank_genes_groups(adata, groupby="condition")

    with pytest.raises(ValueError):
        mg.get_rank_genes_tables(adata, out_group_fractions=True, **kwargs)


def test_mask_rank_genes(adata):
    """Test if genes are masked in adata.uns['rank_genes_groups']."""

    sc.tl.rank_genes_groups(adata, groupby="condition")

    genes = adata.var.index.tolist()[:10]
    mg.mask_rank_genes(adata, genes)
    tables = mg.get_rank_genes_tables(adata)

    for key in tables:
        table_names = tables[key]["names"].tolist()
        assert len(set(genes) - set(table_names)) == len(genes)  # all genes are masked


@pytest.mark.parametrize("condition_col, error, contrast", [
    ("not_present", "are not found in adata.obs. Available columns are:", None),
    ("condition", None, "valid"),  # with specifically selected constrasts
    ("condition", None, None),  # with all contrasts
    ("condition", "is not valid. Valid contrasts are:", [("invalid", "invalid")])])
def test_run_deseq2(adata, condition_col, error, contrast):
    """Test if deseq2 is run and returns a dataframe."""

    # test if error is raised
    if isinstance(error, str):
        if condition_col == "not_present":
            with pytest.raises(KeyError, match=error):
                mg.run_deseq2(adata, sample_col="samples", condition_col=condition_col, layer="raw")
        elif condition_col == "condition":
            with pytest.raises(ValueError, match=error):
                mg.run_deseq2(adata, sample_col="samples", contrasts=contrast, condition_col=condition_col, layer="raw")

    else:  # should run without exceptions
        if contrast == "valid":
            conditions = list(set(adata.obs[condition_col]))
            contrast = [(conditions[0], conditions[1])]

        df = mg.run_deseq2(adata, sample_col="samples", contrasts=contrast, condition_col=condition_col, layer="raw")

        assert isinstance(df, pd.DataFrame)


@pytest.mark.parametrize(
    "score_name, gene_set, inplace",
    [
        ("test1", "gene_set", False),
        ("test2", os.path.join(os.path.dirname(__file__), 'data', 'test_score_genes.txt'), True)
    ],
    indirect=["gene_set"]
)
def test_score_genes(adata_score, score_name, gene_set, inplace):
    """Test if genes are scored and added to adata.obs."""

    assert score_name not in adata_score.obs.columns

    out = mg.score_genes(adata_score, gene_set, score_name=score_name, inplace=inplace)

    if inplace:
        assert out is None
        assert score_name in adata_score.obs.columns
    else:
        assert score_name not in adata_score.obs.columns
        assert score_name in out.obs.columns


@pytest.mark.parametrize("groupby", ["samples"])
def test_run_rank_genes(adata, groupby):
    """Test ranking genes function."""

    adata.uns["log1p"] = {"base": [1, 2, 3]}
    mg.run_rank_genes(adata, groupby=groupby, n_genes=10)
    assert adata.uns[f"rank_genes_{groupby}"]
    assert adata.uns[f"rank_genes_{groupby}_filtered"]


def test_run_rank_genes_fail(adata):
    """Test if invalid input is caught."""

    adata = adata.copy()
    adata.obs["invalid_cat"] = "invalid"

    with pytest.raises(ValueError, match='groupby must contain at least two groups.'):
        mg.run_rank_genes(adata, groupby="invalid_cat")


def test_pairwise_rank_genes(adata):
    """Test pairwise_rank_genes success."""
    output = mg.pairwise_rank_genes(adata=adata, groupby="samples")

    assert isinstance(output, pd.DataFrame)
