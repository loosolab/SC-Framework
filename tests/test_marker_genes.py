import pytest
import os
import scanpy as sc
import numpy as np
import tempfile
import sctoolbox.marker_genes as mg


# ---------------------------- FIXTURES -------------------------------- #

@pytest.fixture
def adata():

    np.random.seed(1)  # set seed for reproducibility

    h5ad = os.path.join(os.path.dirname(__file__), 'data', 'adata.h5ad')
    adata = sc.read_h5ad(h5ad)

    sample_names = ["C1_1", "C1_2", "C2_1", "C2_2", "C3_1", "C3_2"]
    adata.obs["samples"] = np.random.choice(sample_names, size=adata.shape[0])
    adata.obs["condition"] = adata.obs["samples"].str.split("_", expand=True)[0]
    adata.obs["condition-col"] = adata.obs["condition"]

    # Raw counts for DESeq2
    adata.layers["raw"] = adata.layers["spliced"] + adata.layers["unspliced"]

    return adata


@pytest.fixture
def adata_score(adata):
    """ Prepare adata for scoring/ cell cycle test. """

    # set gene names as index instead of ensemble ids
    adata.var.reset_index(inplace=True)
    adata.var['gene'] = adata.var['gene'].astype('str')
    adata.var.set_index('gene', inplace=True)
    adata.var_names_make_unique()

    return adata


@pytest.fixture
def s_genes(adata_score):
    return adata_score.var.index[:int(len(adata_score.var) / 2)].tolist()


@pytest.fixture
def g2m_genes(adata_score):
    return adata_score.var.index[int(len(adata_score.var) / 2):].tolist()


@pytest.fixture
def g2m_file(g2m_genes):
    """ Write a tmp file, which is deleted after usage. """
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = os.path.join(tmpdir, "g2m_genes.txt")

        with open(tmp, "w") as f:
            f.writelines([g + "\n" for g in g2m_genes])

        yield tmp


@pytest.fixture
def s_file(s_genes):
    """ Write a tmp file, which is deleted after usage. """
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = os.path.join(tmpdir, "s_genes.txt")

        with open(tmp, "w") as f:
            f.writelines([g + "\n" for g in s_genes])

        yield tmp


@pytest.fixture
def gene_set(adata_score):
    return adata_score.var.index.to_list()[:50]


# ------------------------------ TESTS --------------------------------- #

def test_get_chromosome_genes():
    """ Test if get_chromosome_genes get the right genes from the gtf """

    gtf = os.path.join(os.path.dirname(__file__), 'data', 'genes.gtf')

    with pytest.raises(Exception):
        mg.get_chromosome_genes(gtf, "NA")

    genes_chr1 = mg.get_chromosome_genes(gtf, "chr1")
    genes_chr11 = mg.get_chromosome_genes(gtf, "chr11")

    assert genes_chr1 == ["DDX11L1", "WASH7P", "MIR6859-1"]
    assert genes_chr11 == ["DGAT2"]


@pytest.mark.parametrize("species, gene_column", [("mouse", None),
                                                  ("unicorn", "gene")])
def test_label_genes(adata, species, gene_column):
    """ Test of genes are labeled in adata.var """

    if species is None:
        with pytest.raises(ValueError):
            mg.label_genes(adata, species=species)  # no species given, and it cannot be found in infoprocess

    else:
        mg.label_genes(adata, gene_column=gene_column, species=species)

    added_columns = ["is_ribo", "is_mito", "cellcycle", "is_gender"]
    missing = set(added_columns) - set(adata.var.columns)  # test that columns were added

    if species == "mouse":
        assert len(missing) == 0
    else:
        assert "is_mito" in adata.var.columns and "is_ribo" in adata.var.columns  # is_gender and cellcycle are not added


def test_get_rank_genes_tables(adata):
    """ test if rank gene tables are created and saved to excel file """

    sc.tl.rank_genes_groups(adata, groupby="condition")

    tables = mg.get_rank_genes_tables(adata, out_group_fractions=True, save_excel="rank_genes.xlsx")

    assert len(tables) == 3
    assert os.path.exists("rank_genes.xlsx")


def test_mask_rank_genes(adata):
    """ Test if genes are masked in adata.uns['rank_genes_groups'] """

    sc.tl.rank_genes_groups(adata, groupby="condition")

    genes = adata.var.index.tolist()[:10]
    mg.mask_rank_genes(adata, genes)
    tables = mg.get_rank_genes_tables(adata)

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
def test_predict_cell_cycle(adata_score, species, s_genes, g2m_genes, inplace):
    """ Test if cell cycle is predicted and added to adata.obs """
    expected_columns = ["S_score", "G2M_score", "phase"]

    assert not any(c in adata_score.obs.columns for c in expected_columns)

    if species == "unicorn":
        with pytest.raises(ValueError):
            mg.predict_cell_cycle(adata_score, species=species)
            return

    out = mg.predict_cell_cycle(adata_score, species=species, s_genes=s_genes, g2m_genes=g2m_genes, inplace=inplace)

    if inplace:
        assert out is None
        assert all(c in adata_score.obs.columns for c in expected_columns)
    else:
        assert not any(c in adata_score.obs.columns for c in expected_columns)
        assert all(c in out.obs.columns for c in expected_columns)


@pytest.mark.parametrize(
    "score_name, gene_set, inplace",
    [
        ("test1", "gene_set", False),
        ("test2", os.path.join(os.path.dirname(__file__), 'data', 'test_score_genes.txt'), True)
    ],
    indirect=["gene_set"]
)
def test_score_genes(adata_score, score_name, gene_set, inplace):
    """ Test if genes are scored and added to adata.obs """

    assert score_name not in adata_score.obs.columns

    out = mg.score_genes(adata_score, gene_set, score_name=score_name, inplace=inplace)

    if inplace:
        assert out is None
        assert score_name in adata_score.obs.columns
    else:
        assert score_name not in adata_score.obs.columns
        assert score_name in out.obs.columns


# Outcommented because the CI job currently does not have R and DESeq2 installed
# Can be outcommented for testing locally
#
# @pytest.mark.parametrize("condition_col, error",
#                         [("not_present", "was not found in adata.obs.columns"),
#                          ("condition-col", "not a valid column name within R"),
#                          ("condition", None)])
# def test_deseq(adata, condition_col, error):
#    """ Test if deseq2 is run and returns a dataframe """
#
#    # test if error is raised
#    if isinstance(error, str):
#        with pytest.raises(ValueError, match=error):
#            mg.run_deseq2(adata, sample_col="samples", condition_col=condition_col, layer="raw")
#
#    else:  # should run without exceptions
#        df = mg.run_deseq2(adata, sample_col="samples", condition_col=condition_col, layer="raw")
#
#        assert type(df).__name__ == "DataFrame"
