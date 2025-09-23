"""Bio related utility functions."""

import numpy as np
import pandas as pd
import re
import requests
import apybiomart
from scipy.sparse import issparse
import gzip
import argparse
import os
import pybedtools
import scanpy as sc

from beartype.typing import Optional, Literal, Tuple, Any
from beartype import beartype

import sctoolbox.utils as utils
import sctoolbox.utils.decorator as deco


@deco.log_anndata
@beartype
def pseudobulk_table(adata: sc.AnnData,
                     groupby: str,
                     how: Literal['mean', 'sum'] = "mean",
                     layer: Optional[str] = None,
                     percentile_range: Tuple[int, int] = (0, 100),
                     chunk_size: int = 1000,
                     gene_index: Optional[str] = None,
                     clean: bool = True) -> pd.DataFrame:
    """
    Get a pseudobulk table of values per cluster.

    Parameters
    ----------
    adata : sc.AnnData
        Anndata object with counts in .X.
    groupby : str
        Column name in adata.obs from which the pseudobulks are created.
    how : Literal['mean', 'sum'], default 'mean'
        How to calculate the value per group (psuedobulk).
    layer : Optional[str], default None
        Name of an anndata layer to use instead of `adata.X`.
    percentile_range : Tuple[int, int], default (0, 100)
        The percentile of cells used to calculate the mean/sum for each feature.
        Is used to limit the effect of individual cell outliers, e.g. by setting (0, 95) to exclude high values in the calculation.
    chunk_size : int, default 1000
        If percentile_range is not default, chunk_size controls the number of features to process at once. This is used to avoid memory issues.
    gene_index : Optional[str], default None
        Column in adata.var that holds gene symbols/ ids.
    clean : bool, default True
        Removes NaN indexes and adds a suffix to duplicated indexes (e.g. 'gene_1', 'gene_2'). Especially important in combination with `gene_index`.
        Changes index type to str.

    Returns
    -------
    pd.DataFrame
        DataFrame with aggregated counts (adata.X). With groups as columns and genes as rows.

    Raises
    ------
    KeyError
        If the groupby column does not exist in adata.obs.
    """

    # Check if groupby column exists
    if groupby not in adata.obs.columns:
        raise KeyError(f"Column '{groupby}' not found in adata.obs.")

    groupby_categories = adata.obs[groupby].astype('category').cat.categories

    if layer is not None:
        mat = adata.layers[layer]
    else:
        mat = adata.X

    # Fetch the mean/ sum counts across each category in cluster_by
    res = pd.DataFrame(index=adata.var[gene_index] if gene_index else adata.var_names,
                       columns=groupby_categories)
    for column_i, clust in enumerate(groupby_categories):

        cluster_values = mat[list(adata.obs[groupby].isin([clust])), :]

        if percentile_range == (0, 100):  # uses all cells
            if how == "mean":
                vals = cluster_values.mean(0)
                res[clust] = vals.A1 if issparse(cluster_values) else vals
            elif how == "sum":
                vals = cluster_values.sum(0)
                res[clust] = vals.A1 if issparse(cluster_values) else vals
        else:

            n_features = cluster_values.shape[1]

            # Calculate mean individually per gene/feature
            for i in range(0, n_features, chunk_size):

                chunk_values = cluster_values[:, i:i + chunk_size]
                chunk_values = chunk_values.A if issparse(chunk_values) else chunk_values
                chunk_values = chunk_values.astype(float)

                # Calculate the lower and upper limits for each feature
                limits = np.percentile(chunk_values, percentile_range, axis=0, method="lower")
                lower_limits = limits[0]
                upper_limits = limits[1]

                # Set values outside the limits to nan and calculate mean/sum
                bool_filter = (chunk_values < lower_limits) | (chunk_values > upper_limits)
                chunk_values[bool_filter] = np.nan

                if how == "mean":
                    vals = np.nanmean(chunk_values, axis=0)
                elif how == "sum":
                    vals = np.nansum(chunk_values, axis=0)

                res.iloc[i:i + chunk_size, column_i] = vals

    # remove NA and add suffix to duplicate indexes
    if clean:
        res = res.loc[res.index.dropna()]
        res.index = sc.anndata.utils.make_index_unique(res.index.astype(str), join="_")

    return res


#####################################################################
#                        Format adata indexes                       #
#####################################################################

@deco.log_anndata
@beartype
def barcode_index(adata: sc.AnnData) -> None:
    """
    Check if the barcode is the index.

    Will replace the index with `adata.obs["barcode"]` if index does not contain barcodes.

    TODO refactor
    - name could be more descriptive
    - return adata
    - inplace parameter
    - use logger
    ...

    Parameters
    ----------
    adata : sc.AnnData
        Anndata to perform check on.
    """

    # regex for any barcode
    regex = re.compile(r'([ATCG]{8,16})')
    # get first index element
    first_index = str(adata.obs.index[0])
    # check if the first index element is a barcode
    if regex.match(first_index):
        index_is_barcode = True
    else:
        index_is_barcode = False

    if not adata.obs.index.name == "barcode" and not index_is_barcode:
        # check if the barcode column is in the obs
        if 'barcode' in adata.obs.columns:
            print('setting adata.obs.index = adata.obs[barcode]')
            adata.obs = adata.obs.set_index("barcode")
    elif not adata.obs.index.name == "barcode" and index_is_barcode:
        print('setting adata.obs.index.name = barcode')
        adata.obs.index.name = 'barcode'
    else:
        print('barcodes are already the index')


#####################################################################
#                  Converting between gene id and name              #
#####################################################################

@beartype
def get_organism(ensembl_id: str,
                 host: str = "http://www.ensembl.org/id/") -> str:
    """
    Get the organism name to the given Ensembl ID.

    Parameters
    ----------
    ensembl_id : str
        Any Ensembl ID. E.g. ENSG00000164690
    host : str
        Ensembl server address.

    Returns
    -------
    str
        Organism assigned to the Ensembl ID

    Raises
    ------
    ConnectionError
        If there is an unexpected (or no) response from the server.
    ValueError
        If the returned organism is ambiguous.
    """

    # this will redirect
    url = f"{host}{ensembl_id}"
    response = requests.get(url)

    if response.status_code != 200:
        raise ConnectionError(f"Server response: {response.status_code}.\n With link {url}. Is the host/ path correct?")

    # get redirect url
    # e.g. http://www.ensembl.org/Homo_sapiens/Gene/...
    # get species name from url
    species = response.url.split("/")[3]

    # invalid id
    if species == "Multi":
        raise ValueError(f"Organism returned as '{species}' ({response.url}).\n Usually due to invalid Ensembl ID. Make sure to use an Ensembl ID as described in http://www.ensembl.org/info/genome/stable_ids/index.html")

    return species


# TODO: add unit test
@beartype
def gene_id_to_name(ids: list[str], species: str) -> pd.DataFrame:
    """
    Get Ensembl gene names to Ensembl gene id.

    Parameters
    ----------
    ids : list[str]
        List of gene ids. Set to `None` to return all ids.
    species : str
        Species matching the gene ids. Set to `None` for list of available species.

    Returns
    -------
    pd.DataFrame
        DataFrame with gene ids and matching gene names.

    Raises
    ------
    ValueError
        If provided Ensembl IDs or organism is invalid.
    """

    if not all(id.startswith("ENS") for id in ids):
        raise ValueError("Invalid Ensembl IDs detected. A valid ID starts with 'ENS'.")

    avail_species = sorted([s.split("_gene_ensembl")[0] for s in apybiomart.find_datasets()["Dataset_ID"]])

    if species is None or species not in avail_species:
        raise ValueError("Invalid species. Available species are: ", avail_species)

    id_name_mapping = apybiomart.query(
        attributes=["ensembl_gene_id", "external_gene_name"],
        dataset=f"{species}_gene_ensembl",
        filters={}
    )

    if ids:
        # subset to given ids
        return id_name_mapping[id_name_mapping["Gene stable ID"].isin(ids)]

    return id_name_mapping


# TODO: add unit test, currently no usage
@deco.log_anndata
@beartype
def convert_id(adata: sc.AnnData,
               id_col_name: Optional[str] = None,
               index: bool = False,
               name_col: str = "Gene name",
               species: str = "auto",
               inplace: bool = True) -> Optional[sc.AnnData]:
    """
    Add gene names to adata.var.

    Parameters
    ----------
    adata : sc.AnnData
        AnnData with gene ids.
    id_col_name : Optional[str], default None
        Name of the column in `adata.var` that stores the gene ids.
    index : boolean, default False
        Use index of `adata.var` instead of column name speciefied in `id_col_name`.
    name_col : str, default "Gene name"
        Name of the column added to `adata.var`.
    species : str, default "auto"
        Species of the dataset. On default, species is inferred based on gene ids.
    inplace : bool, default True
        Whether to modify adata inplace.

    Returns
    -------
    Optional[sc.AnnData]
        AnnData object with gene names.

    Raises
    ------
    ValueError
        If invalid parameter choice or column name not found in adata.var.
    """

    if not id_col_name and not index:
        raise ValueError("Either set parameter id_col_name or index.")
    elif not index and id_col_name not in adata.var.columns:
        raise ValueError("Invalid id column name. Name has to be a column found in adata.var.")

    if not inplace:
        adata = adata.copy()

    # get gene ids
    if index:
        gene_ids = list(adata.var.index)
    else:
        gene_ids = list(adata.var[id_col_name])

    # infer species from gene id
    if species == "auto":
        ensid = gene_ids[0]

        species = get_organism(ensid)

        # bring into biomart format
        # first letter of all words but complete last word e.g. hsapiens, mmusculus
        spl_name = species.lower().split("_")
        species = "".join(map(lambda x: x[0], spl_name[:-1])) + spl_name[-1]

        print(f"Identified species as {species}")

    # get id <-> name table
    id_name_table = gene_id_to_name(ids=gene_ids, species=species)

    # create new .var and replace in adata
    new_var = pd.merge(
        left=adata.var,
        right=id_name_table.set_index("Gene stable ID"),
        left_on=id_col_name,
        left_index=index,
        right_index=True,
        how="left"
    )
    new_var["Gene name"].fillna('', inplace=True)
    new_var.rename(columns={"Gene name": name_col}, inplace=True)

    adata.var = new_var

    if not inplace:
        return adata


@deco.log_anndata
@beartype
def unify_genes_column(adata: sc.AnnData,
                       column: str,
                       unified_column: str = "unified_names",
                       species: str = "auto",
                       inplace: bool = True) -> Optional[sc.AnnData]:
    """
    Given an adata.var column with mixed Ensembl IDs and Ensembl names, this function creates a new column where Ensembl IDs are replaced with their respective Ensembl names.

    Parameters
    ----------
    adata : sc.AnnData
        AnnData object
    column : str
        Column name in adata.var
    unified_column : str, default "unified_names"
        Defines the column in which unified gene names are saved. Set same as parameter 'column' to overwrite original column.
    species : str, default "auto"
        Species of the dataset. On default, species is inferred based on gene ids.
    inplace : bool, default True
        Whether to modify adata or return a copy.

    Returns
    -------
    Optional[sc.AnnData]
        AnnData object with modified gene column.

    Raises
    ------
    ValueError
        If column name is not found in `adata.var` or no Ensembl IDs in selected column.
    """

    if column not in adata.var.columns:
        raise ValueError(f"Invalid column name. Name has to be a column found in adata.var. Available names are: {adata.var.columns}.")

    if not inplace:
        adata = adata.copy()

    # check for ensembl ids
    ensids = [el for el in adata.var[column] if el.startswith("ENS")]

    if not ensids:
        raise ValueError(f"No Ensembl IDs in adata.var['{column}'] found.")

    # infer species from gene id
    if species == "auto":
        ensid = ensids[0]

        species = get_organism(ensid)

        # bring into biomart format
        # first letter of all words but complete last word e.g. hsapiens, mmusculus
        spl_name = species.lower().split("_")
        species = "".join(map(lambda x: x[0], spl_name[:-1])) + spl_name[-1]

        print(f"Identified species as {species}")

    # get id <-> name table
    id_name_table = gene_id_to_name(ids=ensids, species=species)

    count = 0
    for index, row in adata.var.iterrows():
        if row[column] in id_name_table['Gene stable ID'].values:
            count += 1

            # replace gene id with name
            adata.var.at[index, unified_column] = id_name_table.at[id_name_table.index[id_name_table["Gene stable ID"] == row[column]][0], "Gene name"]
        else:
            adata.var.at[index, unified_column] = adata.var.at[index, column]
    print(f'{count} ensembl gene ids have been replaced with gene names')

    if not inplace:
        return adata


#####################################################################
#                   Check integrity of gtf file                     #
#####################################################################

# TODO: add unit test
@beartype
def _gtf_integrity(gtf: str) -> bool:
    """
    Check if the provided file follows the gtf-format.

    TODO rather return False than raise an error.

    Checks the following:
        - file-ending
        - header ##format: gtf
        - number of columns == 9
        - regex pattern of column 9 matches gtf specific format

    Parameters
    ----------
    gtf : str
        Path to file.

    Returns
    -------
    bool
        True if the file is a valid gtf-file.

    Raises
    ------
    argparse.ArgumentTypeError
        If the file is not in gtf-format.
    """

    regex_header = '#+.*'
    regex_format_column = '#+format: gtf.*'  # comment can start with one or more '#'

    # Initialize parameters
    first_entry = []
    header = False
    format_gtf = False

    if utils.checker._is_gz_file(gtf):
        fp = gzip.open(gtf, 'rt')  # read text (rt) mode
    else:
        fp = open(gtf)

    # Check header (if present) and first entry
    for line in fp:
        if re.match(regex_header, line):
            header = True
            if re.match(regex_format_column, line):
                # Check if format information in the header matches gtf
                format_gtf = True
        else:
            first_entry = line.split(sep='\t')
            break

    fp.close()  # done reading from file

    # Check if number of columns matches 9
    if len(first_entry) != 9:
        raise argparse.ArgumentTypeError('Number of columns in the gtf file unequal 9')

    # If there was a header, check that the header matches gtf
    if header and not format_gtf:
        raise argparse.ArgumentTypeError('Header in gtf file does not match gtf format')

    # Extract gene_id information from column 9 and check if the format is gtf (not gff)
    column_9 = first_entry[8]
    column_9_split = column_9.split(sep=';')
    gene_id = column_9_split[0]  # TODO; it cannot always be assumed that the gene_id is the first entry in column 9
    # gtf specific format of the gene_id column (gff3: gene_id="xxxxx"; gtf: gene_id "xxxxx")
    regex_gene_id = 'gene_id ".*"'

    # check match of the pattern
    if not re.match(regex_gene_id, gene_id):
        raise argparse.ArgumentTypeError('gtf file is corrupted')

    return True  # the function only returns of no error is raised


@beartype
def _overlap_two_bedfiles(bed1: str, bed2: str, overlap: str, **kwargs: Any) -> None:
    """
    Overlap two bedfiles and writes the overlap to a new bedfile.

    Parameters
    ----------
    bed1 : str
        path to bedfile1
    bed2 : str
        path to bedfile2
    overlap : str
        path to output bedfile
    **kwargs : Any
        Additional arguments passed to pybedtools.BedTool.intersect

    Returns
    -------
    None
    """
    # Ensure that path to bedtools is set
    pybedtools.helpers.set_bedtools_path(utils.checker._add_path())
    # Load the bed files using Pybedtools
    bedfile1 = pybedtools.BedTool(bed1)
    bedfile2 = pybedtools.BedTool(bed2)

    # Perform the intersection
    # The kwargs can be passed directly to the intersect function
    intersected = bedfile1.intersect(bedfile2, **kwargs)

    # Save the output
    intersected.saveas(overlap)


@beartype
def _read_bedfile(bedfile: str, comment: str = "#") -> list:
    """
    Read in a bedfile and returns a list of the rows.

    Parameters
    ----------
    bedfile : str
        path to bedfile
    comment : str, default "#"
        Ignore lines starting with the given characters.

    Returns
    -------
    list
        list of bedfile rows
    """
    bed_list = []
    with open(bedfile, 'rb') as file:
        for row in file:
            row = row.decode("utf-8")

            # skip comment lines
            if row.startswith(comment):
                continue

            row = row.split('\t')
            line = [str(row[0]), int(row[1]), int(row[2]), str(row[3]), int(row[4])]
            bed_list.append(line)

    return bed_list


@beartype
def _bed_is_sorted(bedfile: str, comment: str = "#") -> bool:
    """
    Check if a bedfile is sorted by the start position.

    Parameters
    ----------
    bedfile : str
        path to bedfile
    comment : str, default "#"
        Ignore lines starting with the given characters.

    Returns
    -------
    bool
        True if bedfile is sorted
    """
    with open(bedfile) as file:
        counter = 0
        previous = 0
        for row in file:
            # skip comment lines
            if row.startswith(comment):
                continue

            row = row.split('\t')
            if previous > int(row[1]):
                file.close()
                return False

            previous = int(row[1])

            counter += 1
            if counter > 100:
                break

    file.close()
    return True


@beartype
def _sort_bed(bedfile: str, sorted_bedfile: str) -> None:
    """
    Sort a bedfile by the start position.

    Parameters
    ----------
    bedfile : str
        path to bedfile
    sorted_bedfile : str
        path to sorted bedfile

    Returns
    -------
    None
    """
    sort_cmd = f'sort -k1,1 -k2,2n {bedfile} > {sorted_bedfile}'
    os.system(sort_cmd)
