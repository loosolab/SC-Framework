import math
import os
import statistics
import sys


def annot_ct(adata=None, genes_adata=None, output_path=None, db_path=None, cluster_path=None, cluster_column=None, rank_genes_column=None, sample="sample", ct_column="cell_types", tissue="all", db="panglao", inplace=True):
    """
    If the script is called via a package (atactoolbox), please use this function.
    This function calculates potential cell types per cluster and adds them to the obs table of the anndata object.

    Parameters
    ----------
    adata : anndata.AnnData, default None
        The anndata object containing clustered data to annotate.
    genes_adata : anndata.AnnData, default None
        The anndata object which contains gene names as index aswell as rank genes groups.
    output_path : string, default None
        The path to the folder where the annotation file will be written and where the ranks folder will be created.
    db_path : string, default None
        The path to the cell type marker gene database file.
    cluster_path : string, default None
        The path to the folder which contains the "cluster files": Tab-separated files containing the genes and
        the corresponding ranked scores.
    cluster_column : string, default None
        The column of the .obs table which contains the clustering information.
    rank_genes_column : string, default None
        The column of the .uns table which contains the rank genes scores.
    sample : string, default "sample"
        The name of the sample.
    ct_column : string, default "cell_types"
        The column of the .obs table which will include the new cell type annotation.
    tissue : string, default "all"
        If tissue is not "all", only marker genes found in the entered tissue will be taken into account.
    db : string, default "panglao"
        The name of the cell type marker gene database which will be used.
    inplace : boolean, default True
        Whether to add the annotations to the adata object in place.

    Returns
    --------
    If inplace == True, the annotation is added to adata.var in place. 
    Else, a copy of the adata object is returned with the annotations added.

    """
    if inplace == False:
        adata = adata.copy()

    if output_path and db_path:
        cluster_path = f"{output_path}/ranked/clusters/{cluster_column}/"
        ct_path = f"{output_path}/{cluster_column}/"

        print("Output folder: " + ct_path, "\nDB file: " + db_path, f"\nCluster folder: {cluster_path}",
        "\nTissue: " + tissue, "\nDB: " + db)
        if adata and genes_adata and cluster_column:
            # Create folders containing the annotation assignment table aswell as the detailed scoring files per cluster
            if not os.path.exists(f'{output_path}/ranked/clusters/{cluster_column}'):
                os.makedirs(f'{output_path}/ranked/clusters/{cluster_column}')
                print(f'Created folder: {output_path}/ranked/clusters/{cluster_column}')

            if not os.path.exists(f'{output_path}/ranked/output/{cluster_column}'):
                os.makedirs(f'{output_path}/ranked/output/{cluster_column}')
                print(f'Created folder: {output_path}/ranked/output/{cluster_column}')

            # Write one file per cluster containing gene names and ranked gene scores
            print("Writing one file per cluster containing gene names and ranked gene scores.")
            for cluster in adata.obs[f'{cluster_column}'].unique():
                with open(f'{output_path}/ranked/clusters/{cluster_column}/{sample}.{cluster}', 'w') as file:
                    for index, gene in enumerate(genes_adata.uns[f'{rank_genes_column}']['names'][cluster]):
                        score = genes_adata.uns[f'{rank_genes_column}']['scores'][cluster][index]
                        file.write(f'{gene.split("_")[0]}\t{score}\n')

            # Perform the actual cell type annotation per clustering resolution
            print("Starting cell type annotation.")
            perform_cell_type_annotation(
                f"{output_path}/ranked/output/{cluster_column}/", db_path, cluster_path, tissue, db=db)

            # Add information to the adata object
            print("Adding information to the adata object.")
            cta_dict = {}
            with open(f'{output_path}/ranked/output/{cluster_column}/annotation.txt') as file:
                for line in file:
                    cluster, ct = line.split('\t')
                    cta_dict[cluster] = ct.rstrip()
            adata.obs[f'{ct_column}'] = adata.obs[f'{cluster_column}'].map(cta_dict)

            print(f"Finished cell type annotation! The results are found in the .obs table {ct_column}.")

            if inplace == False:
                return adata

        elif cluster_path:
            print("Output folder: " + output_path, "\nDB file: " + db_path, "\nCluster folder: " + cluster_path,
                    "\nTissue: " + tissue, "\nDB: " + db)
            perform_cell_type_annotation(
                f"{output_path}/ranked/output/{cluster_column}/", db_path, cluster_path, tissue, db=db)
            print(f"Cell type annotation of output path {output_path} finished.")

        else:
            pass


def get_panglao(path, tissue="all"):
    """
    Read and parse the panglao cell type marker gene database file.

    path : string
        The path to the panglao cell type marker gene database file.

    tissue : string, default "all"
        If tissue is not "all", only marker genes found in the entered tissue will be taken into account.

    Returns
    -------
    dictionary :
        Dictionary which contains a dictionary per cell type. The inner dictionary contains the corresponding
        marker genes (keys) and the values of the ubiquitousness indices (values).
    """

    tissues = [tissue]
    panglao_dict = {}
    panglao_rank_dict = {}

    with open(path, "r") as panglao_file:
        panglao_file.readline()
        for line in panglao_file.readlines():
            species, gene_symb, ct, n_genes, ub_i, organ = line.split("\t")
            us = float(ub_i)
            if us != 0:
                us = round(math.sqrt(1 / us))
            else:
                us = 32
            if "all" in tissues and "Hs" in species:
                if ct not in panglao_dict.keys():
                    panglao_dict[ct] = []
                genes = [(us, gene_symb)]
                if len(n_genes.split("|")) > 1:
                    for gene in n_genes.split("|"):
                        genes.append((us, gene.upper()))
                elif n_genes != "NA":
                    genes.append((us, n_genes))
                for gene in genes:
                    panglao_dict[ct].append(gene)
            elif any(t in organ.lower() for t in tissues) and "Hs" in species:
                if ct not in panglao_dict.keys():
                    panglao_dict[ct] = []
                genes = [(us, gene_symb)]
                if len(n_genes.split("|")) > 1:
                    for gene in n_genes.split("|"):
                        genes.append((us, gene.upper()))
                elif n_genes != "NA":
                    genes.append((us, n_genes))
                for gene in genes:
                    panglao_dict[ct].append(gene)

    for ct in panglao_dict.keys():
        rank_dict = {}
        for gene in panglao_dict[ct]:
            rank_dict[gene[1]] = gene[0]

        panglao_rank_dict[ct] = rank_dict

    return panglao_rank_dict


def calc_ranks(cm_dict, annotated_clusters):
    """
    Identify cell types of each cluster by ranking each potential cell type using fitting genes, ranked scores,
    quantity of available marker genes per cell type aswell as using the panglao ubiquitousness index.

    Parameters
    ----------
    cm_dict : dictionary
        Dictionary which contains the cell marker database.
    annotated_clusters :
        Dictionary which contains the summed up ranked scores per gene for each cluster.

    Returns
    -------
    dictionary :
        The dictionary which contains the scores, the quantity of hits, the overall marker genes and
        the ubiquitousness index per cell type for each cluster.
    """

    ct_dict = {}

    for key in annotated_clusters.keys():
        ct_dict[key] = {}

    for celltype in cm_dict.keys():
        gene_count = len(cm_dict[celltype])
        for c in annotated_clusters.keys():
            count = 0
            ranks = []
            ub_scores = []

            for mgene in annotated_clusters[c].keys():
                if mgene in cm_dict[celltype].keys():
                    gene_score, ub_score = annotated_clusters[c][mgene], cm_dict[celltype][mgene]
                    gene_score = gene_score * ub_score
                    ranks.append(gene_score)
                    ub_scores.append(ub_score)
                    count += 1

            if count > 4:
                ub_mean = round(statistics.mean(ub_scores))
                ct_dict[c][celltype] = [round(sum(ranks) / math.sqrt(count)), count, gene_count,
                                        ub_mean]

    return ct_dict


def get_cell_types(cluster_path, db_path, tissue="all", db="panglao"):
    """
    Prepare database and clusters for upcoming ranking calculations.

    Parameters
    ----------
    cluster_path : string
        The path to the folder which contains the "cluster files": Tab-separated files containing the
        genes and the corresponding ranked scores.
    db_path : string
        The path to the cell type marker gene database file.
    tissue : string, default "all"
        If tissue is not "all", only marker genes found in the entered tissue will be taken into account.
    db : string, default "panglao"
        The name of the cell type marker gene database which will be used.

    Returns
    -------
    dictionary :
        The dictionary which contains the scores, the quantity of hits, the overall marker genes and
        the ubiquitousness index per cell type for each cluster.
    """

    if db == "panglao":
        print("Loading PanglaoDB")
        db_dict = get_panglao(db_path, tissue=tissue)
    else:
        print("DB " + db + " not supported.")
        exit(1)

    annotated_clusters = get_annotated_clusters(cluster_path=cluster_path)

    return calc_ranks(db_dict, annotated_clusters)


def get_annotated_clusters(cluster_path):
    """
    Read cluster files and sum ranked scores if genes appear more than once per file.

    Parameters
    ----------
    cluster_path : string
        The path to the folder which contains the "cluster files": Tab-separated files containing the
        genes and the corresponding ranked scores.

    Returns
    -------
    dictionary :
        Dictionary which contains the summed up ranked scores per gene for each cluster.
    """

    annotated_clusters = {}
    files = os.listdir(cluster_path)
    for file in [x for x in files if not x.startswith(".")]:
        cname = file.split(".")[1]
        annotated_dict = {}
        with open(cluster_path + file) as cfile:
            annotated_dict[cname] = []
            lines = cfile.readlines()
            for line in lines:
                split = line.split("\t")
                if len(split) == 2:
                    annotated_dict[cname].append(
                        [split[0], float(split[1].rstrip())])

        sum_dict = {}
        for gene in annotated_dict[cname]:
            if gene[0] in sum_dict.keys():
                sum_dict[gene[0]] += gene[1]
            else:
                sum_dict[gene[0]] = gene[1]

        annotated_clusters[cname] = sum_dict

    return annotated_clusters


def perform_cell_type_annotation(output, db_path, cluster_path, tissue="all", db="panglao"):
    """
    Performs cell type identification, generate cell type assignment table
    and create ranks folder with files for further investigation (one per cluster).

    Parameters
    ----------
    output : string
        The path to the folder where the annotation file will be written and where the ranks folder will be created.
    db_path : string
        The path to the cell type marker gene database file.
    cluster_path : string
        The path to the folder which contains the "cluster files": Tab-separated files containing the genes and
        the corresponding ranked scores.
    tissue : string, default "all"
        If tissue is not "all", only marker genes found in the entered tissue will be taken into account.
    db : string, default "panglao"
        The name of the cell type marker gene database which will be used.
    """
    opath = output + "/ranks/"
    if not os.path.exists(opath):
        os.makedirs(opath)

    ct_dict = get_cell_types(cluster_path, db_path, tissue,
                             db=db)
    write_annotation(ct_dict, output)


def write_annotation(ct_dict, output):
    """
    Writes a tab-separated file that contains exactly one cell type (the one with the highest score)
    for each cluster to which at least one cell type could be assigned ("cell type assignment table").

    Parameters
    ----------
    ct_dict : dictionary
        The dictionary which contains the scores, the quantity of hits, the overall marker genes and
        the ubiquitousness index per cell type for each cluster. This dictionary is being returned
        by the calc_ranks() method.
    output : string
        The path to the folder where the annotation file will be written.

    """
    with open(output + "/annotation.txt", "w") as c_file:
        for dic in ct_dict.keys():
            sorted_dict = dict(
                sorted(ct_dict[dic].items(), key=lambda r: (r[1][0], r[1][1]), reverse=True))
            with open(output + "/ranks/" + "cluster_" + dic, "w") as d_file:
                for key in sorted_dict.keys():
                    d_file.write(key)
                    for value in sorted_dict[key]:
                        d_file.write("\t" + str(value))
                    d_file.write("\n")
            if len(sorted_dict.keys()) > 0:
                c_file.write(dic + "\t" + str(next(iter(sorted_dict))) + "\n")


def main():
    """
    Using ranked genes scores per gene per cluster to perform cell type annotation.
    Command line parameters: output path, path to marker db file, path to cluster folder (containing tab separated gene and score),
    tissue ("all" for all tissues), database (only panglao implemented yet).
    """

    if len(sys.argv) == 6:
        output, db_path, cluster_path, tissue, db = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], \
            sys.argv[5]
        print("Output folder: " + output, "\nDB file: " + db_path, "\nCluster folder: " + cluster_path,
              "\nTissue: " + tissue, "\nDB: " + db)
        perform_cell_type_annotation(
            output, db_path, cluster_path, tissue, db=db)
        print("Cell type annotation of " + output + " finished.")
    else:
        print("Please use three, five, six or seven parameters only!")
        print("Example: python3 cell_type_annotation.py output_path panglao_path cluster_path all panglao")
        exit(1)


if __name__ == '__main__':
    main()
