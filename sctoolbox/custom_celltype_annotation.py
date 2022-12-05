import math
import os
import statistics
import sys
import argparse
from collections import Counter


def get_c_dict(clusters):
    """
    Count all genes which appear in any cluster
    :param clusters: dictionary
    :return: dictionary
    """
    c_dict = {}

    for c in clusters.keys():
        for gene in clusters[c].keys():
            if gene in c_dict.keys():
                c_dict[gene] += 1
            else:
                c_dict[gene] = 1

    return c_dict


def get_duplicates(c_dict, th):
    """
    Detect genes which appear in >= th clusters
    :param c_dict: dictionary
    :param th: int
    :return: list of strings
    """
    filtered = []

    for gene in c_dict.keys():
        if c_dict[gene] >= th:
            filtered.append(gene)

    return filtered


def get_panglao(tissue, panglao, connect=False, smooth=False):
    """
    Read and parse panglao database file
    :param tissue: string
    :param panglao: string
    :param connect: boolean
    :param smooth: boolean
    :return: dictionary
    """
    tissues = [tissue]
    path = panglao
    panglao_dict = {}
    panglao_rank_dict = {}

    if connect:
        tissues.append("tissue")

    if smooth:
        tissues.append("smooth")

    if "artery" in tissues:
        tissues.append("vessel")

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


def get_hcm(tissue, hcm, connect=False, smooth=False):
    """
    Read and parse panglao database file
    :param tissue: string
    :param hcm: string
    :param connect: boolean
    :param smooth: boolean
    :return: dictionary
    """
    tissues = [tissue]
    path = hcm
    hcm_dict = {}

    if connect:
        tissues.append("tissue")

    if smooth:
        tissues.append("smooth")

    if "artery" in tissues:
        tissues.append("vessel")

    with open(path, "r") as hcm_file:
        hcm_file.readline()
        for line in hcm_file.readlines():
            organ, ct, gene_symb = line.split("\t")

            if any(t in organ.lower() for t in tissues):
                if ct not in hcm_dict.keys():
                    hcm_dict[ct] = []
                genes = []
                if len(gene_symb.split(", ")) > 1:
                    for gene in gene_symb.split(", "):
                        genes.append(gene.upper().rstrip())
                elif gene_symb != "NA":
                    genes.append(gene_symb.upper().rstrip())
                for gene in genes:
                    hcm_dict[ct].append(gene)

    return hcm_dict


def get_cell_types(cpath, db_path, tissue, db="panglao", filter_genes=False, connect=False, smooth=False):
    """
    Prepare database and clusters for upcoming ranking calculations
    :param cpath: string
    :param db_path: string
    :param tissue: string
    :param connect: boolean
    :param smooth: boolean
    :return: dictionary
    """
    if db == "panglao":
        print("Loading PanglaoDB")
        db_dict = get_panglao(tissue, db_path, connect=connect, smooth=smooth)
    elif db == "hcm":
        print("Loading Human Cell Marker DB")
        db_dict = get_hcm(tissue, db_path, connect=connect, smooth=smooth)
    else:
        print("DB " + db + " not supported.")
        exit(1)
        
    if filter_genes:
        clusters = get_annotated_clusters(path=cpath)
        
        print("Detecting genes which appear in " + str(math.ceil(len(clusters) / 1.5)) + " or more clusters")
        dupls = get_duplicates(get_c_dict(clusters), math.ceil(len(clusters) / 1.5))

        print("Filter genes which appear in " + str(math.ceil(len(clusters) / 1.5)) + " or more clusters")
        filtered_clusters = {}
        for cluster_num in clusters.keys():
            filtered_clusters[cluster_num] = {}
            for gene in clusters[cluster_num].keys():
                if gene not in dupls:
                    filtered_clusters[cluster_num][gene] = clusters[cluster_num][gene]
    else:
        filtered_clusters = get_annotated_clusters(path=cpath)



    def calc_ranks(cm_dict):
        """
        Identify cell types of each cluster by ranking each fitting gene by peak value,
        quantity and panglao ubiquitousness index
        :param cm_dict: dictionary
        :return: dictionary
        """
        ct_dict = {}

        for key in filtered_clusters.keys():
            ct_dict[key] = {}

        for celltype in cm_dict.keys():
            print("Calculating scores of cell type " + celltype)
            gene_count = len(cm_dict[celltype])
            for c in filtered_clusters.keys():
                count = 0
                ranks = []
                ub_scores = []

                for mgene in filtered_clusters[c].keys():
                    if db == "panglao":
                        if mgene in cm_dict[celltype].keys():
                            ranks.append(filtered_clusters[c][mgene] * cm_dict[celltype][mgene])
                            ub_scores.append(cm_dict[celltype][mgene])
                            count += 1
                    elif db == "hcm":
                        if mgene in cm_dict[celltype]:
                            ranks.append(filtered_clusters[c][mgene])
                            count += 1

                if count > 4:
                    if db == "panglao":
                        ub_score = round(statistics.mean(ub_scores))
                        ct_dict[c][celltype] = [round(sum(ranks) / math.sqrt(count)), count, gene_count,
                                                ub_score]
                    elif db == "hcm":
                        ct_dict[c][celltype] = [round(sum(ranks) / gene_count), count, gene_count]
        return ct_dict

    return calc_ranks(db_dict)


def annot_peaks(apath, ppath, cpath):
    """
    Annotate parsed narrow peak files with parsed uropa finalhits files and save annotated cluster files to cpath
    :param apath: string
    :param ppath: string
    :param cpath: string
    """
    for afile in os.listdir(apath):
        with open(cpath + afile.split("_peaks")[0], "w") as cfile:
            for pfile in os.listdir(ppath):
                if afile.split("_peaks")[0] == pfile.split("_peaks")[0]:
                    with open(apath + afile) as a_file:
                        alines = a_file.readlines()
                    with open(ppath + pfile) as p_file:
                        plines = p_file.readlines()
                    for i in range(len(alines)):
                        asplit = alines[i].split()
                        # TODO: remove artifacts with missing columns
                        # Adhoc workaround:
                        if len(asplit) > 2:
                            aline = asplit[0] + asplit[1] + asplit[2]
                            for j in range(len(plines)):
                                psplit = plines[j].split()
                                # TODO: remove artifacts with missing columns
                                # Adhoc workaround:
                                if len(psplit) > 2:
                                    pline = psplit[0] + psplit[1] + psplit[2]
                                    if aline == pline:
                                        cfile.write(alines[i].split("\t")[3].rstrip() + "\t" + plines[j].split("\t")[3])
                                        break


def get_annotated_clusters(path):
    """
    Read cluster files and sum peak values if genes appear more than once per file
    :param path: string
    :return: dictionary
    """
    annotated_clusters = {}
    files = os.listdir(path)
    for file in [x for x in files if not x.startswith(".")]:
        cname = file.split(".")[1]
        annotated_dict = {}
        with open(path + file) as cfile:
            annotated_dict[cname] = []
            lines = cfile.readlines()
            for line in lines:
                split = line.split("\t")
                if len(split) == 2:
                    annotated_dict[cname].append([split[0], float(split[1].rstrip())])

        sum_dict = {}
        for gene in annotated_dict[cname]:
            if gene[0] in sum_dict.keys():
                sum_dict[gene[0]] += gene[1]
            else:
                sum_dict[gene[0]] = gene[1]

        annotated_clusters[cname] = sum_dict

    return annotated_clusters


def perform_cell_type_annotation(output, db_path, annot, npeaks, tissue, db="panglao", connect=True, smooth=False):
    """
    Perform cell type identification and annotation, create cluster folder, generate cell type assignment table
    and create ranks folder with files for further investigation
    :param output: string
    :param db_path: string
    :param annot: string
    :param npeaks: string
    :param tissue: string
    :param db: string
    :param connect: boolean
    :param smooth: boolean
    """
    apath, ppath, cpath, opath = [annot, npeaks,
                                  output + "/cluster/", output + "/ranks/"]

    if not os.path.exists(cpath):
        os.makedirs(cpath)
        print("Annotating peaks of clusters")
        annot_peaks(apath=apath, ppath=ppath, cpath=cpath)

    if not os.path.exists(opath):
        os.makedirs(opath)

    print("Starting cell type annotation")
    ct_dict = get_cell_types(cpath, db_path, tissue, db=db, connect=connect, smooth=smooth)
    write_annotation(ct_dict, output)


def perform_cell_type_annotation_c_folder(output, db_path, c_folder, tissue, db="panglao", filter_genes=False, connect=True, smooth=False):
    """
    Perform cell type identification with cluster folder only, generate cell type assignment table
    and create ranks folder with files for further investigation
    :param output: string
    :param db_path: string
    :param c_folder: string
    :param tissue: string
    :param db: string
    :param connect: boolean
    :param smooth: boolean
    """
    opath = output + "/ranks/"
    if not os.path.exists(opath):
        os.makedirs(opath)

    print("Starting cell type annotation")
    ct_dict = get_cell_types(c_folder, db_path, tissue, db=db, filter_genes=filter_genes, connect=connect, smooth=smooth)
    write_annotation(ct_dict, output)


def write_annotation(ct_dict, output):
    with open(output + "/annotation.txt", "w") as c_file:
        for dic in ct_dict.keys():
            sorted_dict = dict(sorted(ct_dict[dic].items(), key=lambda r: (r[1][0], r[1][1]), reverse=True))
            with open(output + "/ranks/" + "cluster_" + dic, "w") as d_file:
                for key in sorted_dict.keys():
                    d_file.write(key)
                    for value in sorted_dict[key]:
                        d_file.write("\t" + str(value))
                    d_file.write("\n")
            if len(sorted_dict.keys()) > 0:
                c_file.write(dic + "\t" + str(next(iter(sorted_dict))) + "\n")


def create_combined_files(score_path, peak_path, output_path):
    """
    Create files containing peak values combined with ranked gene scores
    :param score_path: string
    :param peak_path: string
    :param output_path: string
    """
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    score_dict = get_annotated_clusters(score_path)
    peak_dict = get_annotated_clusters(peak_path)

    clusters = list(score_dict.keys())
    clusters.extend(list(peak_dict.keys()))
    clusters = list(set(clusters))
    
    comb_dict = {}
    for cluster in clusters:
        if cluster in score_dict.keys() and cluster in peak_dict.keys():
            for gene in score_dict[cluster].keys():
                if gene in peak_dict[cluster].keys():
                    if cluster not in comb_dict.keys():
                        comb_dict[cluster] = {}
                    comb_dict[cluster][gene] = int(score_dict[cluster][gene] + peak_dict[cluster][gene]) / 2
        elif cluster in score_dict.keys():
            for gene in score_dict[cluster].keys():
                    if cluster not in comb_dict.keys():
                        comb_dict[cluster] = {}
                    comb_dict[cluster][gene] = score_dict[cluster][gene]
        elif cluster in peak_dict.keys():
            for gene in peak_dict[cluster].keys():
                    if cluster not in comb_dict.keys():
                        comb_dict[cluster] = {}
                    comb_dict[cluster][gene] = peak_dict[cluster][gene]

    for cluster in comb_dict.keys():
        with open(output_path + "/combined_cluster." + str(cluster), "w") as comb_file:
            for gene in comb_dict[cluster].keys():
                comb_file.write(gene + "\t" + str(comb_dict[cluster][gene]) + "\n")

    print("Created combined cluster files.")


def main():
    """
    Using annotated peaks and narrow peak values from each cluster to perform cell type annotation.
    Command line parameters: output path, path to marker db file, path to annotated peaks, path to narrow peaks,
    tissue, database (panglao or hcm) and connective tissue. Leave sixth parameter blank if not taking connective
    tissue into account.
    """

    if len(sys.argv) == 8:
        output, db_path, annot, npeaks, tissue, db, connect = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], \
                                                              sys.argv[5], sys.argv[6], True
        print("Output folder: " + output, "\nDB file: " + db_path, "\nAnnotated peaks: " + annot,
              "\nNarrow peaks: " + npeaks, "\nTissue: " + tissue, "\nDB: " + db,
              "\nInclude connective tissue: " + str(connect))
        perform_cell_type_annotation(output, db_path, annot, npeaks, tissue, db=db, connect=connect)
        print("Cell type annotation of " + output + " finished.")
    elif len(sys.argv) == 7:
        filter_genes = False
        output, db_path, c_folder, tissue, db, connect = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], \
                                                              sys.argv[5], False
        if sys.argv[6] !="filter":
            print("Output folder: " + output, "\nDB file: " + db_path, "\nAnnotated peaks: " + annot,
                  "\nNarrow peaks: " + npeaks, "\nTissue: " + tissue, "\nDB: " + db,
                  "\nInclude connective tissue: " + str(connect))
            perform_cell_type_annotation_c_folder(output, db_path, annot, npeaks, tissue, db=db, filter_genes=filter_genes)
        else:
            filter_genes = True
            print("Output folder: " + output, "\nDB file: " + db_path, "\nCluster folder: " + c_folder,
                  "\nTissue: " + tissue, "\nDB: " + db, "\nFilter genes: True")
            perform_cell_type_annotation_c_folder(output, db_path, c_folder, tissue, db=db, connect=connect, filter_genes=filter_genes)
        print("Cell type annotation of " + output + " finished.")
    elif len(sys.argv) == 6:
        output, db_path, c_folder, tissue, db = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], \
                                                         sys.argv[5]
        print("Output folder: " + output, "\nDB file: " + db_path, "\nCluster folder: " + c_folder,
              "\nTissue: " + tissue, "\nDB: " + db)
        perform_cell_type_annotation_c_folder(output, db_path, c_folder, tissue, db=db)
        print("Cell type annotation of " + output + " finished.")
    elif len(sys.argv) == 4:
        score_path, peak_path, output_path = sys.argv[1], sys.argv[2], sys.argv[3]
        create_combined_files(score_path, peak_path, output_path)
    else:
        print("Please use three, five, six or seven parameters only!")
        print("Example: python3 cell_type_annotation.py output_path panglao_path annot_path npeaks_path \"gi tract\" panglao T")
        exit(1)


if __name__ == '__main__':
    main()

