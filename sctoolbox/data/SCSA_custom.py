#! /usr/bin/python
#########################################################################
# File Name: scRNA_anno.py
# > Author: CaoYinghao
# > Mail: caoyinghao@gmail.com
#########################################################################

import sys
import argparse
import os

import numpy as np
from numpy import abs, mean, array, std, log2
# import pandas as pd
from pandas import DataFrame, read_csv, ExcelWriter
from scipy.sparse import coo_matrix


class Annotator(object):
    def __init__(self, args):
        self.args = args

    @staticmethod
    def to_output(h_values, wb, outtag, cname, title):
        if outtag.lower() == "ms-excel":
            h_values.to_excel(wb, sheet_name="Cluster " + cname + " " + title, index=False)
        else:
            h_values.to_csv(wb, sep="\t", quotechar="\t", index=False, header=False)

    def print_class(self, h_values, cname):
        """print cell predictions with scores."""

        o = ""
        titlebar = "-" * 60 + "\n"

        if h_values is None:
            if self.args.noprint:
                return "E", None, "-", "-", "-"

            o += titlebar
            o += "{0:<10}{1:^30}{2:<10}".format("Type", "Cell Type", "Score")
            o += "\n" + "-" * 60 + "\n"
            o += "{0:<10}{1:^30}{2:<10}".format("-", "-", "-")
            o += "\n" + titlebar

            return "E", None, "-", "-", "-"

        elif h_values.size == 0:
            if self.args.noprint:
                return "N", None, "-", "-", "-"
            o += titlebar
            o += "{0:<10}{1:^30}{2:<10}".format("Type", "Cell Type", "Score")
            o += "\n" + "-" * 60 + "\n"
            o += "{0:<10}{1:^30}{2:<10}".format("-", "-", "-")
            o += "\n" + titlebar
            return "N", None, "-", "-", "-"

        elif h_values.size == 3:
            if self.args.noprint:
                return "Good", o, h_values.values[0][0], h_values.values[0][1], "-"
            o += titlebar
            o += "{0:<10}{1:^30}{2:<10}{3:<5}".format("Type", "Cell Type", "Score", "Times")
            o += "\n" + "-" * 60 + "\n"
            o += "{0:<10}{1:^30}{2:<10.4f}".format("Good", h_values.values[0][0], h_values.values[0][1])
            o += "\n" + titlebar
            return "Good", o, h_values.values[0][0], h_values.values[0][1], "-"

        elif h_values.size == 2:
            if self.args.noprint:
                return "Good", o, h_values.values[0][0], h_values.values[0][1], "-"
            o += titlebar
            o += "{0:<10}{1:^30}{2:<10}{3:<5}".format("Type", "Cell Type", "Score", "Times")
            o += "\n" + "-" * 60 + "\n"
            o += "{0:<10}{1:^30}{2:<10.4f}".format("Good", h_values.values[0][0], h_values.values[0][1])
            o += "\n" + titlebar
            return "Good", o, h_values.values[0][0], h_values.values[0][1], "-"

        elif float(h_values.iloc[0, 1]) / float(h_values.iloc[1, 1]) >= 2 or float(h_values.iloc[1, 1] < 0):
            times = np.abs(float(h_values.iloc[0, 1]) / float(h_values.iloc[1, 1]))
            if self.args.noprint:
                return "Good", o, h_values.values[0][0], h_values.values[0][1], times
            o += titlebar
            o += "{0:<10}{1:^30}{2:<10}{3:<5}".format("Type", "Cell Type", "Score", "Times")
            o += "\n" + titlebar
            o += "{0:<10}{1:^30}{2:<10.4f}{3:<5.1f}".format("Good", h_values['Cell Type'].values[0], h_values['Z-score'].values[0], times)
            o += "\n" + titlebar
            return "Good", o, h_values['Cell Type'].values[0], h_values['Z-score'].values[0], times

        else:
            times = np.abs(float(h_values.iloc[0, 1]) / float(h_values.iloc[1, 1]))
            if self.args.noprint:
                return "?", o, str(h_values['Cell Type'].values[0]) + "|" + str(h_values['Cell Type'].values[1]), str(h_values['Z-score'].values[0]) + "|" + str(h_values['Z-score'].values[1]), times
            o += titlebar
            o += "{0:<10}{1:^30}{2:<10}{3:<5}".format("Type", "Cell Type", "Score", "Times")
            o += "\n" + titlebar
            o += "{0:<10}{1:^30}{2:<10.4f}{3:<5.1f}".format("?", h_values['Cell Type'].values[0], h_values['Z-score'].values[0], times)
            o += "\n" + titlebar
            o += "{0:<10}{1:^29}({2:<.4f})".format("", "(" + h_values['Cell Type'].values[1] + ")", h_values['Z-score'].values[1])
            o += "\n" + titlebar
            return "?", o, str(h_values['Cell Type'].values[0]) + "|" + str(h_values['Cell Type'].values[1]), str(h_values['Z-score'].values[0]) + "|" + str(h_values['Z-score'].values[1]), times

    def calcu_scanpy_group(self, expfile):
        """deal with scanpy input matrix"""

        exps = read_csv(expfile, index_col=0)
        cnum = set()
        pname = "p"
        lname = "l"
        nname = "n"
        for c in exps.columns:
            k, v = c.split("_")
            cnum.add(k)
            if v.startswith("p"):  # pvalue
                pname = v
            elif v.startswith("n"):  # gene name
                nname = v
            elif v.startswith("l"):  # logfc
                lname = v

        self.pname = pname
        self.lname = lname
        self.nname = nname

        # Initialize the output
        outs = []
        self.wb = None
        if self.args.output:
            if self.args.outfmt.lower() == "ms-excel":
                if not self.args.output.endswith(".xlsx") and (not self.args.output.endswith(".xls")):
                    self.args.output += ".xlsx"
                self.wb = ExcelWriter(self.args.output)

            elif self.args.outfmt.lower() == "txt":
                self.wb = open(self.args.output, "w")
                self.wb.write("Cell Type\tZ-score\tCluster\n")

            else:
                print("Error output format: -m, -outfmt,(ms-excel,[txt])")
                sys.exit(0)

        # Loop over clusters
        for i in list(sorted(cnum)):
            cname = str(i)

            o = " ".join(["#" * 30, "Cluster", cname, "#" * 30]) + "\n"
            if self.args.noprint is False:
                print(o)

            lcol = cname + "_" + lname  # logfoldchanges
            pcol = cname + "_" + pname  # pvals
            ncol = cname + "_" + nname  # names
            # ptitle = cname + "_" + pname  #pvals
            if lcol not in exps.columns:
                print(lcol, "column not in the input table!")
                sys.exit(0)

            # filter on logfc and pvalue
            newexps = exps[[ncol, lcol, pcol]][(exps[lcol] >= self.args.foldchange) & (exps[pcol] <= self.args.pvalue)]
            if newexps.empty:
                raise ValueError('Dataframe empty after filtering. Try loosening the filter criteria.')
            print("Cluster " + cname + " Gene number:", newexps[ncol].unique().shape[0])

            # Calculate the H values
            h_values, colnames = self.get_cell_matrix(newexps, lcol, ncol)

            if self.args.output:
                h_values['Cluster'] = cname
                Annotator.to_output(h_values, self.wb, self.args.outfmt, cname, "Cell Type")

            t, o_str, c, v, times = self.print_class(h_values, cname)
            outs.append([cname, t, c, v, times])
            if self.args.noprint is False:
                print(o_str)

        if self.args.output:
            self.wb.close()

        if self.args.noprint is False:
            print("#" * 80 + "\n")

        return outs

    def get_exp_matrix_loop(self, exps, lcol, ncol, genenames, cellnames, cell_matrix, abs_tag=True):
        """format the cell_deg_matrix and calculate the zscore of certain cell types."""

        # filter gene expressed matrix according to the markers
        gene_exps = exps.loc[:, [ncol, lcol]][exps[ncol].isin(genenames)]

        gene_matrix = np.asmatrix(gene_exps.sort_values(ncol)[lcol]).T
        gene_matrix = gene_matrix * np.mean(gene_matrix)   # ## / np.min(gene_matrix))

        if gene_matrix.shape[0] != cell_matrix.shape[1]:
            # print(gene_matrix.shape,cell_matrix.shape)
            # print(len(gene_exps[fid].unique()))
            # print(gene_matrix)
            # print(cell_matrix)
            print("Error for inconsistent gene numbers, please check your expression csv for '" + ncol + "'")
            return None

        # nonzero = np.matrix(np.count_nonzero(cell_matrix, axis=1)).T
        cell_deg_matrix = cell_matrix * gene_matrix
        cell_deg_matrix = np.matrix(np.array(cell_deg_matrix))

        out = DataFrame({"Z-score": cell_deg_matrix.A1}, index=cellnames)
        out.sort_values(['Z-score'], inplace=True, ascending=False)

        if abs_tag:
            out['Z-score'] = abs(out['Z-score'])
        else:
            out = out[out['Z-score'] > 0]

        if (out.shape[0] > 1):
            out['Z-score'] = (out['Z-score'] - mean(out['Z-score'])) / std(out['Z-score'], ddof=1)

        return out

    def get_user_cell_gene_names(self, exps, ncol):
        """ find expressed markers according to the user markers and expressed matrix."""

        # columns in input markers
        gcol = self.args.genecol
        ccol = self.args.cellcol

        # Subset markers to the genes in the expression matrix
        cluster_genes = set(exps[ncol])
        whole_fil = self.usermarkers[gcol].isin(cluster_genes)
        fc = self.usermarkers[[ccol, gcol, 'weight']][whole_fil]

        if fc.shape[0] == 0:
            print("!WARNING3:Zero marker sets found")
            print("!WARNING3:Change the threshold or tissue name and try again?")
            print("!WARNING3:EnsemblID or GeneID,try '-E' command?")
            return fc, [], cluster_genes

        # Remove duplicate rows and sum weights
        fc.columns = [ccol, gcol, 'c']
        fc.set_index([ccol, gcol])

        newfc = fc.groupby([ccol, gcol]).sum()
        names = newfc.index

        newfc['c1'] = names
        newfc[gcol] = newfc['c1'].apply(lambda x: x[1])
        newfc[ccol] = newfc['c1'].apply(lambda x: x[0])
        newfc.drop(['c1'], inplace=True, axis=1)
        newfc.reset_index(drop=True, inplace=True)

        newfc['c'] = log2(newfc['c'] + 0.05)  # * np.min(newfc['c'])
        fc = newfc

        cellnames = sorted(set(self.usermarkers[ccol].unique()))
        genenames = sorted(set(fc[gcol].unique()))

        return fc, cellnames, genenames

    def get_cell_matrix(self, exps, lcol, ncol):
        """ Combine cell matrix with weight-matrix"""

        cell_value, genenames = self.get_cell_matrix_detail(exps, lcol, ncol)

        # Database weight-matrix
        wm = [1]
        weight_matrix = np.asmatrix(wm).T

        if genenames is None:
            return DataFrame(), None

        if cell_value is None:
            return DataFrame(), set(genenames)

        last_value = array(cell_value) * weight_matrix
        result = DataFrame({"Cell Type": cell_value.index, "Z-score": last_value.A1})
        result = result.sort_values(by="Z-score", ascending=False)

        return result, set(genenames)

    def get_cell_matrix_detail(self, exps, lcol, ncol):
        """ Calculate the cell type scores"""

        fc, cellnames, genenames = self.get_user_cell_gene_names(exps, ncol)
        cellnum = len(cellnames)
        genenum = len(genenames)

        if not genenames:
            return None, None

        if fc.shape[0] == 0:
            return None, set(genenames)

        exps = exps[exps[ncol].isin(genenames)]

        gcol = self.args.genecol
        ccol = self.args.cellcol

        rowdic = dict(zip(cellnames, range(cellnum)))
        coldic = dict(zip(genenames, range(genenum)))
        fc_cell = fc[ccol].map(lambda x: rowdic[x])
        fc_gene = fc[gcol].map(lambda x: coldic[x])

        newdf = DataFrame({ccol: fc_cell, gcol: fc_gene, "c": fc['c']})
        cell_coo_matrix = coo_matrix((newdf['c'], (newdf[ccol], newdf[gcol])), shape=(cellnum, genenum))
        cell_matrix = cell_coo_matrix.toarray()

        if self.args.noprint is False:
            print("Celltype Num:", cellnum)
            print("Gene Num in database:", genenum)
            print("Not Zero:", cell_coo_matrix.count_nonzero())

        cell_values = self.get_exp_matrix_loop(exps, lcol, ncol, genenames, cellnames, cell_matrix)

        return cell_values, set(genenames)

    def read_user_markers(self):
        """ usermarker db preparation """

        if not os.path.exists(self.args.db):
            err = f"User marker database does not exists! Path: {self.args.db}"
            sys.exit(err)

        self.usermarkers = read_csv(self.args.db, sep="\t")
        self.usermarkers['weight'] = 1

        cellcol = self.args.cellcol
        genecol = self.args.genecol
        if self.args.noprint is False:
            print("DB celltypes:", len(self.usermarkers[cellcol].unique()))
            print("DB genes:", len(self.usermarkers[genecol].unique()))

    def run_detail_cmd(self):
        """main command"""

        if not os.path.exists(self.args.input):
            tempname = "./" + self.args.input
            if not os.path.exists(tempname):
                print(tempname)
                print("Input file does not exists!", self.args.input)
                sys.exit(0)

        self.read_user_markers()

        outs = self.calcu_scanpy_group(self.args.input)
        return outs


class Process(object):

    def __init__(self):
        pass

    def get_parser(self):
        desc = """Program: SCSA
  Version: 1.0
  Email  : <yhcao@ibms.pumc.edu.cn>
        """

        parser = argparse.ArgumentParser(formatter_class=argparse.RawDescriptionHelpFormatter, description=desc)

        parser.add_argument('-i', '--input', required=True, help="Input file for marker annotation (Only CSV format supported).")
        parser.add_argument('-o', '--output', help="Output file for marker annotation.")
        parser.add_argument('-d', '--db', help="'User-defined marker database in table format with at least two columns (celltype name and gene name). Use -C and -G to specify the column names of celltype and gene name.")
        parser.add_argument('-C', '--cellcol', default="cell_name", help='Column name of cellname in user-defined marker database')
        parser.add_argument('-G', '--genecol', default="gene", help='Column name of genename in user-defined marker database')
        parser.add_argument('-f', "--foldchange", default=2, help="Fold change threshold for marker filtering. (2.0)")
        parser.add_argument('-p', "--pvalue", default=0.05, help="P-value threshold for marker filtering. (0.05)")
        parser.add_argument('-m', '--outfmt', default="ms-excel", help="Output file format for marker annotation. (ms-excel,[txt])")
        parser.add_argument('-b', "--noprint", action="store_true", default=False, help="Do not print any detail results.")

        return parser

    def run_cmd(self, args):

        args.foldchange = float(args.foldchange)
        args.pvalue = float(args.pvalue)

        anno = Annotator(args)
        outs = anno.run_detail_cmd()

        print("#Cluster", "Type", "Celltype", "Score", "Times")
        for o in outs:
            print(o)
        pass


if __name__ == "__main__":
    p = Process()
    parser = p.get_parser()
    args = parser.parse_args()
    p.run_cmd(args)
