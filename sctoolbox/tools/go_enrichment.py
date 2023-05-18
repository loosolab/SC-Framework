import seaborn as sns
import warnings
import gseapy as gp
import pandas as pd
import numpy as np

import sctoolbox.utils as utils


class GOresult(pd.DataFrame):

    @property
    def _constructor(self):
        return GOresult

    def barplot(self,
                gene_set=None,
                top_term=20,
                title=None,
                colors=None,
                figsize=None,
                save=None,
                ):

        # Take title from object if not given
        if title is None and hasattr(self, "title"):
            title = self.title

        table_to_plot = self[self['Adjusted P-value'] < self.cutoff]

        if gene_set is None:
            table_to_plot = table_to_plot.groupby("Gene_set").head(top_term)
            group = "Gene_set"

        else:
            if gene_set not in self.gene_sets:
                raise ValueError()

            table_to_plot = table_to_plot[self["Gene_set"] == gene_set]
            table_to_plot = table_to_plot.head(top_term)
            group = None

        # Calculate uptimal figure size
        if figsize is None:
            figsize = (5, len(table_to_plot) / 3 + 0.5)

        # Color per group
        n_groups = len(set(table_to_plot["Gene_set"]))
        if colors is None:
            colors = [sns.color_palette()[i] for i in range(n_groups)]

        # Plot barplot
        ax = gp.barplot(table_to_plot,
                        group=group,
                        top_term=top_term,
                        figsize=figsize,
                        color=colors)

        threshold = -np.log10(self.cutoff)
        _ = ax.axvline(threshold, color="black", linestyle="--")

        # Adjust title / xlabel
        ax.set_xlabel(ax.get_xlabel(), fontweight="normal")  # remove bold text
        if title is not None:
            ax.set_title(title, fontsize=20)

        utils.save_figure(save)

        return ax

    def bubbleplot(self, gene_set):
        pass


def GO_enrichment(genes, organism,
                  gene_sets=None,
                  threshold=0.05,
                  verbose=True):

    if gene_sets is None:
        available_gene_sets = gp.get_library_name(organism=organism)
        gene_sets = available_gene_sets

    else:
        # if gene_set not in available_gene_sets:
        for gene_set in gene_sets:
            pass
            # raise error

    # Enricr API
    enr = gp.enrichr(genes,
                     gene_sets=gene_sets,
                     organism=organism,
                     outdir=None,
                     cutoff=threshold,
                     verbose=verbose)
    result_obj = GOresult(enr.results)

    # Save information to the object
    for att in ['module', 'gene_sets', 'gene_list', 'cutoff']:
        val = getattr(enr, att)

        # Set attribute and catch warning
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=UserWarning, message="Pandas doesn't allow columns to be created")
            setattr(result_obj, att, val)

    return result_obj
