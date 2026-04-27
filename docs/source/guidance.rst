Guidance & Recommendations
==========================

Here we focus on key steps in the analysis, to give a better understanding on the influence of parameter choices. Also check the :doc:`/tutorials/index`.

# TODO
# add details on why steps are done and what the influence of important parameters is
# Something akin to parameter xy controls ... which has effect ... we recommend ...
# Support this with figures and sources.
# Within the notebooks add a link to the sections in this document

RNA
---

# TODO check the last workshop slides


Doublet detection
~~~~~~~~~~~~~~~~~

Doublets or multiplets are artifacts that are created when two or more cells are sequenced with the same barcode. They can appear as separate cell populations in the embedding step and should be removed. We utilize `Scrublet <https://github.com/swolock/scrublet>`_ for doublet detection and follow their recommendations. The stringency of whether a cell classifies as a doublet may be controled through the ``doublet_threshold`` parameter. On default, this is automatically set via Scrublet, however, manual intervention is possible. The ``doublet_threshold`` applies to the ``doublet_score`` a score that ranges from 0-1. The score may be interpreted as the chance of a cell being a doublet, where higher values mean *more likely*.

.. figure:: image/scrublet_threshold.png
   :alt: Scrublet doublet score distribution histograms.

   Histrograms of barcodes distributed by doublet score. The black vertical line indicates the doublet threshold. Lower values (left of the line) are considered singlets, while higher values (right of the line) are treated as doublets. Image taken from `Scrublet <https://github.com/swolock/scrublet>`__.

The barcodes are usually distributed bimodal, where the first (left) modality represents singlets and the second (right) modality represents doublets. The ``doublet_threshold`` should be placed in the vally between the two modalities, as scores higher than the threshold are considered doublets.

Further references:

- `Single-cell best practices <https://www.sc-best-practices.org/preprocessing_visualization/quality_control.html#doublet-detection>`_
- `Wolock et al. <https://doi.org/10.1016/j.cels.2018.11.005>`_

Filter metrics
~~~~~~~~~~~~~~





- filter (mito, ribo, apoptosis, gender, etc.); what are good values?
- automated threshold calculation
- highly variable genes
- pca + pc selection
- nearest neighbors
- batch correction (LISI)
- embedding (umap/tsne)
- clustering (recluster; mention marker genes)
- marker genes (needs details)
  - min_in_group_fraction
  - min_fold_change
  - max_out_group_fraction

ATAC
----

- filter
  - fld score
  - overlap
  - frip score
  - tsse
  - binarize
- doublet
- normalization (tf-idf vs total)
- highly variable features
- lsi vs pca
