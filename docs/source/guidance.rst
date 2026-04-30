Guidance & Recommendations
==========================

Here we focus on key steps in the analysis, to give a better understanding on the influence of parameter choices. Also check the :doc:`/tutorials/index`. The following should be viewed as suggestions and recommendations that worked for us and others (see the added references), however they should not be treated as definitive rules and may change in the future.

# TODO
# add details on why steps are done and what the influence of important parameters is
# Something akin to parameter xy controls ... which has effect ... we recommend ...
# Support this with figures and sources.
# Within the notebooks add a link to the sections in this document

RNA
---

# TODO check the last workshop slides


Doublet detection
^^^^^^^^^^^^^^^^^

Doublets or multiplets are artifacts that are created when two or more cells are sequenced with the same barcode. They can appear as separate cell populations in the embedding step and should be removed. We utilize `Scrublet <https://github.com/swolock/scrublet>`_ for doublet detection and follow their recommendations. The stringency of whether a cell classifies as a doublet may be controled through the ``doublet_threshold`` parameter. On default, this is automatically set via Scrublet, however, manual intervention is possible. The ``doublet_threshold`` applies to the ``doublet_score`` a score that ranges from 0-1. The score may be interpreted as the chance of a cell being a doublet, where higher values mean *more likely*.

.. figure:: image/scrublet_threshold.png
   :alt: Scrublet doublet score distribution histograms.

   Histrograms of barcodes distributed by doublet score. The black vertical line indicates the doublet threshold. Lower values (left of the line) are considered singlets, while higher values (right of the line) are treated as doublets. Image taken from `Scrublet <https://github.com/swolock/scrublet>`__.

The barcodes are usually distributed bimodal, where the first (left) modality represents singlets and the second (right) modality represents doublets. The ``doublet_threshold`` should be placed in the vally between the two modalities, as scores higher than the threshold are considered doublets.

Further references:

- `Single-cell best practices - doublet <https://www.sc-best-practices.org/preprocessing_visualization/quality_control.html#doublet-detection>`_
- `Wolock et al. <https://doi.org/10.1016/j.cels.2018.11.005>`_

Filter metrics
^^^^^^^^^^^^^^

Here we look at metrics commonly used in QC filtering. The SC-Framework has two methods to automatically set upper and lower thresholds for each metric, which are explained below. The automatic thresholds can be adjusted by the user as necessary.

Automatic thresholds
""""""""""""""""""""

The SC-Framework has two methods to automatically set filter thresholds, called *MAD* and *GMM*. The *MAD* method, short for median absolute deviation, is calculated as


.. math:: :label: mad_formula

  MAD=median(|X - median(X)|)

  MAD_{score}=median(X) \pm MAD \cdot n_{1, 2}

where :math:`X` is the metric to find a threshold for and :math:`n_{1, 2}` are multipliers for upper and lower bounds. The *GMM* thresholds are calculated similar with the key difference of calculating the thresholds on the largest gaussian mixture within the data. The *GMM* method based on *MAD* developed by the SC-Framework.

.. math:: :label: gmm_formula

  GMM_{score}=mean(GMM(X)_{max}) \pm \sigma_{GMM(X)_{max}} \cdot n_{1, 2}


where :math:`GMM(X)_{max}` is the largest component from a gaussian mixture model of the metric :math:`X` and :math:`n_{1, 2}` are multipliers for upper and lower bounds.

.. note::

  The automatic thresholds should be viewed as suggestions and can be adjusted as necessary.

.. seealso::

  :func:`sctoolbox.tools.qc_filter.mad_threshold`
  :func:`sctoolbox.tools.qc_filter.gmm_threshold`


Barcode (cell) Metrics
""""""""""""""""""""""

Filtering is process of removing low quality data, e.g. cells, while keeping biological relevant information. This is often a trade-off. Stricter filters will remove more low quality data but have the potential to also remove more biological details, while lenient filters keep more data, which increases the influence of low quality data, potentially hiding biological information. "Good" filtering thresholds are highly data dependent and usually require trial and error. However, there are some general guidelines, which are discussed in the following.

.. figure:: image/filter_example.png
   :alt: An example of SC QC cell filtering. A violin plot per filter metric shows thresholds the distribution of data samples.

   A QC cell filter example showing a filter metric for each plot with thresholds. Low quality barcodes could be due to stressed cells, empty droplets, contamination, etc.

Given the figure above the most basic idea is to select your thresholds in a way that outliers, i.e. subpopulations of low quality, are removed that show up with different metrics.

**Technical metrics**

Technical metrics are metrics calculated from the data without the need of external information. The most common are *n_genes*, the number of genes detected for a barcode, and *total_counts*, the total number of reads per cell. They often come in multiple variations, e.g. on a log scale.

# TODO add recommendations

**Biological metrics**

Biological metrics relate to external information such as counting the mitochondria related reads per cell. The SC-Framework integrates mitochondrial, ribosomal, apoptosis and gender related gene identification and counting, with mitchondrial and ribosomal being the most commonly used.

The rule of thumb for these metrics is that high values mean the cell is stressed, which can mean it is low quality and should be filtered. However, individual thresholds are metric, dataset and sequencing method dependent. For example, cells with a mitochondrial reads above 5-10% are often filtered but this should be reconsidered if, e.g., the data contains muscle related cells (increased) or the data stems from single nucleus sequencing (decreased).

# TODO ribo recommendation; around 60%?
# TODO apoptosis? Not enough used?
# TODO gender?

Further references:

- `Single-cell best practices - filtering <https://www.sc-best-practices.org/preprocessing_visualization/quality_control.html#filtering-low-quality-cells>`_
- `Luecken et al. <https://doi.org/10.15252/msb.20188746>`_


Dimension reduction
^^^^^^^^^^^^^^^^^^^



Further references:

- `Single-cell best practices - dimension reduction <https://www.sc-best-practices.org/preprocessing_visualization/dimensionality_reduction.html>`_
- `UMAP <https://pair-code.github.io/understanding-umap/>`_
- `tSNE <https://distill.pub/2016/misread-tsne/?_ga=2.135835192.888864733.1531353600-1779571267.1531353600>`_


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
