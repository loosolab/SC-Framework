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

Filtering is a process of removing low quality data, e.g. cells, while keeping biological relevant information. This is often a trade-off. Stricter filters will remove more low quality data but have the potential to also remove more biological details, while lenient filters keep more data, which increases the influence of low quality data, potentially hiding biological information. "Good" filtering thresholds are highly data dependent and usually require trial and error. However, there are some general guidelines, which are discussed in the following.

.. figure:: image/filter_example.png
   :alt: An example of SC QC cell filtering. A violin plot per filter metric shows thresholds the distribution of data samples.

   A QC cell filter example showing a filter metric for each plot with thresholds. Low quality barcodes could be due to stressed cells, empty droplets, contamination, etc.

Given the figure above the most basic idea is to select your thresholds in a way that outliers, i.e. subpopulations of low quality, are removed that show up with different metrics.

**Technical metrics**

Technical metrics are metrics calculated from the data without the need of external information. The most common are *n_genes*, the number of genes detected for a barcode, and *total_counts*, the total number of reads per cell. They often come in multiple variations, e.g. on a log scale.

*n_genes* and *total_counts* inferring the coverage of the single cells. A lower minimum should be set to remove barcodes with a weak signal. Conversely, cells with exceptionally high values could be doublet cells and should also be removed. The exact values for setting the thresholds vary depending on the dataset, but lower limits of 100–1,000 are feasible.

**Biological metrics**

Biological metrics relate to external information such as counting the mitochondria related reads per cell. The SC-Framework integrates mitochondrial, ribosomal, apoptosis and gender related gene identification and counting, with mitchondrial and ribosomal being the most commonly used.

The rule of thumb for these metrics is that high values mean the cell is stressed, which can mean it is low quality and should be filtered. However, individual thresholds are metric, dataset and sequencing method dependent. For example, cells with a mitochondrial reads above 5-10% are often filtered but this should be reconsidered if, e.g., the data contains muscle related cells (increased) or the data stems from single nucleus sequencing (decreased).

Another important metric is ribosomal content. If the cells were healthy and the preparation process was successful, most of the ribosomal RNA should have been removed. If the RNA is degrading, the ribosomal content will rise. Therefore, filtering could also remove broken and dying cells. Based on our experience cells with more than 60% ribosomal content should be removed.

# TODO apoptosis? Not enough used?
# TODO gender?

Further references:

- `Single-cell best practices - filtering <https://www.sc-best-practices.org/preprocessing_visualization/quality_control.html#filtering-low-quality-cells>`_
- `Luecken et al. <https://doi.org/10.15252/msb.20188746>`_


Dimension reduction
^^^^^^^^^^^^^^^^^^^

**Highly variable genes**
The expression of some genes varies greatly between cells. These genes are considered highly informative with regard to the underlying biology. We therefore select these cells. To achieve this we ask the user for a a range (*min_limit*, *max_limit*) and call 'scanpy.pp.highly_variable_genes' in a loop varying *min_mean* until the number of HVGs is within the given range. The parameters *min_mean*, *max_mean*, *min_disp* and *max_disp* are forwarded to the scanpy function. For more details see: `Scanpy readthedocs <https://scanpy.readthedocs.io/en/stable/generated/scanpy.pp.highly_variable_genes.html#scanpy-pp-highly-variable-genes>`_.

**PCA + PC selection**
To reduce the high dimensionality, we employed Principal Component Analysis (PCA) (DOI: 10.1038/nmeth.4346). PCA projects high-dimensional data onto lower dimensions, known as principal components (PCs), while retaining trends and patterns. The final number of dimensions is set by the *n_pcs* parameter. Our default setting of 50 will most likely capture more variance than is necessary.
Next, we subset the PCs to remove batch effects and reduce noise. Therefore, we remove those PCs that are highly correlated with the metrics *corr_thresh* and those that explain very little variance *perc_thresh*.

**nearest neighbors**
Before calculating a 2D representation, a nearest neighbour graph of the cells is calculated using scanpy.pp.neighbors. We ask the user to provide the number of neighbours *n_neighbors*, its the size of local neighborhood used for manifold approximation. Small values capture local data, while large values will capture a more global view. Acceptable values are between 2 and 100. For more details see: `Scanpy readthedocs <https://scanpy.readthedocs.io/en/stable/api/generated/scanpy.pp.neighbors.html>`_.

**Embedding UMAP/TSNE**
2D/3D representations are achieved by a `UMAP <https://arxiv.org/abs/1802.03426>`_ or `t-SNE <https://www.jmlr.org/papers/volume9/vandermaaten08a/vandermaaten08a.pdf>`_ embedding on the neighbour graph. Here we are also using scanpy functions namely sc.tl.umap() and sc.tl.tsne(). Finding feasible parameters for the embedding can be quite challenging, therefore we have built wrapper functions that execute UMAP or t-SNE with user defined ranges of varying parameters. Here we ask to user to set either *dist_range* and *spread_range* to find UMAP parameters or *perplexity_range* and *learning_rate_range* for t-SNE parameters. For more details on the UMAP/t-SNE parameters see: `Scanpy readthedocs <https://scanpy.readthedocs.io/en/stable/generated/scanpy.tl.umap.html>`_.

Batch correction
^^^^^^^^^^^^^^^^

A selection of batch methods are available to perform the correction. The user is asked to set *batch_methods* to perform. Available are the following methods: 'bbknn', 'mnn', 'harmony', 'scanorama', 'combat'. Results can differ greatly and the user is asked to pick a correction to proceed. Beside annotated embeddings, the Local Inverse Simpson Index (LISI) facilitates the decision measuring the degree of mixing of the cells in regard to the batch column. We are using a implementation of `Harmony <https://www.nature.com/articles/s41592-019-0619-0#Sec11>`_.

Clustering
^^^^^^^^^^



Further references:

- `Single-cell best practices - dimension reduction <https://www.sc-best-practices.org/preprocessing_visualization/dimensionality_reduction.html>`_
- `UMAP <https://pair-code.github.io/understanding-umap/>`_
- `tSNE <https://distill.pub/2016/misread-tsne/?_ga=2.135835192.888864733.1531353600-1779571267.1531353600>`_


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
