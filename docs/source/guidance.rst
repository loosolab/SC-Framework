Guidance & Recommendations
==========================

Here we focus on key steps in the analysis to give a better understanding of how parameter choices influence results. Also check the :doc:`/tutorials/index`. The following should be viewed as suggestions and recommendations that worked for us and others (see the added references); they should not be treated as definitive rules and may change in the future.

RNA
---


Doublet detection
^^^^^^^^^^^^^^^^^

Doublets or multiplets are artifacts that are created when two or more cells are sequenced with the same barcode. They can appear as separate cell populations in the embedding step and should be removed. We utilize `Scrublet <https://github.com/swolock/scrublet>`_ for doublet detection and follow their recommendations. The stringency of whether a cell classifies as a doublet may be controlled through the ``doublet_threshold`` parameter. On default, this is automatically set via Scrublet, however, manual intervention is possible. The ``doublet_threshold`` applies to the ``doublet_score`` a score that ranges from 0-1. The score may be interpreted as the chance of a cell being a doublet, where higher values mean *more likely*.

.. figure:: image/scrublet_threshold.png
   :alt: Scrublet doublet score distribution histograms.

   Histograms of barcodes distributed by doublet score. The black vertical line indicates the doublet threshold. Lower values (left of the line) are considered singlets, while higher values (right of the line) are treated as doublets. Image taken from `Scrublet <https://github.com/swolock/scrublet>`__.

The barcodes are usually distributed bimodally, where the first (left) modality represents singlets and the second (right) modality represents doublets. The ``doublet_threshold`` should be placed in the valley between the two modalities, as scores higher than the threshold are considered doublets.

Further references:

- `Single-cell best practices - doublet <https://www.sc-best-practices.org/preprocessing_visualization/quality_control.html#doublet-detection>`_
- `Wolock et al. <https://doi.org/10.1016/j.cels.2018.11.005>`_

Filter metrics
^^^^^^^^^^^^^^

Here we look at metrics commonly used in QC filtering. The SC-Framework has two methods to automatically set upper and lower thresholds for each metric, which are explained below. The automatic thresholds can be adjusted by the user as necessary.

Automatic thresholds
""""""""""""""""""""

The SC-Framework has two methods to automatically set filter thresholds, called *MAD* and *GMM*. The *MAD* method, short for median absolute deviation, is calculated as


.. math::
   :label: mad_formula

   MAD=median(|X - median(X)|)

   MAD_{score}=median(X) \pm MAD \cdot n_{1, 2}

where :math:`X` is the metric to find a threshold for and :math:`n_{1, 2}` are multipliers for upper and lower bounds. The *GMM* thresholds are calculated similar with the key difference of calculating the thresholds on the largest gaussian mixture within the data. The *GMM* method is based on *MAD* and was developed as part of SC-Framework.

.. math::
   :label: gmm_formula

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

*n_genes* and *total_counts* reflect the coverage of individual cells. A lower minimum should be set to remove barcodes with a weak signal. Conversely, cells with exceptionally high values could be doublets and should also be removed. The exact threshold values vary by dataset, but lower limits of 100–1,000 are feasible.

**Biological metrics**

Biological metrics relate to external information, such as counting mitochondria-related reads per cell. SC-Framework integrates identification and counting of mitochondrial, ribosomal, apoptosis-related, and gender-related genes, with mitochondrial and ribosomal content being the most commonly used metrics.

As a rule of thumb, high values indicate cellular stress, which is associated with low quality and warrants filtering. Individual thresholds are metric-, dataset-, and sequencing-method-dependent. For example, cells with mitochondrial reads above 5–10% are often filtered, but this should be reconsidered if the data contain muscle-related cells (where mitochondrial content is elevated) or originate from single-nucleus sequencing (where it is decreased).

Ribosomal content is another important metric. In a healthy preparation, most ribosomal RNA should have been removed; if RNA is degrading, ribosomal content rises. Filtering on this metric therefore helps remove broken and dying cells. Based on our experience, cells with more than 60% ribosomal content should be removed.

Further references:

- `Single-cell best practices - filtering <https://www.sc-best-practices.org/preprocessing_visualization/quality_control.html#filtering-low-quality-cells>`_
- `Luecken et al. <https://doi.org/10.15252/msb.20188746>`_


Dimension reduction
^^^^^^^^^^^^^^^^^^^

**Highly variable genes**

The expression of some genes varies greatly between cells. These genes are considered highly informative with respect to the underlying biology and are therefore selected before dimensionality reduction. To identify them, the user provides a target range (*min_limit*, *max_limit*), and SC-Framework calls ``scanpy.pp.highly_variable_genes`` in a loop, varying *min_mean* until the number of highly variable genes (HVGs) falls within that range. The parameters *min_mean*, *max_mean*, *min_disp*, and *max_disp* are forwarded to the Scanpy function. For further details see the `Scanpy documentation <https://scanpy.readthedocs.io/en/stable/generated/scanpy.pp.highly_variable_genes.html#scanpy-pp-highly-variable-genes>`_.

**PCA and PC selection**

To reduce the high dimensionality of the data, SC-Framework applies Principal Component Analysis (PCA) (`Becht et al., 2019 <https://doi.org/10.1038/nmeth.4346>`_). PCA projects high-dimensional data onto lower-dimensional principal components (PCs) while retaining the dominant trends and patterns. The final number of PCs is set via the *n_pcs* parameter; the default of 50 typically captures more variance than is necessary.

PCs are subsequently filtered to reduce noise and remove batch-driven variation. PCs that are highly correlated with technical covariates (controlled by *corr_thresh*) and PCs that explain very little variance (controlled by *perc_thresh*) are discarded.

.. figure:: image/pc_subset.png
   :alt: principal component subsetting to remove unwanted batch effects.
   :width: 60%

**Nearest neighbours**

Before computing a 2D representation, a nearest-neighbour graph of cells is constructed using ``scanpy.pp.neighbors``. The *n_neighbors* parameter sets the size of the local neighbourhood used for manifold approximation. Small values emphasise local structure, while large values capture more global relationships. Acceptable values are between 2 and 100. For further details see the `Scanpy documentation <https://scanpy.readthedocs.io/en/stable/api/generated/scanpy.pp.neighbors.html>`_.

**Embedding (UMAP / t-SNE)**

2D and 3D representations are produced by `UMAP <https://arxiv.org/abs/1802.03426>`_ or `t-SNE <https://www.jmlr.org/papers/volume9/vandermaaten08a/vandermaaten08a.pdf>`_ applied to the neighbour graph via ``sc.tl.umap()`` and ``sc.tl.tsne()``. Because finding suitable embedding parameters can be challenging, SC-Framework provides wrapper functions that execute UMAP or t-SNE over user-defined parameter ranges. For UMAP the user sets *dist_range* and *spread_range*; for t-SNE the user sets *perplexity_range* and *learning_rate_range*. Ranges follow the format ``(minimum, maximum, step)``. The wrapper returns plots for all parameter combinations, allowing the user to select the most appropriate embedding. For further details on the underlying parameters see the `Scanpy documentation <https://scanpy.readthedocs.io/en/stable/generated/scanpy.tl.umap.html>`_.

.. figure:: image/umap_parameters.png
   :alt: Output of search_umap_parameters().
   :width: 60%

Batch correction
^^^^^^^^^^^^^^^^

.. figure:: image/batchcorrection-intro.png
   :alt: https://www.10xgenomics.com/analysis-guides/introduction-batch-effect-correction.
   :width: 60%

*https://www.10xgenomics.com/analysis-guides/introduction-batch-effect-correction*

SC-Framework supports several batch correction methods. The user selects which to run via *batch_methods*; available options are ``bbknn``, ``mnn``, ``harmony``, ``scanorama``, and ``combat``. Because results can differ considerably between methods, the user reviews the corrected embeddings and selects the most appropriate correction. In addition to annotated embeddings, the Local Inverse Simpson Index (LISI) aids the decision by quantifying the degree of cell mixing with respect to the batch variable. LISI scores are computed using the `Harmony <https://www.nature.com/articles/s41592-019-0619-0#Sec11>`_ implementation.

.. figure:: image/batch_corr_framework.png
   :alt: anndata_overview() after running batch corrections.
   :width: 60%

Clustering
^^^^^^^^^^

SC-Framework performs clustering with the Leiden algorithm via ``sc.tl.leiden()``. As with the embedding step, a wrapper function executes clustering across a range of resolution values defined by *cluster_res_range*, following the same ``(minimum, maximum, step)`` format. All resulting clusterings are plotted so the user can select the resolution that best separates the expected cell populations. The number of columns in the overview plot is controlled by *cluster_ncols*. For a detailed description of the Leiden algorithm see `Traag et al., 2019 <https://www.nature.com/articles/s41598-019-41695-z>`_.

Marker Genes
^^^^^^^^^^^^

Identifying marker genes is an important step before proceeding with downstream analyses. This group comparison — often referred to as differential expression analysis — is supported in two ways: a Scanpy-based approach for simple comparisons, and a pyDESeq2 workflow for more complex analyses when replicates are available.

For the Scanpy-based approach, SC-Framework provides a wrapper function that first calls ``sc.tl.rank_genes_groups()`` to populate the rank-gene tables in the AnnData object, then filters the result with ``sc.tl.filter_rank_genes_groups()``. The user provides the following parameters in the notebook:

- *marker_labels* — the column name in ``adata.var`` used for ranking.
- The ranking method, forwarded to ``sc.tl.rank_genes_groups()``.
- *top_n* — how many genes to rank per group, forwarded to ``sc.tl.rank_genes_groups()``.
- *min_in_group_fraction* — the minimum fraction of cells in the target group that must express the gene for it to be considered a valid marker.
- *min_fold_change* — the minimum expression difference across groups required to call a gene a marker.
- *max_out_group_fraction* — the maximum fraction of cells in other groups that may express the gene.

For further details on the individual parameters see the `Scanpy documentation <https://scanpy.readthedocs.io/en/stable/generated/scanpy.tl.rank_genes_groups.html>`_.

For the pyDESeq2 workflow, the user provides *sample_col* (batch or replicate information), *condition_col* (the variable to compare), and *layer_raw* (the layer to use as input). The data are normalised with ``sc.pp.normalize_total()`` before DESeq2 is called. SC-Framework wraps pyDESeq2 and handles the user-supplied parameters internally.

ATAC
----

Barcode Filtering
^^^^^^^^^^^^^^^^^

Barcode filtering is applied to scATAC-seq data in the same way as for scRNA-seq data. Most metrics used for RNA-seq — such as total read counts or feature counts — apply equally to ATAC-seq. In addition, there are ATAC-seq-specific QC metrics, which are described below.

**FLD Score**

The fragment length distribution (FLD) is a common quality proxy in ATAC-seq data. The FLD score is calculated by `PEAKQC <https://academic.oup.com/bib/article/26/5/bbaf465/8255857>`_ and positively correlates with a more pronounced nucleosomal banding pattern in the fragment length distribution, reflecting DNA integrity and appropriate Tn5 insertion ratios. Cells with low FLD scores can be excluded. The minimum threshold is dataset- and analysis-dependent; removing clear outliers is a practical starting point.

.. figure:: image/peakqc_fig2.png
   :alt: FLD score correlating with nucleosomal banding.
   :width: 60%

*https://academic.oup.com/bib/article/26/5/bbaf465/8255857*

**Overlap**

The fraction of fragments overlapping defined features, such as promoter regions, is another quality proxy for ATAC-seq. An enrichment of fragments in promoter regions is expected in accessible chromatin data, so this metric reflects the signal-to-noise ratio. Cells with low promoter-overlap fractions should be excluded. From our experience, 0.3 is a reasonable lower threshold.

**Transcription Start Site Enrichment (TSSe)**

Transcription Start Site enrichment (`TSSe <https://www.encodeproject.org/atac-seq/>`_) is conceptually similar to the promoter-overlap metric but is bias-corrected for the expected enrichment in regions flanking the transcription start site. As a starting point, cells with very low TSSe scores can be removed as outliers.

.. figure:: image/tsse.png
   :alt: Transcription Start Site enrichment.
   :width: 30%

*https://academic.oup.com/bib/article/26/5/bbaf465/8255857*

**Fraction of Reads in Peaks (FRiP)**

The Fraction of Reads in Peaks (`FRiP <https://www.encodeproject.org/atac-seq/>`_) is a widely used metric for assessing signal-to-noise ratio. In accessible chromatin data, reads are expected to accumulate in open chromatin regions rather than occurring randomly across the genome. From our experience, filtering cells with a FRiP below 0.1 is an effective but lenient starting threshold.


Doublet detection (ATAC)
^^^^^^^^^^^^^^^^^^^^^^^^

Doublet removal in scATAC-seq exploits a property specific to DNA-based assays: because ATAC-seq reads genomic DNA, each locus should appear exactly twice — once per allele. Cells with unexpectedly high locus overlap are therefore likely doublets. `AMULET <https://ucarlab.org/wp-content/uploads/2024/11/AMULET-a-novel-read-count-based.pdf>`_ uses this principle to detect multiplets; SC-Framework integrates the core AMULET algorithm directly. The user provides the following parameters:

- *amulet_q_threshold* — q-value cutoff (FDR-corrected p-value) for calling a barcode a significant doublet.
- *amulet_repeat_filter* — optional blacklist of repetitive genomic features to exclude, as these may produce artifactual multi-mapping overlaps.
- *amulet_expected_overlap* — expected number of overlapping reads per locus under the null (should be 2, reflecting the diploid genome).
- *amulet_max_insert_size* — maximum fragment insert size included in the overlap count.
- *amulet_min_overlap* — minimum overlap count per region required for a region to be included in the statistical test.

Normalization (ATAC)
^^^^^^^^^^^^^^^^^^^^

The normalization method is selected via *norm_method* and can be either TF-IDF or total count normalization. `TF-IDF <https://www.nature.com/articles/nature25981>`_ (term frequency-inverse document frequency), originally developed for text retrieval, scores each feature (here, an open chromatin region) by its importance within a cell relative to its prevalence across all cells. This highlights cell-defining accessible regions. Total count normalization, by contrast, rescales each cell so that all cells share the same total count after normalization — the same approach commonly applied to scRNA-seq data.

Further references:

- `Single-cell best practices - dimension reduction <https://www.sc-best-practices.org/preprocessing_visualization/dimensionality_reduction.html>`_
- `UMAP <https://pair-code.github.io/understanding-umap/>`_
- `tSNE <https://distill.pub/2016/misread-tsne/?_ga=2.135835192.888864733.1531353600-1779571267.1531353600>`_


Multi-omics
-----------

The multi-omics workflow integrates two independently analyzed modalities into a joint embedding using MOFA+. This allows you to perform clustering on the integrated latent space, compare cluster solutions across modalities, and ultimately prepare a combined dataset for visualization in CELLxGENE.

.. figure:: image/multiomics-flowchart.png
   :alt: Flowchart for multiomics analysis using the SC-Framework
   :width: 60%

Step 1: Analyze Both Modalities Individually
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Before integrating your data, each modality must be processed independently using the standard single-modality analysis pipeline by running notebooks 1 to 4.
To continue with the analysis, each AnnData object is required to have one clustering column in obs and at least one two-dimensional embedding (e.g., UMAP)

Step 2: Multi-omics Analysis
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The main multi-omics analysis is done in the multiomics notebook. The process consists of three sub-steps:

**2a. Comparing Clusterings from Individual Analyses**
The clusterings obtained from the individual modality analyses are compared with one another. This is done by examining the clusters and embeddings from both modalities side by side, enabling you to assess visually and quantitatively how well the two modalities align at the level of individual cell populations.
In addition to the visual comparison, a Cluster Comparison Score is calculated for each pair of clusters. This score ranges from 0 to 1; a higher score indicates greater overlap between the clusters being compared. Therefore, a score close to 1 suggests that the corresponding clusters from Modality A and Modality B capture a very similar set of cells, while a score close to 0 indicates little to no overlap.
To further support the interpretation of these results, a set of additional plots is generated alongside the score matrix. These plots provide alternative views of the cluster relationships and help you to understand the similarities and differences between the two modalities' clustering solutions.

**2b/c. Data integration and clustering**
Both AnnData objects are converted into a single MuData object.
This step uses MOFA+ to learn a joint latent space from both modalities. MOFA+ takes as input the two modalities and learns a joint latent space, from which an integrated embedding (UMAP) is calculated. This embedding captures the shared and complementary structures of both modalities.
The Leiden algorithm is then used to assign each cell to an integrated cluster based on this embedding.

**2d. Data Export**
For the data export, the integrated embedding and clustering are transferred to the individual AnnData objects.
These individual objects can then be used for further downstream analyses.
Additionally, to the individual AnnData, the combined MuData object is exported.

Step 3: Downstream Analysis
^^^^^^^^^^^^^^^^^^^^^^^^^^^
For downstream analysis, each AnnData object with an integrated embedding and clustering can be analysed using the provided downstream notebooks (e.g., GSEA, ligand-receptor analysis, etc.).
Alternatively, additional analyses can be performed on the combined MuData object.

Step 4: Prepare for interactive visualization
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Once both AnnData objects have been finalised, they can be merged into a single AnnData object supported by CELLxGENE via the CELLxGENE Preparation Notebook (general_notebooks/prepare_for_cellxgene.ipynb)
