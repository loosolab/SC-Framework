0.9.0 (in progress)
-------------------
- GSEA: Fix library gene-set overlap by converting all gene names to uppercase
- pl.gsea.term_dotplot: Fix example; Fix index==None bug
- added additional qc metrices for ATAC-seq to the first notebook (#256)

Changes to notebooks
^^^^^^^^^^^^^^^^^^^^
- added notebook versioning (#115)

0.8.0 (14-06-24)
----------------
- from_mtx: support more folder structures and variable file now optional (#234, #240)
- ligand-receptor: download_db added support for LIANA resources
- revised tsse scoring and fixed matplotlib version conflict (#257)
- add cyclone (pycirclize based plot) as hairball alternative (#223)
- remove legacy import structure
- implement lazy module loading 
- wrapped up native scrublet (#242, #150)
- prepare_for_cellxgene: Account for duplciate var indices
- added number of features to ATAC nb 3 and added combat as an available batch correct algorithm (#245)
- removed cleanup temp for the selfservice container (#258)

Changes to notebooks
^^^^^^^^^^^^^^^^^^^^
- rna/ atac more subset PC description
- rna/ atac clustering renamed "recluster" -> "revise cluster"
- Add GSEA notebook (#172)
- rna/atac assembly notebook update from_mtx (#234, #240)

0.7.0 (23-04-24)
----------------
- Added code examples for tools and utils (#140)
    - recluster 
    - group_heatmap
    - plot_venn
    - in_range
- Fix notebooks in readthedocs documentation (#220)
- Removed custom_marker_annotation script
- Disintegrated FLD scoring and added PEAKQC to setup.py (#233)
- fixed PCA-var plot not fitting into anndata_overview (#232)

Changes to notebooks
^^^^^^^^^^^^^^^^^^^^
- Overhaul RNA & ATAC notebooks structure (includes #207)
- Revise RNA notebook 4 recluster section (#201)

0.6.1 (28-03-24)
----------------
- Fix release pages by renaming the release-pages: job to pages:
- refactor move clean-orphaned-tags to new stage .post (#229)

0.6 (27-03-24)
--------------
- Fix unable to determine R_HOME error (#190)
- implemented propose_pcs to automatically select PCA components (#187)
- add correlation barplot to plot_pca_variance
- created correlation_matrix method by restructuring plot_pca_correlation
- Fix beartype issue with Lists and Iterables containing Literals (#227)
- CICD overhaul (#191)
- fixed notebook version in the env to 6.5.2 (#199, partly #44)

Changes to notebooks
^^^^^^^^^^^^^^^^^^^^
- Move proportion_analysis notebooks to general notebooks (#195 and #214)
- replace scanpy pseudotime with scFates in pseudotime_analysis notebook
- prepare_for_cellxgene: Adapt to new mampok verison 2.0.9
- prepare_for_cellxgene: Allows the user to set an analyst manually (#213)
- rna 03_batch revision (#209, #202, #200, #152)
- 05_marker_genes: Complete Overhaul (#181)

0.5 (04-03-24)
--------------

- add receptor_genes & ligand_genes parameters to connectionPlot and decreased runtime
- readme update(#188)
- Fix error when writing adata converted from an R object (#205, #180)
- Marker Repo integration (#162)
- Set scvelo version to >=0.3.1 (#193)
- Added fa2 as dependency for pseudotime analysis
- anndata_overview: fix issue where colorbars for continuous data was not shown
- added ability to use highly variable features using the lsi() function (#165)
- removed deprecated group_heatmap, umap_pub (replaced by gene_expression_heatmap, plot_embedding)
- add doku page
- start change log

Changes to notebooks
^^^^^^^^^^^^^^^^^^^^
- rna assembly: refactor
- prepare_for_cellxgene: Added BN_public as possible deployment cluster (#192)
- 14_velocity_analysis: Remove duplicate parameter (#194)
- pseudotime_analysis: Save generated plots (#211)
- rna 03_batch: added qc metrics to overview plot


0.4 (31-1-24)
-------------
- Fix get_rank_genes_tables for groups without marker genes (#179)
- Bugfixes for CI jobs
- Fix check_changes pipeline
- Fix typos (#173 & #174)
- Include kwargs in utils.bioutils._overlap_two_bedfiles(#177)
- Implemented _add_path() to automatically add python path to environment
- added tests for _add_path() and _overlap_two_bedfiles() (#177)
- constraint ipywidgets version to 7.7.5 to fix the quality_violinplot() (#151)(#143)
- Add temp_dir to calc_overlap_fc.py (#167) and revised related functions
- more testing (mainly sctoolbox.tools) (#166)
- gerneral text revisions

Changes to notebooks
^^^^^^^^^^^^^^^^^^^^
- Add pseudotime & velocity analysis notebooks (#164)
- Update receptor-ligand notebook (#176)
- Refactored annotate_genes() from ATAC-notebook 05 to 04 and removed 05 (#175)

0.3 (30-11-2023)
----------------
- Add parameter type hinting including runtime type checking (#46)
- Fixed prepare_for_cellxgene color issue (#145, #146)
- Add CI/CD container build pipeline for testing (#135)
- Fixed example for gene_expression_heatmap and smaller bugfixes related to marker genes (#124)
- Removed pl.group_heatmap as it is fully covered by pl.gene_expression_heatmap
- Removed 'sinto' as dependency and added code in 'create_fragment_file' to create fragment file internally (solves #147)
- The function 'create_fragment_file' was moved to bam tools.
- Added "n_genes" parameter to tools.marker_genes.get_rank_genes_tables, and set the default to 200 (#153)
- Fixed CI/CD build job rules. Only trigger build job when files changed or triggered manually
- Add parameter to plot_pca_correlation to plot correlation with UMAP components (#157)
- Handle NaN values for plot_pca_correlation (#156)
- implemented prepare_for_cellxgene
- Added pl.embedding.plot_embedding() function to plot embeddings with different styles, e.g. hexbin and density (#149)
- Modified pl.embedding.plot_embedding() to plot different embedding dimensions
- Deprecated pl.umap_pub as this is now covered by pl.plot_embedding
- changed typing to beartype.typing
- Added GenomeTracks plotting
- Fix batch evaluation for small datasets (#148)
- Added **kwargs to functions which are wrappers for other functions
- added RAGI cluster validation to clustering.py (!201)
- started disintegrating fld scoring (!201)
- reorganised ATAC-notebooks (!201)

Changes to notebooks
^^^^^^^^^^^^^^^^^^^^
- Added prepare for cellxgene notebook (#139)
- Added plot of highly expressed genes to RNA notebook 03 (#43)
- Changed structure of notebooks in directory; added "notebooks" subdirectories for RNA and ATAC


0.2 (30-08-2023)
----------------
- fix error in prepare_for_cellxgene caused by .uns[_color] not matching .obs column. (#176)
- implemented prepare_for_cellxgene (#147)
- fixed raw value copy issue in rna/02-batch notebook
- Added parameters for the TOBIAS flags in the config file to write_TOBIAS_config()
- Added logging verbose and decorator to ATAC related functions
- Fix "shell not found" error for CI pipeline (#129)
- Pinned scikit-learn to version <=1.2.2 (#128)
- Added script for gene correlation and comparison between two conditions
- Added check for marker gene lists (#103)
- Keep notebook metadata on push to prevent deleting kernel information
- Added sctoolbox as default kernel to RNA & ATAC notebooks
- Added check of column validity to tools.marker_genes.run_DESeq2() (#134)
- Increase test coverage for plotting functions (#126)
- Apply fixes to bugs found by increasing the test coverage.
- Added type hinting to functions.
- Revised doc-strings.
- run_rank_genes() auto converts groupby column to type 'category' (#137)
- Fix parameter for gene/cell filtering (#136)
- Add Check to _filter_object() if column contains only boolean (#110)
- Add support of matrx and numpy.ndarray type of adata.X for predict_sex (#111)
- Add method to get pd.DataFrame columns with list of regex (#90)
- Added 'pairwise_scatter' method for plotting QC metrics (#54)
- Add ATAC quality metrics TSSe (ENCODE), FRiP
- Revised FLD density plotting
- Adjusted style of default values in docs (#33)
- Added 'plot_pca_correlation' for plotting PCA correlation with obs/var columns (#118)
- Removed outdated normalization methods.
- Changed all line endings to LF (#138)
- Disabled threads parameter for tSNE (#130)
- Added 'plot_starsolo_quality' and 'plot_starsolo_UMI' to plotting module (#78)
- Fixed issues with clustered dotplot with new code (#122)

Changes to RNA notebooks
^^^^^^^^^^^^^^^^^^^^^^^^
- Added display of 3D UMAP html in notebook 04 (#119)

Changes to ATAC notebooks
^^^^^^^^^^^^^^^^^^^^^^^^^
- Fixed assembling atac notebook 01
- Fixed get_atac_thresholds_wrapper and renamed it to get_thresholds_wrapper
- Added custome cwt implementation
- Added additional parameters to add_insertsize_metrics
- Revised nucleosomal score scoring

0.1.1 (24-05-2023)
------------------
- Fixed import issue
- Make version accessible
- Added check for CHANGES.rst in gitlab-ci
- Pinned numba==0.57.0rc1 due to import error (#117)
- Fixed bug in tools.norm_correct.atac_norm
- Added check for sctoolbox/_version.py file in gitlab-ci

0.1 (22-05-2023)
----------------
- First version
