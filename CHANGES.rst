0.3 (in progress)
=================

- Added pl.embedding.embedding() function to plot embeddings with different styles, e.g. hexbin and density
- Deprecated pl.umap_pub as this is now covered by pl.embedding

Changes to notebooks:

- Added plot of highly expressed genes to RNA notebook 03 (#43)


0.2 (in progress)
=================
- Add paramter to plot_pca_correlation to plot correlation with UMAP components (#157)
- Handle NaN values for plot_pca_correlation (#156)
- implemented prepare_for_cellxgene
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
- Add CI/CD container build pipeline for testing (#135)
- Disabled threads parameter for tSNE (#130)
- Added 'plot_starsolo_quality' and 'plot_starsolo_UMI' to plotting module (#78)
- Fixed issues with clustered dotplot with new code (#122)
- Add parameter type hinting including runtime type checking (#46)
- Removed 'sinto' as dependency and added code in 'create_fragment_file' to create fragment file internally (solves #147)
- The function 'create_fragment_file' was moved to bam tools.
- Added "n_genes" parameter to tools.marker_genes.get_rank_genes_tables, and set the default to 200 (#153)
- Fixed CI/CD build job rules. Only trigger build job when files changed or triggered manually

Changes to notebooks
--------------------
- Added display of 3D UMAP html in notebook 04 (#119)

ATAC notebooks
^^^^^^^^^^^^^^
- Fixed assembling atac notebook 01
- Fixed get_atac_thresholds_wrapper and renamed it to get_thresholds_wrapper
- Added custome cwt implementation
- Added additional parameters to add_insertsize_metrics
- Revised nucleosomal score scoring

0.1.1 (24-05-2023)
==================
- Fixed import issue
- Make version accessible
- Added check for CHANGES.rst in gitlab-ci
- Pinned numba==0.57.0rc1 due to import error (#117)
- Fixed bug in tools.norm_correct.atac_norm
- Added check for sctoolbox/_version.py file in gitlab-ci

0.1 (22-05-2023)
================
- First version
