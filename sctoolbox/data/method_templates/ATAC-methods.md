{#- ---------- Core Analysis ---------- -#}
The analysis was performed with the SC-Framework (Schultheis et al., doi: 10.5281/zenodo.11065517, version {{ args["sctoolbox"].version }}), utilizing the following tools as part of its integrated analysis environment. {##}
{#- ----- assembly notebook -#}
{%- if args["01_assembly"] -%}
    Mapped and quantified data was assembled by the SC-Framework, which created an initial dataset of {{ args["01_assembly"].var_count }} features and {{ args["01_assembly"].obs_count }} cells. {##}
{%- endif -%}
{#- ----- quality control notebook -#}
{%- if args["02_QC"] -%}
    {#- binarization -#}
    {%- if args["02_QC"].binarized -%}
        The dataset is binarized to correspond to open and closed chromatin states. {##}
    {%- endif -%}
    {#- Doublet Prediction -#}
    {%- if args["02_QC"].doublet -%}
        Scrublet doublet prediction (Wolock et al., doi: 10.1016/j.cels.2018.11.005) identified {{ args["02_QC"].doublet }} multiplet cells, subsequently removed from the dataset. {##}
    {%- endif -%}
    {#- cell filter -#}
    Low quality cells were filtered based on quality control metrics with{% if args["02_QC"].obs_global %} a global threshold over all samples{% else %} individual thresholds for each sample {% endif %}, which caused the removal of {{ args["02_QC"].cell }} cells. {##}
    {#- feature filter -#}
    {%- if args["02_QC"].mitochondrial or args["02_QC"].gender -%}
        Features were omitted if they belonged to {##}
        {#- mito filter -#}
        {%- if args["02_QC"].mitochondrial %}the mitochondria ({{ args["02_QC"].mitochondrial }} features){% if args["02_QC"].gender %}, {% else %}. {% endif %}{% endif %}
        {#- gono filter -#}
        {%- if args["02_QC"].gender %} or were gonosomal ({{ args["02_QC"].gender }} features). {% endif %}
    {%- endif -%}
    {#- ambient rna -#}
    {%- if args["02_QC"].ambient -%}
        The dataset was denoised by correcting for ambient RNA utilizing scAR (Sheng et al., doi: 10.1101/2022.01.14.476312). {##}
    {%- endif -%}
    {#- UROPA -#}
    {%- if args["02_QC"].UROPA -%}
        Features were annotated to genes using UROPA (Kondili et al., doi: 10.1038/s41598-017-02464-y). {##}
    {%- endif -%}
    {#- qc summary -#}
    {{ args["02_QC"].obs_count }} cells and {{ args["02_QC"].var_count }} features were left after quality control. {##}
{%- endif -%}
{#- ----- normalization and batch notebook -#}
{%- if args["03_batch_correction"] -%}
    {%- if args["03_batch_correction"]["norm"] == "total" -%}
        {#- normalize total+log1p -#}
        Variances introduced by sequencing depth were corrected through total count normalization and outliers were stabilized using the log1p. {##}
        {#- HVF+PCA -#}
        Principle Component Analysis (PCA) was conducted based on {{ args["03_batch_correction"].hvg }} highly variable features (HVFs). The resulting Principle Components (PCs) were filtered based on their correlation to quality metrics and explained variance. {{ args["03_batch_correction"].pc_count }} PCs were retained. {##}
    {%- elif args["03_batch_correction"]["norm"] == "tfidf" -%}
        {#- normalize tfidf+log1p -#}
        Variances introduced by sequencing depth were corrected through Term Frequency-Inverse Document Frequencies (TF-IDF) and outliers were stabilized using the log1p. {##}
        {#- HVF+LSI -#}
        Latent semantic indexing (LSI) was conducted based on {{ args["03_batch_correction"].hvg }} highly variable features (HVFs). The resulting components were filtered based on their correlation to quality metrics and explained variance. {{ args["03_batch_correction"].pc_count }} components were retained. {##}
    {%- endif -%}
    {#- num neighbors -#}
    A neighbor graph was constructed based on the selected components using {{ args["03_batch_correction"].neighbor_count}} neighbors. {##}
    {#- batch method -#}
    {%- if args["03_batch_correction"].batch != "uncorrected" -%}
        Potential batch effects between samples were corrected using {{ args["03_batch_correction"].batch }}. {##}
    {%- endif -%}
{%- endif -%}
{#- ----- clustering notebook -#}
{%- if args["04_clustering"] -%}
    {#- embedding+params -#}
    The data was transformed into a low-dimensional embedding using {{ args["04_clustering"].embedding }} ({{ args["04_clustering"].emb_cite }}, {{ args["04_clustering"].emb_params }}). {##}
    {#- clustering method + clust num + param -#}
    The clustering was done via the {{ args["04_clustering"].cluster_name }} algorithm ({{ args["04_clustering"].cluster_cite }}) with a resolution of {{ args["04_clustering"].resolution }}. The clustering was further refined to {{ args["04_clustering"].cluster_num }} final clusters. {##}
{%- endif %}
{# ---------- Downstream Analysis ---------- #}
{#- ----- group marker notebook -#}
{%- if args["group_markers"] -%}
    {#- group marker -#}
    Marker genes/features, i.e. genes/features predominantely expressed in one cluster/ group, were computed using Scanpy's "rank_genes_groups" (Wolf et al., doi: 10.1186/s13059-017-1382-0). The markers were further filtered to (1) be expressed in at least {{ args["group_markers"].min_perc }}% of cells within the target group, (2) have a fold change >{{ args["group_markers"].min_fc }}, and (3) be expressed in less than {{ args["group_markers"].max_perc }}% cells outside the target group. {##}
    {#- condition marker -#}
    Condition specific markers were identified by applying the same function and filter to each group split by condition. {##}
{%- endif -%}
{#- ----- cell type annotation notebook -#}
{%- if args["annotation"] -%}
    Cell groups were annotated using the SC-Frameworks MarkerRepo (https://gitlab.gwdg.de/loosolab/software/annotate_by_marker_and_features), a tool to identify cell types based on matches between marker genes and its internal database. {##} 
{%- endif -%}
{#- ----- GSEA -#}
{%- if args["GSEA"] -%}
    Gene set enrichment analysis (GSEA) was conducted using the marker genes on the GSEApy tool (Fang et al., doi: 10.1093/bioinformatics/btac757). {##}
{%- endif -%}
{#- ----- Proportion analysis notebook -#}
{%- if args["proportion_analysis"] -%}
    Proportional changes in cell numbers of {{ args["proportion_analysis"].cluster }} over {{ args["proportion_analysis"].condition }} were detected using Scanpro (Alayoubi et al., doi: 10.1038/s41598-024-66381-7). {##}
{%- endif -%}
{#- ----- Pseudotime trajectory notebook -#}
{%- if args["pseudotime_analysis"] -%}
    Trajectory (pseudotime) analysis was done through the scFates package (Faure et al., doi: 10.1093/bioinformatics/btac746).
{%- endif -%}
