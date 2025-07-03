"""Tools for TOBIAS usage."""
import yaml
import os
import glob

import scanpy as sc
import sctoolbox.utils as utils
import sctoolbox.tools as tools

from beartype import beartype
from beartype.typing import Optional, Literal


# from: https://github.com/yaml/pyyaml/issues/127#issuecomment-525800484
class _SpaceDumper(yaml.SafeDumper):
    # HACK: insert blank lines between top-level objects
    # inspired by https://stackoverflow.com/a/44284819/3786245
    def write_line_break(self, data=None):
        super().write_line_break(data)

        if len(self.indents) == 1:
            super().write_line_break()


@beartype
def write_TOBIAS_config(out_path: str,
                        bams: list[str] = [],
                        names: Optional[list[str]] = None,
                        fasta: Optional[str] = None,
                        blacklist: Optional[str] = None,
                        gtf: Optional[str] = None,
                        motifs: Optional[str] = None,
                        organism: Literal["human", "mouse", "zebrafish"] = "human",
                        output: str = "TOBIAS_output",
                        plot_comparison: bool = True,
                        plot_correction: bool = True,
                        plot_venn: bool = True,
                        coverage: bool = True,
                        wilson: bool = True) -> None:
    """
    Write a TOBIAS config file from input bams/fasta/blacklist etc.

    Parameters
    ----------
    out_path : str
        Path to output yaml file.
    bams : list[str], default []
        List of paths to bam files.
    names : Optional[list[str]], default None
        List of names for the bams. If None, the names are set to the bam file names with common prefix and suffix removed.
    fasta : Optional[str], default None
        Path to fasta file.
    blacklist : Optional[str], default None
        Path to blacklist file.
    gtf : Optional[str], default None
        Path to gtf file.
    motifs : Optional[str], default None
        Path to motif file.
    organism : Literal["human", "mouse", "zebrafish"], default 'human'
        Organism name. TOBIAS supports 'human', 'mouse' or 'zebrafish'.
    output : str, default 'Tobias_output'
        Output directory of the TOBIAS run.
    plot_comparison : bool, default True
        Flag for the step of plotting comparison of the TOBIAS run.
    plot_correction : bool, default True
        Flag for the step of plotting correction of the TOBIAS run.
    plot_venn : bool, default True
        Flag for the step of plotting venn diagramms of the TOBIAS run.
    coverage : bool, default True
        Flag for coverage step of the TOBIAS run.
    wilson : bool, default True
        Flag for wilson step of the TOBIAS run.
    """

    # Remove any common prefix and suffix from names
    if names is None:
        prefix = os.path.commonprefix(bams)
        suffix = utils.general.longest_common_suffix(bams)
        names = [utils.general.remove_prefix(s, prefix) for s in bams]
        names = [utils.general.remove_suffix(s, suffix) for s in names]

    # Start building yaml
    data = {}
    data["data"] = {names[i]: bams[i] for i in range(len(bams))}
    data["run_info"] = {"organism": organism.lower(),
                        "blacklist": blacklist,
                        "fasta": fasta,
                        "gtf": gtf,
                        "motifs": motifs,
                        "output": output}

    # Flags for parts of pipeline to include/exclude
    data["flags"] = {"plot_comparison": plot_comparison,
                     "plot_correction": plot_correction,
                     "plot_venn": plot_venn,
                     "coverage": coverage,
                     "wilson": wilson}

    # Default module parameters
    data["macs"] = "--nomodel --shift -100 --extsize 200 --broad"
    data["atacorrect"] = ""
    data["footprinting"] = ""
    data["bindetect"] = ""

    # Write dict to yaml file
    with open(out_path, 'w') as f:
        yaml.dump(data, f, Dumper=_SpaceDumper, default_flow_style=False, sort_keys=False)

    print(f"Wrote TOBIAS config yaml to '{out_path}'")

@beartype
def prepare_tobias(adata: sc.AnnData,
                   groupby: str,
                   output: str,
                   path_bam: str,
                   barcode_column: Optional[str] = None,
                   barcode_tag: str = 'CB',
                   fasta: Optional[str] = None,
                   motifs: Optional[str] = None,
                   gtf: Optional[str] = None,
                   blacklist: Optional[str] = None,
                   organism: str = 'human',
                   yml: str = "TOBIAS_config.yml",
                   plot_comparison: bool = True,
                   plot_correction: bool = True,
                   plot_venn: bool = True,
                   coverage: bool = False,
                   wilson: bool = False,
                   threads: int = 4):
    """
    Split ATAC-seq bamfile by adata.obs column and prepare TOBIAS run.

    Parameters
    ----------
    adata : AnnData
        Annotated data matrix.
    groupby : str
        Column in adata.obs to create the pseudobam files based on.
    output : str
        Path to save the output files.
    path_bam : str
        Path
    barcode_column : str
        Column in adata.obs that contains the barcode information. If None adata.obs.index is used.
    barcode_tag : str
        Tag to extract the barcode from the read name.
    fasta : str
        Path to the organism fasta file.
    motifs : str
        Path to the motifs file or directory.
    gtf : str
        Path to the organisms gtf file (genes).
    blacklist : str
        Path to the blacklist file.
    organism : str
        Organism. options = ["mouse", "human", "zebrafish"]
    yml : str
        Name of TOBIAS config yaml. Cannot be named "config.yml" or it will be overwritten when running TOBIAS.
    plot_comparison : bool
        TOBIAS flag for plotting comparison between condition.
    plot_correction : bool
        TOBIAS flag for plotting correction.
    plot_venn : bool
        TOBIAS flag for plotting venn diagrams.
    coverage : bool
        TOBIAS flag for coverage calculation.
    wilson : bool
        TOBIAS flag for wilson calculation.
    threads : int
        Number of threads to use.
    """
    # Check if directory for TOBIAS run exists, if not create it
    if os.path.exists(output):
        print(
            f"WARNING: The directory \'{output}\' already exists. Any files in this directory may be overwritten, which can cause inconsistencies.")
    else:
        utils.io.create_dir(output)

    # Get path for TOBIAS input and create directory
    path_TOBIAS_in = os.path.abspath(os.path.join(output, "input", ""))

    if not os.path.exists(path_TOBIAS_in):
        utils.io.create_dir(path_TOBIAS_in)

    # Get path for TOBIAS output and create directory
    path_TOBIAS_out = os.path.abspath(os.path.join(output, "output"))

    if not os.path.exists(path_TOBIAS_out):
        utils.io.create_dir(path_TOBIAS_out)

    # check if the correct barcode tag is used
    tools.bam.check_barcode_tag(adata, path_bam, barcode_tag)

    # Prepare splitting of the bam file if necessary
    if groupby not in adata.obs.columns:
        raise ValueError(f"Column \'{groupby}\' not found in adata.obs. Please give valid input.")

    if len(adata.obs[groupby].unique()) == 1:
        split_bam = False
    else:
        split_bam = True

    # Split bam file by cluster
    if split_bam:

        if threads > 1:
            parallel = True
        else:
            parallel = False

        tools.bam.split_bam_clusters(adata,
                               bams=path_bam,
                               groupby=groupby,
                               barcode_col=barcode_column,
                               read_tag=barcode_tag,
                               output_prefix=path_TOBIAS_in,
                               reader_threads=threads,
                               writer_threads=threads,
                               parallel=parallel,
                               pysam_threads=threads)

    # Handle single condition
    else:
        bams = os.path.abspath(path_bam)

    # Get paths to bam files for TOBIAS run
    bams = glob.glob("".join([path_TOBIAS_in, "*.bam"]))
    bams = [os.path.abspath(f) for f in bams]

    # If there is no blacklist file given, create a mock blacklist file
    if blacklist is None:
        blacklist = os.path.join(path_TOBIAS_out, "blacklist.bed")
        f = open(blacklist, "w")
        f.write("chr1\t0\t1\n")
        f.close()

    # Make paths absolute
    if fasta is not None:
        fasta = os.path.abspath(fasta)
    if motifs is not None:
        motifs = os.path.abspath(motifs)
    if gtf is not None:
        gtf = os.path.abspath(gtf)
    if blacklist is not None:
        blacklist = os.path.abspath(blacklist)

    print("Writing TOBIAS config yaml.")

    # Call function to write TOBIAS config yml
    write_TOBIAS_config(os.path.join(path_TOBIAS_in, yml),
                        bams=bams,
                        fasta=fasta,
                        gtf=gtf,
                        motifs=motifs,
                        blacklist=blacklist,
                        organism=organism,
                        output=path_TOBIAS_out,
                        plot_comparison=plot_comparison,
                        plot_correction=plot_correction,
                        plot_venn=plot_venn,
                        coverage=coverage,
                        wilson=wilson)

    return path_TOBIAS_in, path_TOBIAS_out, yml
