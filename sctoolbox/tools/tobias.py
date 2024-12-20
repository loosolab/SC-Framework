"""Tools for TOBIAS usage."""
import yaml
import os
import sctoolbox.utils as utils

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
