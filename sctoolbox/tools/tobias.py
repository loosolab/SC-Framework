import yaml
import os
import sctoolbox.utils as utils


# from: https://github.com/yaml/pyyaml/issues/127#issuecomment-525800484
class _SpaceDumper(yaml.SafeDumper):
    # HACK: insert blank lines between top-level objects
    # inspired by https://stackoverflow.com/a/44284819/3786245
    def write_line_break(self, data=None):
        super().write_line_break(data)

        if len(self.indents) == 1:
            super().write_line_break()


def write_TOBIAS_config(out_path,
                        bams=[],
                        names=None,
                        fasta=None,
                        blacklist=None,
                        gtf=None,
                        motifs=None,
                        organism="human",
                        output="TOBIAS_output",
                        plot_comparison=True,
                        plot_correction=True,
                        plot_venn=True,
                        coverage=True,
                        wilson=True):
    """
    Write a TOBIAS config file from input bams/fasta/blacklist etc.

    Parameters
    ----------
    out_path : string
        Path to output yaml file.
    bams : list of strings, optional
        List of paths to bam files.
    names : list of strings, optional
        List of names for the bams. If None, the names are set to the bam file names with common prefix and suffix removed. Default: None.
    fasta : string, optional
        Path to fasta file. Default: None.
    blacklist : string, optional
        Path to blacklist file. Default: None.
    gtf : string, optional
        Path to gtf file. Default: None.
    motifs : string, optional
        Path to motif file. Default: None.
    organism : string, optional
        Organism name. TOBIAS supports 'human', 'mouse' or 'zebrafish'. Default: "human".
    output : string, optional
        Output directory of the TOBIAS run. Default: "TOBIAS_output".
    plot_comparison : boolean, optional
        Flag for the step of plotting comparison of the TOBIAS run. Default: True.
    plot_correction : boolean, optional
        Flag for the step of plotting correction of the TOBIAS run. Default: True.
    plot_venn : boolean, optional
        Flag for the step of plotting venn diagramms of the TOBIAS run. Default: True.
    coverage : boolean, optional
        Flag for coverage step of the TOBIAS run. Default: True.
    wilson: boolean, optional
        Flag for wilson step of the TOBIAS run. Default: True.
    """

    # Check organism input
    organism = organism.lower()
    valid_organisms = ["human", "mouse", "zebrafish"]
    if organism not in valid_organisms:
        raise ValueError(f"'{organism}' is not a valid organism. Valid organisms are: " + ", ".join(valid_organisms))

    # Remove any common prefix and suffix from names
    if names is None:
        prefix = os.path.commonprefix(bams)
        suffix = utils.longest_common_suffix(bams)
        names = [utils.remove_prefix(s, prefix) for s in bams]
        names = [utils.remove_suffix(s, suffix) for s in names]

    # Start building yaml
    data = {}
    data["data"] = {names[i]: bams[i] for i in range(len(bams))}
    data["run_info"] = {"organism": organism,
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
