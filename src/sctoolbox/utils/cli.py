"""Command line utility to download the analysis notebooks and put them into the right structure."""

import argparse
import sys
from sctoolbox.utils.creators import add_analysis
from sctoolbox._version import __version__


def cli_args() -> argparse.ArgumentParser:
    """
    Arguments for the CLI.

    Returns
    -------
    Namespace
        The parsed command line arguments
    """
    p = argparse.ArgumentParser(description="Download and prepare analysis notebooks from the SC-Framework repository (https://github.com/loosolab/SC-Framework).")

    p.add_argument("-p", "--path", help="Path to the output directory.", required=True)
    p.add_argument("-n", "--name", help="The name of the analysis. Will create a directory with this name in the given 'path'.", required=True)
    p.add_argument("-m", "--method", help="The method matching your data type.", choices=["rna", "atac"], required=True)
    p.add_argument("-g", "--exclude_general", help="Whether to download the general notebooks.", action="store_false")
    p.add_argument("-t", "--token", help="A GitHub access token. Useful to circumvent API throttling.")
    p.add_argument("-r", "--reference", help="Download notebooks of a specific version. Either a branch name, version tag or commit SHA. Will download the latest version from main on default.")
    p.add_argument("-v", "--version", help="Show the SC-Framework version.", action="version", version=f"SC-Framework: {__version__}")

    return p


def main() -> None:
    """CLI for ``sctoolbox.utils.creators.add_analysis``."""
    # create the CLI argument parser
    parser = cli_args()

    # print the help message if no arguments are supplied
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit()

    # parse the arguments
    args = parser.parse_args()

    # run add_analysis
    add_analysis(
        dest=args.path,
        analysis_name=args.name,
        method=args.method,
        general=args.exclude_general,
        access_token=args.token,
        reference=args.reference
    )
