"""Jupyter notebook related functions."""

import os
from IPython.core.magic import register_line_magic
from IPython.display import HTML, display
import sctoolbox.utils as utils

from beartype.typing import Optional


def _is_notebook() -> bool:
    """
    Check if function is run within a notebook.

    Returns
    -------
    bool
        True if running from a notebook, False otherwise.
    """

    try:
        _ = get_ipython()
        return True
    except NameError:
        return False


if _is_notebook():
    @register_line_magic
    def bgcolor(color: str, select: Optional[list[int]] = None) -> None:
        """
        Set background color of current jupyter cell or the selected cells.

        Adapted from https://stackoverflow.com/a/68902884.
        Note: Jupyter notebook v6+ needed

        Change color of the cell by either calling the function
        `bgcolor("yellow")`
        `bgcolor("yellow", [0, 2, 4])`
        or with magic (has to be first line in cell!)
        `%bgcolor yellow`
        Cell selection is not possible with magic.

        Parameters
        ----------
        color : str
            Background color of the cell. A valid CSS color e.g.:
                - red
                - rgb(255,0,0)
                - #FF0000
            See https://www.rapidtables.com/web/css/css-color.html
        select : Optional[list[int]], default None
            Index of cells where the background color should be changed. Leave empty to change the color of the current cell.
            Note: The index only accounts for code-cells.
            Note: Not functional in 'magic-mode'.
        """
        if select:
            # color selected cells
            script = f"""
                    // get all code cells
                    let all_cells = document.querySelectorAll('.code_cell,.jp-CodeCell')
                    // define the cell selection
                    let selection = [{','.join(map(str, select))}];
                    // select the cells
                    let sel_cells = selection.map(x=>all_cells[x]);
                    // select the parts of the cells that should be colored (code area)
                    let ca = sel_cells.map(e=>Array.from(e.querySelectorAll('.CodeMirror-sizer,.highlight'))).flat();
                    // change the background color of all selected elements
                    ca.forEach(e=>e.style.background='{color}');
                    """
        else:
            # color the current cell
            script = f"""
                    // select the cell
                    let cell = [this.closest('.code_cell,.jp-CodeCell')];
                    // select the part of the cell that should be colored (code area)
                    let ca = [].slice.call(cell[0].querySelectorAll('.CodeMirror-sizer,.highlight'));
                    // change the background color of all selected elements
                    ca.forEach(e=>e.style.background='{color}');
                    """

        display(HTML(f'<img src onerror="{script}" style="display:none">'))


def clear() -> None:
    """
    Clear stout of console or jupyter notebook.

    https://stackoverflow.com/questions/37071230/clear-overwrite-standard-output-in-python
    """

    import platform

    if _is_notebook():
        utils.checker.check_module("IPython")
        from IPython.display import clear_output

        clear_output(wait=True)
    elif platform.system() == 'Windows':
        os.system('cls')
    else:
        os.system('clear')


def _compare_version(nb_name: str) -> None:
    """
    Compare installed sctoolbox version with notebook version.

    Parameters
    ----------
    nb_name : str
        Name of notebook file.
    """

    import sctoolbox
    import nbformat
    import warnings
    from packaging import version

    nb = nbformat.read(nb_name, as_version=4)
    if "sc_framework" in nb["metadata"]:
        if "version" in nb["metadata"]["sc_framework"]:
            # parse versions
            nb_ver = version.parse(str(nb["metadata"]["sc_framework"]["version"]))
            sc_ver = version.parse(sctoolbox.__version__)

            if nb_ver != sc_ver:
                ver_dif, arrow = ("an older", "<") if nb_ver < sc_ver else ("a newer", ">")
                warnings.warn(f"The notebook has {ver_dif} version compared to the installed sctoolbox version ({nb_ver} {arrow} {sc_ver}). Some functions may not work!")
            return

    warnings.warn("The Notebook seems to be outdated (notebook version: N/A). Some functions may not work! Consider updating.")
