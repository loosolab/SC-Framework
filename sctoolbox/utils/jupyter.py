"""Jupyter notebook related functions."""

import os
from IPython.core.magic import register_line_magic
from IPython.display import HTML, display
import sctoolbox.utils as utils

from typing import Optional


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
    def bgcolor(color: str, cell: Optinal[str] = None) -> None:
        """
        Set background color of current jupyter cell.

        Adapted from https://stackoverflow.com/a/53746904.
        Note: Jupyter notebook v6+ needed

        Change color of the cell by either calling the function
        `bgcolor("yellow")`
        or with magic (has to be first line in cell!)
        `%bgcolor yellow`

        Parameters
        ----------
        color : str
            Background color of the cell. A valid CSS color e.g.:
                - red
                - rgb(255,0,0)
                - #FF0000
            See https://www.rapidtables.com/web/css/css-color.html
        cell : Optional[str], default None
            Code of the cell that will be evaluated.
        """

        script = f"""
                var cell = this.closest('.code_cell');
                var editor = cell.querySelector('.CodeMirror-sizer');
                editor.style.background='{color}';
                this.parentNode.removeChild(this)
                """

        display(HTML(f'<img src onerror="{script}">'))


def clear() -> None:
    """
    Clear stout of console or jupyter notebook.

    https://stackoverflow.com/questions/37071230/clear-overwrite-standard-output-in-python
    """

    import platform

    if _is_notebook():
        utils.check_module("IPython")
        from IPython.display import clear_output

        clear_output(wait=True)
    elif platform.system() == 'Windows':
        os.system('cls')
    else:
        os.system('clear')
