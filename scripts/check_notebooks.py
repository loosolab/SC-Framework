"""
Script to check whether notebook(s) contains output. Fails if it does.
"""

import sys
import json


def check_notebook(path):
    """ Check whether a notebook contains output. Exits with error if it does. """

    with open(path, 'r') as f:
        nb = json.load(f)

    for cell in nb['cells']:
        if 'outputs' in cell and len(cell['outputs']) > 0:
            print('Notebook {} contains output!'.format(path))
            sys.exit(1)


if __name__ == '__main__':

    if len(sys.argv) < 2:
        print('Usage: check_notebooks.py notebook1.ipynb notebook2.ipynb ...')
        sys.exit(1)

    for path in sys.argv[1:]:
        check_notebook(path)
