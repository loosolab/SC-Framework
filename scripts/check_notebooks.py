"""
Script to check whether notebook(s) contains output. Fails if it does.
"""

import sys
import json
import glob


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

    paths = [glob.glob(f) for f in sys.argv[1:]]
    paths = sum(paths, [])  # flatten list

    for path in paths:
        print(f"Checking {path} for output...")
        check_notebook(path)
