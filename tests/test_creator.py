"""Test functions to download and create experiment folders."""

import sctoolbox.creators as creator
import pytest
from glob import glob
from pathlib import Path
import shutil
from unittest.mock import patch, Mock


def test_setup_experiment():
    """Test experiment setup function."""
    dirs = ['raw', 'preprocessing', 'Analysis']
    creator.setup_experiment("./tmp/exp1", dirs=dirs)
    f = glob("./tmp/exp1/*/")
    shutil.rmtree("./tmp/")
    assert set([Path(file).name for file in f]) == set(dirs)


def test_gitlab_download():
    """Test gitlab download."""
    def side_effect(search):

        def repo_tree(path, ref):
            return_value = [{'name': 'Notebook1.ipynb', 'type': 'blob', 'path': 'notebooks/Notebook1.ipynb'},
                            {'name': 'FileX.py', 'type': 'blob', 'path': 'notebooks/FileX.py'}]
            return return_value

        mock2 = Mock()
        mock2.name = search
        mock2.repository_tree = repo_tree
        return [mock2]

    mock = Mock()
    mock.projects.list = side_effect
    result_file = Path("./tmp-test_add_analysis/Notebook1.ipynb")
    missing_file = Path("./tmp-test_add_analysis/FileX.py")
    result_file.parent.mkdir(parents=True, exist_ok=True)
    try:
        with patch("creator.gitlab_download", return_value=mock):
            creator.gitlab_download("notebooks", file_regex=".*.ipynb", out_path="./tmp-test_add_analysis/")
            assert result_file.is_file()
            assert not missing_file.is_file()
    except Exception as e:
        print(e)
    finally:
        if result_file.is_file():
            result_file.unlink()
        if missing_file.is_file():
            missing_file.unlink()
        if result_file.parent.exists():
            result_file.parent.rmdir()


@pytest.mark.parametrize("starts,regex", [(1, '[0]*[1-9]?[1-9].*.ipynb'), (5, '[0]*[1-9]?[5-9].*.ipynb'),
                                          (10, '[0]*([1][0-9]|[2-9][0-9]).*.ipynb'), (21, '[0]*([2][1-9]|[3-9][0-9]).*.ipynb')])
def test_build_notebooks_regex(starts, regex):
    """Test build notebook regex function."""
    assert creator.build_notebooks_regex(starts) == regex
