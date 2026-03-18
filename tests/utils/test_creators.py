"""Test functions to download and create experiment folders."""

import sctoolbox.utils.creators as creator
import pytest
from glob import glob
from pathlib import Path
import shutil
from unittest.mock import patch, Mock


# --------------------------- TESTS --------------------------------- #


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


def test_setup_experiment():
    """Test experiment setup function."""
    dirs = ['raw', 'preprocessing', 'Analysis']
    creator.setup_experiment("./tmp/exp1", dirs=dirs)
    f = glob("./tmp/exp1/*/")
    shutil.rmtree("./tmp/")
    assert set([Path(file).name for file in f]) == set(dirs)


@pytest.mark.parametrize("starts,regex", [(1, '[0]*[1-9]?[1-9].*.ipynb'), (5, '[0]*[1-9]?[5-9].*.ipynb'),
                                          (10, '[0]*([1][0-9]|[2-9][0-9]).*.ipynb'), (21, '[0]*([2][1-9]|[3-9][0-9]).*.ipynb')])
def test_build_notebooks_regex(starts, regex):
    """Test build notebook regex function."""
    assert creator.build_notebooks_regex(starts) == regex


def test_add_analysis_FNF(tmp_path):
    """Test the add_analysis FileNotFoundError."""
    path = tmp_path / "not_existing"

    assert not path.exists()

    with pytest.raises(FileNotFoundError):
        creator.add_analysis(dest=str(path), analysis_name="FNF")


def test_add_analysis_FE(tmp_path):
    """Test the add_analysis FileExistsError."""
    name = "existing_analysis"
    path = tmp_path / name

    path.mkdir()

    assert path.exists()

    with pytest.raises(FileExistsError):
        creator.add_analysis(dest=str(path.parent), analysis_name=name)


def test_add_analysis(mocker, tmp_path):
    """Test the add_analysis function."""
    mock_ghd = mocker.patch("sctoolbox.utils.creators.github_download")

    creator.add_analysis(dest=str(tmp_path), analysis_name="new_analysis")

    assert mock_ghd.call_count == 2


def test_github_download(tmp_path, mocker):
    """Test the github_download function."""
    # mock key functions used within github_download
    class mockGitHub:
        class mockRepo:
            class mockContent:
                def __init__(self, name, path, decoded_content) -> None:
                    self.name = name
                    self.path = path
                    self.decoded_content = decoded_content
                    self.type = "file"

            def get_contents(self, *args, **kwargs):
                return [
                    self.mockContent(name="fileA.txt", path="path/to/fileA.txt", decoded_content=b"caf\xc3\xa9"),
                    self.mockContent(name="fileB.txt", path="fileB.txt", decoded_content="milk")
                ]

        def get_repo(self, *args, **kwargs):
            return self.mockRepo()

        # to enable context manager usage
        def __enter__(self):
            return self

        def __exit__(self, type, value, traceback):
            pass

    mocker.patch("sctoolbox.utils.creators.Github", return_value=mockGitHub())

    # the directory is empty
    assert not any(tmp_path.iterdir())

    creator.github_download(path=str(tmp_path), outpath=str(tmp_path))

    # the directory contains "downloaded" files
    assert any(tmp_path.iterdir())
