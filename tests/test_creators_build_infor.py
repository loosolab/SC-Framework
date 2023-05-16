import sctoolbox.creators as creator
import anndata
import pytest
import pandas as pd
from glob import glob
from pathlib import Path
import shutil
from unittest.mock import patch, Mock


@pytest.fixture
def test_adata():
    """ Create anndata object. """
    adata = anndata.AnnData(
        pd.DataFrame(
            {
                "a": pd.Series(1, index=list(range(2)), dtype="float32"),
                "b": pd.Series(1, index=list(range(2)), dtype="float32")
            },
            index=["a", "b", "c", "d"]
        )
    )

    return adata


@pytest.fixture
def color_list():
    """ Create list of colors. """
    return ['red', 'blue', 'green', 'pink', 'chartreuse', 'gray', 'yellow', 'brown',
            'purple', 'orange', 'wheat', 'lightseagreen', 'cyan', 'khaki', 'cornflowerblue',
            'olive', 'gainsboro', 'darkmagenta', 'slategray', 'ivory', 'darkorchid', 'papayawhip',
            'paleturquoise', 'oldlace', 'orangered', 'lavenderblush', 'gold', 'seagreen', 'deepskyblue',
            'lavender', 'peru', 'silver', 'midnightblue', 'antiquewhite', 'blanchedalmond', 'firebrick',
            'greenyellow', 'thistle', 'powderblue', 'darkseagreen', 'darkolivegreen', 'moccasin',
            'olivedrab', 'mediumseagreen', 'lightgray', 'darkgreen', 'tan', 'yellowgreen', 'peachpuff',
            'cornsilk', 'darkblue', 'violet', 'cadetblue', 'palegoldenrod', 'darkturquoise', 'sienna',
            'mediumorchid', 'springgreen', 'darkgoldenrod', 'magenta', 'steelblue', 'navy', 'lightgoldenrodyellow',
            'saddlebrown', 'aliceblue', 'beige', 'hotpink', 'aquamarine', 'tomato', 'darksalmon', 'navajowhite',
            'lawngreen', 'lightsteelblue', 'crimson', 'mediumturquoise', 'mistyrose', 'lightcoral', 'mediumaquamarine',
            'mediumblue', 'darkred', 'lightskyblue', 'mediumspringgreen', 'darkviolet', 'royalblue', 'seashell',
            'azure', 'lightgreen', 'fuchsia', 'floralwhite', 'mintcream', 'lightcyan', 'bisque', 'deeppink',
            'limegreen', 'lightblue', 'darkkhaki', 'maroon', 'aqua', 'lightyellow', 'plum', 'indianred', 'linen',
            'honeydew', 'burlywood', 'goldenrod', 'mediumslateblue', 'lime', 'lightslategray', 'forestgreen', 'dimgray',
            'lemonchiffon', 'darkgray', 'dodgerblue', 'darkcyan', 'orchid', 'blueviolet', 'mediumpurple',
            'darkslategray', 'turquoise', 'salmon', 'lightsalmon', 'coral', 'lightpink', 'slateblue', 'darkslateblue',
            'white', 'sandybrown', 'chocolate', 'teal', 'mediumvioletred', 'skyblue', 'snow', 'palegreen', 'ghostwhite',
            'indigo', 'rosybrown', 'palevioletred', 'darkorange', 'whitesmoke']


@pytest.mark.parametrize("key,value", [(1, 2), ("test", "string"), ("test", 1), (5, "test"), ("asdf", [1, 2, 3]), ("dict", {})])
def test_build_infor_wo_infoprocess(test_adata, key, value):
    """ Test if key-value pair is added to adata.uns["infoprocess"]. Without prior dict existing. """
    creator.build_infor(test_adata, key, value)

    assert test_adata.uns["infoprocess"][key] == value


def test_build_infor_w_infoprocess(test_adata):
    """ Test if key-value pair is added to empty adata.uns["infoprocess"]. """
    test_adata.uns["infoprocess"] = {}
    creator.build_infor(test_adata, 5, 1)

    assert test_adata.uns["infoprocess"][5] == 1


@pytest.mark.parametrize("key,value", [(1, 2), ("test", "string"), ("test", 1), (1, "test"), ("asdf", [1, 2, 3]), ("dict", {})])
def test_build_infor_no_inplace(test_adata, key, value):
    """ Test if adata object copy contains key-value pair. """
    adata = creator.build_infor(test_adata, key, value, inplace=False)

    assert adata.uns["infoprocess"][key] == value


@pytest.mark.parametrize("key", [[list(), dict()]])
def test_build_infor_invalid_type(test_adata, key):
    """ Test invalid key error. """
    with pytest.raises(TypeError, match="unhashable type: .*"):
        creator.build_infor(test_adata, key, 1)


@pytest.mark.parametrize(("invalid_type", "key", "value"), [(1, 1, 1), ("test", 1, 1), ([1, 2, 3], 1, 1)])
def test_build_infor_no_anndata(invalid_type, key, value):
    """ Test invalid object for anndata. """
    with pytest.raises(TypeError, match="Invalid data type. AnnData object is required."):
        creator.build_infor(invalid_type, key, value)


def test_add_color_set_inplace(test_adata, color_list):
    """ Test if list of color is added. """
    creator.add_color_set(test_adata)

    assert test_adata.uns["color_set"] == color_list


def test_add_color_set_wrong_inplace_nonetype(test_adata):
    """ Test that None is returned for inplace=True. """
    result = creator.add_color_set(test_adata, inplace=True)

    assert result is None


def test_add_color_set_not_inplace(test_adata, color_list):
    """ Test inplace=False returns modified copy. """
    modified_adata = creator.add_color_set(test_adata, inplace=False)

    assert modified_adata.uns["color_set"] == color_list


def test_add_color_set_wrong_not_inplace(test_adata):
    """ Test that inplace=False does not modify original object. """
    creator.add_color_set(test_adata, inplace=False)

    assert "color_set" not in test_adata.uns


@pytest.mark.parametrize("invalid_type", [1, "test", [1, 2, 3]])
def test_add_color_set_no_anndata(invalid_type):
    """ Test error on not anndata input. """
    with pytest.raises(TypeError, match="Invalid data type. AnnData object is required."):
        creator.add_color_set(invalid_type)


def test_setup_experiment():
    """ Test experiment setup function"""
    dirs = ['raw', 'preprocessing', 'Analysis']
    creator.setup_experiment("./tmp/exp1", dirs=dirs)
    f = glob("./tmp/exp1/*/")
    shutil.rmtree("./tmp/")
    assert set([Path(file).name for file in f]) == set(dirs)


def test_gitlab_download():
    """Test gitlab download"""
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


@pytest.mark.parametrize("starts,regex", [(1, '[0]*[1-9?][1-9].*.ipynb'), (5, '[0]*[1-9?][5-9].*.ipynb'),
                                          (10, '[0]*([1][0-9]|[2-9][0-9]).*.ipynb'), (21, '[0]*([2][1-9]|[3-9][0-9]).*.ipynb')])
def test_build_notebooks_regex(starts, regex):
    """ Test build notebook regex function"""
    assert creator.build_notebooks_regex(starts) == regex
