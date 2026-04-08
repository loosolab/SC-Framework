"""Test qc_filter plotting function."""

import pytest
import sctoolbox.plotting.qc_filter as pl
import sctoolbox.tools.insertsize as insertsize
import os
import shutil
import numpy as np
import glob
import matplotlib.pyplot as plt
import seaborn as sns
import functools

from beartype.roar import BeartypeCallHintParamViolation

quant_folder = os.path.join(os.path.dirname(__file__), '../data', 'quant')


# ------------------------------ TESTS --------------------------------- #


@pytest.mark.parametrize("order", [None, ["KO-2", "KO-1", "Ctrl-2", "Ctrl-1"]])
def test_plot_starsolo_quality(order):
    """Test plot_starsolo_quality success."""
    res = pl.plot_starsolo_quality(quant_folder, order=order)

    assert isinstance(res, np.ndarray)


@pytest.mark.parametrize("folder,measures,exception,match", [
    ("invalid", None, ValueError, "No STARsolo summary files found in folder*"),
    (quant_folder, ["invalid"], KeyError, "Measure .* not found in summary table"),
])
def test_plot_starsolo_quality_failure(folder, measures, exception, match):
    """Test plot_starsolo_quality failure with invalid input."""
    with pytest.raises(exception, match=match):
        pl.plot_starsolo_quality(folder, measures=measures)


def test_plot_starsolo_UMI():
    """Test plot_starsolo_UMI success."""
    res = pl.plot_starsolo_UMI(quant_folder)

    assert isinstance(res, np.ndarray)


def test_plot_starsolo_UMI_failure():
    """Test plot_starsolo_UMI failure with invalid input."""

    # Create a quant folder without UMI files
    shutil.copytree(quant_folder, "quant_without_UMI", dirs_exist_ok=True)
    UMI_files = glob.glob("quant_without_UMI/*/solo/Gene/UMI*")
    for file in UMI_files:
        os.remove(file)

    # Test that valueerror is raised
    with pytest.raises(ValueError, match="No UMI files found in folder*"):
        pl.plot_starsolo_UMI("quant_without_UMI")

    # remove folder
    shutil.rmtree("quant_without_UMI")


@pytest.mark.parametrize("groupby", [None, "condition"])
@pytest.mark.parametrize("add_labels", [True, False])
def test_n_cells_barplot(adata, groupby, add_labels):
    """Test n_cells_barplot success."""

    axarr = pl.n_cells_barplot(adata, "louvain", groupby=groupby, add_labels=add_labels)

    if groupby is None:
        assert len(axarr) == 1
    else:
        assert len(axarr) == 2


def test_group_correlation(adata):
    """Test if plot is written to pdf."""

    # Run group correlation
    pl.group_correlation(adata, groupby="condition", save="group_correlation.pdf")

    # Assert creation of file
    assert os.path.isfile("group_correlation.pdf")
    os.remove("group_correlation.pdf")


def test_insertsize_plotting(atac_adata):
    """Test if insertsize plotting works."""

    adata = atac_adata.copy()
    fragments = os.path.join(os.path.dirname(__file__), '..', 'data', 'atac', 'mm10_atac_fragments.bed')
    insertsize.add_insertsize(adata, fragments=fragments)

    ax = pl.plot_insertsize(adata)

    ax_type = type(ax).__name__
    assert ax_type.startswith("Axes")


def test_link_sliders(slider_list):
    """Test _link_sliders success."""
    linkage_list = pl._link_sliders(slider_list)
    assert isinstance(linkage_list, list)
    assert type(linkage_list[0]).__name__ == 'link'


@pytest.mark.parametrize("global_threshold", [True, False])
def test_toggle_linkage(checkbox, slider_list, global_threshold):
    """Test if toggle_linkage runs without error."""
    column = "Test"
    linkage_dict = dict()
    linkage_dict[column] = pl._link_sliders(slider_list) if global_threshold is True else None
    checkbox.observe(functools.partial(pl._toggle_linkage, linkage_dict=linkage_dict, slider_list=slider_list, key=column), names=["value"])
    assert True


def test_update_threshold(slider):
    """Test if update_threshold runs without error."""
    fig, _ = plt.subplots()
    slider.observe(functools.partial(pl._update_thresholds, fig=fig, min_line=1, min_shade=1, max_line=1, max_shade=1), names=["value"])
    assert True


@pytest.mark.parametrize("columns, which, groupby", [(['qc_float', 'LISI_score_pca'], "obs", "condition"),
                                                     (['qc_float', 'LISI_score_pca'], "obs", "cat"),
                                                     (['qc_float_var'], "var", None)])
@pytest.mark.parametrize("color_list", [None, sns.color_palette("Set1", 3)])
@pytest.mark.parametrize("title", [None, "Title"])
def test_quality_violin(adata, groupby, columns, which, title, color_list):
    """Test quality_violin success."""
    figure, slider = pl.quality_violin(adata, columns=columns, groupby=groupby,
                                       which=which, title=title, color_list=color_list)
    assert type(figure).__name__ == "Figure"
    assert isinstance(slider, dict)


@pytest.mark.parametrize("kwargs,exception,match", [
    ({"columns": ["qc_float"], "which": "Invalid"}, BeartypeCallHintParamViolation, None),
    ({"groupby": "condition", "columns": ["qc_float"], "color_list": sns.color_palette("Set1", 1)}, ValueError, "Increase the color_list variable"),
    ({"groupby": "condition", "columns": ["qc_float"], "header": []}, ValueError, "Length of header does not match"),
    ({"columns": ["Invalid"]}, ValueError, "The following columns from 'columns' were not found"),
])
def test_quality_violin_fail(adata, kwargs, exception, match):
    """Test quality_violin failure."""
    with pytest.raises(exception, match=match):
        pl.quality_violin(adata, **kwargs)


def test_get_slider_thresholds_dict(slider_dict):
    """Test get_slider_threshold for non grouped slider_dict."""
    threshold_dict = pl.get_slider_thresholds(slider_dict)
    assert isinstance(threshold_dict, dict)
    assert threshold_dict == {'LISI_score_pca': {'min': 5, 'max': 7}, 'qc_float': {'min': 5, 'max': 7}}


def test_get_slider_thresholds_dict_grouped(slider_dict_grouped):
    """Test get_slider_threshold for grouped slider_dict."""
    threshold_dict = pl.get_slider_thresholds(slider_dict_grouped)
    assert isinstance(threshold_dict, dict)
    assert threshold_dict == {'LISI_score_pca': {'min': 5, 'max': 7},
                              'qc_float': {'min': 5, 'max': 7}}


def test_get_slider_thresholds_dict_grouped_diff(slider_dict_grouped_diff):
    """Test get_slider_threshold for grouped slider_dict with different slider values."""
    threshold_dict = pl.get_slider_thresholds(slider_dict_grouped_diff)
    assert isinstance(threshold_dict, dict)
    assert threshold_dict == {'A': {'1': {'min': 5, 'max': 7},
                                    '2': {'min': 1, 'max': 5}},
                              'B': {'1': {'min': 5, 'max': 7},
                                    '2': {'min': 3, 'max': 4}}}


@pytest.mark.parametrize("thresholds, expected", [({'qcvar1': {'C1': {'min': 0.1, 'max': 0.9},
                                                               'C2': {'min': 0.1, 'max': 0.9},
                                                               'C3': {'min': 0.1, 'max': 0.9}},
                                                    'qcvar2': {'C1': {'min': 0.2, 'max': 0.8},
                                                               'C2': {'min': 0.2, 'max': 0.8},
                                                               'C3': {'min': 0.2, 'max': 0.8}}
                                                    }, True),
                                                  ({'qcvar1': {'C1': {'min': 0.1, 'max': 0.8},
                                                               'C2': {'min': 0.2, 'max': 0.9},
                                                               'C3': {'min': 0.2, 'max': 0.6}},
                                                    'qcvar2': {'C1': {'min': 0.1, 'max': 1},
                                                               'C2': {'min': 0.3, 'max': 0.7},
                                                               'C3': {'min': 0.2, 'max': 0.7}}
                                                    }, False)])
def test_upset_select_cells(adata, thresholds, expected):
    """Test upset_select_cells success."""
    global_thresholds = {'qcvar1': {'min': 0.1, 'max': 0.9},
                         'qcvar2': {'min': 0.2, 'max': 0.8}}

    sample_selection = pl._upset_select_cells(adata, thresholds, groupby='condition')
    global_selection = pl._upset_select_cells(adata, global_thresholds, groupby=None)

    assert expected == (sample_selection == global_selection).all().all()


@pytest.mark.parametrize("groupby,match", [
    (None, "Parameter groupby is set to None while threshold*"),
    ("louvain", "Wrong group selection.*"),
])
def test_upset_select_cells_fail(adata, groupby, match):
    """Test upset_select_cells fail."""
    grouped_thresholds = {
        'qcvar1': {'C1': {'min': 0.1, 'max': 0.9},
                   'C2': {'min': 0.1, 'max': 0.9},
                   'C3': {'min': 0.1, 'max': 0.9}},
        'qcvar2': {'C1': {'min': 0.2, 'max': 0.8},
                   'C2': {'min': 0.2, 'max': 0.8},
                   'C3': {'min': 0.2, 'max': 0.8}}
    }
    with pytest.raises(ValueError, match=match):
        pl._upset_select_cells(adata, grouped_thresholds, groupby=groupby)


@pytest.mark.parametrize("thresholds, groupby", [({'qcvar1': {'min': 0.1, 'max': 0.9},
                                                   'qcvar2': {'min': 0.2, 'max': 0.8}}, None),
                                                 ({'qcvar1': {'C1': {'min': 0.1, 'max': 0.9},
                                                              'C2': {'min': 0.1, 'max': 0.9},
                                                              'C3': {'min': 0.1, 'max': 0.9}},
                                                   'qcvar2': {'C1': {'min': 0.2, 'max': 0.8},
                                                              'C2': {'min': 0.2, 'max': 0.8},
                                                              'C3': {'min': 0.2, 'max': 0.8}}
                                                   }, 'condition')])
def test_upset_plot_filter_impacts(adata, thresholds, groupby):
    """Test upset_plot_filter_impacts success."""
    plot_result = pl.upset_plot_filter_impacts(adata, thresholds=thresholds, groupby=groupby)

    assert isinstance(plot_result, dict)
    assert list(plot_result.keys()) == ['matrix', 'shading', 'totals', 'intersections']
    ax_type = type(plot_result['matrix']).__name__
    assert ax_type.startswith("Axes")
    ax_type = type(plot_result['shading']).__name__
    assert ax_type.startswith("Axes")
    ax_type = type(plot_result['intersections']).__name__
    assert ax_type.startswith("Axes")

    plot_result = pl.upset_plot_filter_impacts(adata, thresholds=thresholds, groupby=groupby, limit_combinations=2)

    assert isinstance(plot_result, dict)
    assert list(plot_result.keys()) == ['matrix', 'shading', 'totals', 'intersections']
    ax_type = type(plot_result['matrix']).__name__
    assert ax_type.startswith("Axes")
    ax_type = type(plot_result['shading']).__name__
    assert ax_type.startswith("Axes")
    ax_type = type(plot_result['intersections']).__name__
    assert ax_type.startswith("Axes")
