"""PowerPoint report generation functions."""

from pathlib import Path
import pptreport as ppt
import yaml
import pandas as pd
import tqdm
from sctoolbox.plotting.general import plot_table
from sctoolbox import settings

from beartype.typing import Optional, Dict, List, Any, Tuple, Literal, Union, Collection, Mapping
from beartype import beartype


@beartype
def _get_slide_kwargs(section: str, slide_section_dict: Dict[str, Dict[str, Any] | List[Dict[str, Any]]], slide_num: Optional[int] = None) -> Dict[str, Any]:
    """
    Returns the kwargs for the respective slide.

    Note: intended for use in `generate_report`

    Parameters
    ----------
    section: str
        The section name. A key of the `slide_section_dict`.
    slide_section_dict: Dict[str, Dict[str, Any] | List[Dict[str, Any]]]
        A dictionary where the keys are section names and the values are:
            - a list of dictionaries each corresponding to kwargs for `pptreport.PowerPointReport.add_slide`.
            - a dictionary corresponding to kwargs for `pptreport.PowerPointReport.add_slide`.
        In case of the first each list element defines a subsequent slide to be generated.
        In case of the second the dictionary is used as a template for all slides in this section.
    slide_num: Optional[int]
        The slide number. The index of the list element to be used.
        Ignored if the section contains a dictionary.

    Returns
    -------
    Dict[str, Any]
        A dict with kwargs for `pptreport.PowerPointReport.add_slide`.
    """
    # get section
    if isinstance(slide_section_dict[section], dict):
        # dict defines a template used for all slides in the section
        return slide_section_dict[section].copy()
    elif isinstance(slide_section_dict[section], list):
        # list defines one template per section slide
        return slide_section_dict[section][slide_num].copy()
    return {}  # TODO remove?


@beartype
def generate_report(
    dataset_name: str,
    section_titles: Dict[str, str],
    slide_sec_kwargs: Dict[str, Dict[str, Any] | List[Dict[str, Any]]],
    file_ext: List[str] = ["png", "md", "txt"],
    template: Optional[str | Path] = None,
    slide_format: Literal["widescreen", "standard", "a4-portait", "a4-landscape"] | Tuple[int | float, int | float] = "widescreen",
    report_dir: Optional[str | Path] = None,
    max_pixels: int | float = 1e7,
    **ppr_kwargs: Any
) -> ppt.PowerPoint:
    """
    Generate a PowerPoint report summarizing the current analysis.

    Parameters
    ----------
    dataset_name: str
        The name of the dataset. Used on the title slide.
    section_titles: Dict[str, str]
        A dictionary with the section id as keys and the display section name as values.
        This dictionary defines the what section to display and their order.
    slide_sec_kwargs: Dict[str, Dict[str, Any] | List[Dict[str, Any]]]
        A dictionary that holds all the information needed to create sections populated with slides.
        The top-level key corresponds to the section id (see `section_titles`). The top-level value is either a dictionary
        to provide a template for all slides in a section or a list of dicts providing a template for each slide of a section.
        The most inner dictionary defines keyword arguments for the `pptreport.PowerPointReport.add_slide` function.
    file_ext: List[str], default ["png", "md", "txt"]
    template: Optional[str]
        A PowerPoint template.
    slide_format: str, default "widescreen"
        Size of the presentation. Can be “standard”, “widescreen”, “a4-portait” or “a4-landscape”. Can also be a tuple of numbers indicating (height, width) in cm.
    report_dir: str | Path, default None
        Path to the directory containing all the report files.
        Set None to use `sctoolbox.settings.report_dir`.
    max_pixels: int, default 1e7
        Images with more pixels will be resized.
    ppr_kwargs: Any
        Keyword arguments forwarded to `pptreport.PowerPointReport`.

    Returns
    -------
    ppt.PowerPoint
        The fully populated PowerPoint object ready to be rendered and saved as a PowerPoint.
    """
    if report_dir is None:
        report_dir = settings.report_dir
    if isinstance(report_dir, str):
        report_dir = Path(report_dir)

    #Initialize presentation
    report = ppt.PowerPointReport(template=template, size=slide_format, **ppr_kwargs)

    #report._slides.clear()

    # add tool version slide
    versions = {}
    for yml in report_dir.glob("**/versions.yml"):
        with open(yml, "r") as file:
            versions |= yaml.safe_load(file)

    versions_file = str(report_dir / "versions.png")

    plot_table(
        table=pd.DataFrame(versions, index=["Version"]).transpose().reset_index(names="Name"),
        show_index=False,
        crop=None,
        col_width=[2, 12],
        save=versions_file
    )

    report.add_slide(
        title="Tool Versions",
        content=versions_file
    )

    # add dataset title slide
    report.add_title_slide(title=f"Dataset {dataset_name}",
                        subtitle="SC-Framework Analysis")

    # create section for each folder (notebook)
    for sec, title in tqdm.tqdm(section_titles.items()):
        sec_dir = report_dir / sec
        
        if not sec_dir.is_dir():
            report.logger.warning(f"Skipping invalid section named '{sec}'.")
            continue

        # collect files
        files = list(Path(sec_dir).glob(f"*[{'|'.join(file_ext)}]"))
        files = [f for f in files if not f.name.startswith("version")]  # skip versions.yml
        # extract prefixes to define the order
        prefixes = sorted(set(f.name.rsplit("_")[0] for f in files))
        
        report.add_title_slide(title=title)
        
        # create one slide for each prefix
        for prefix in prefixes:
            current_files = [f for f in files if f.name.startswith(prefix)]
            
            kwargs = _get_slide_kwargs(
                section=sec,
                slide_num=int(prefix[:2]) - 1,
                slide_section_dict=slide_sec_kwargs
            )
            
            if not "content" in kwargs:
                kwargs["content"] = [str(f) for f in current_files]
            if not "max_pixels" in kwargs:
                kwargs["max_pixels"] = max_pixels
            
            # add slide per prefix
            report.add_slide(
                **kwargs
            )

    return report
