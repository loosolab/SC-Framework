"""Class to create a genome track plot via pyGenomeTracks."""

import os
import tempfile
import subprocess
import matplotlib.pyplot as plt

from beartype import beartype
from beartype.typing import Optional, Any, Literal, Iterable

import sctoolbox.utils as utils
from sctoolbox._settings import settings
logger = settings.logger


@beartype
class GenomeTracks():
    """Class for creating a genome track plot via pyGenomeTracks by collecting different tracks and writing the .ini file.

    Examples
    --------
    .. plot::
        :context: close-figs
        :nofigs:

        import sctoolbox.plotting as pl

        G = pl.GenomeTracks()

        #Add bigwig tracks
        G.add_track("data/tracks/bigwig1.bw", color="red")
        G.add_track("data/tracks/bigwig2.bw", color="blue", orientation="inverted")

        #Add hlines to previous bigwig track
        G.add_hlines([100, 200], color="red")
        G.add_hlines([250], color="blue", line_style="dashed")

        #Add links
        G.add_track("data/tracks/links.arcs", orientation="inverted")

        #Add one line between tracks
        G.add_hline()

        #Add .bed-file regions
        G.add_track("data/tracks/tad_classification.bed", title="bed")
        G.add_track("data/tracks/tad_classification.bed", color="Reds", title="bed colored by score column")

        #Add vlines and highlight
        G.add_track("data/tracks/vlines.bed", file_type="vlines")
        G.add_track("data/tracks/vhighlight.bed", file_type="vhighlight")

        #Add a spacer
        G.add_spacer()

        #Add genes
        G.add_track("data/tracks/genes.gtf", gene_rows=5)

        #Add x-axis
        G.add_spacer()
        G.add_xaxis()

        # Plot
        G.plot(region="X:3000000-3500000", output="genometrack_X.png", trackLabelFraction=0.2)

    .. image:: genometrack_X.png
    """

    def __init__(self):
        """Initialize the GenomeTracks object."""

        self.tracks = []  # dictionary of tracks
        self.type_count = {}
        self.global_defaults = {"height": 2}  # dictionary of default values
        self.type_defaults = {"gtf": {"merge_transcripts": True, "fontsize": 12},
                              "hlines": {"overlay_previous": "share-y"}}

        self.available_types = ["bed", "bedgraph", "bedgraph_matrix", "bigwig", "domains",
                                "epilogos", "fasta", "gtf", "hic_matrix", "hic_matrix_square", "links",
                                "maf", "narrow_peak", "scalebar", "vhighlight", "vlines"]

        self.output = None  # path to the output file if written

    def __repr__(self):
        """Return a string representation of the GenomeTracks object."""
        n_tracks = len(self.tracks)
        return f"GenomeTracks object with {n_tracks} track(s). See <obj>.tracks for details."

    def add_track(self,
                  file: str,
                  file_type: Optional[str] = None,
                  name: Optional[str] = None,
                  **kwargs: Any):
        """Add a track to the GenomeTracks object.

        The track will be added to the configuration file as one element, e.g. .add_track("file1.bed", file_type="bed", name="my_bed") will add the following to the configuration file:
        ```
        [my_bed]
        file = file1.bed
        file_type = bed
        ```

        Additional parameters are decided by <obj>.global_defaults and <obj>.type_defaults, or can be given by kwargs. All options and parameters are available at:
        https://pygenometracks.readthedocs.io/en/latest/content/all_tracks.html

        Parameters
        ----------
        file : str
            Path to the file containing information to be plotted. Can be .bed, .bw, .gtf etc.
        file_type : str, default None
            Specify the 'file_type' argument for pyGenomeTracks. If None, the type will be predicted from the file ending.
        name : str, default None
            Name of the track. If None, the name will be estimated from the file_type e.g. 'bigwig 1'. or 'bed 2'. If the file_type is not available, the name will be the file path.
        **kwargs : arguments
            Additional arguments to be passed to pyGenomeTracks track configuration, for example `height=5` or `title="My track"`.

        Raises
        ------
        ValueError
            If the file_type is not valid.
        """

        # Setup
        track_dict = self.global_defaults.copy()
        track_dict["file"] = file

        # Predict file type
        if file_type is None:
            file_type = self._predict_type(file)
        else:
            # Check if file_type is valid
            if file_type not in self.available_types:
                if file_type == "spacer":
                    raise ValueError("file_type 'spacer' is not valid. Use GenomeTracks.add_spacer() instead.")
                elif file_type == "x-axis":
                    raise ValueError("file_type 'x-axis' is not valid. Use GenomeTracks.add_xaxis() instead.")
                elif file_type == "hlines":
                    raise ValueError("file_type 'hlines' is not valid. Use GenomeTracks.add_hlines() instead.")
                else:
                    raise ValueError(f"file_type '{file_type}' not valid. Choose from {self.available_types}")

        # If filetype was predicted or given; add to track dict
        if file_type is not None:
            if file_type in ["vlines", "vhighlight"]:
                track_dict["type"] = file_type   # file_type = type for some options
            else:
                track_dict["file_type"] = file_type
        track_dict.update(self.type_defaults.get(file_type, {}))  # add type defaults

        # Set title depending on file_type
        if file_type in ["vlines", "vhighlight"]:
            del track_dict["height"]
        else:
            track_dict["title"] = os.path.basename(file)  # per default, can be overwritten with kwargs

        # Final overwrite with kwargs
        track_dict.update(kwargs)

        # Count file-types
        if file_type not in self.type_count:
            self.type_count[file_type] = 1
        else:
            self.type_count[file_type] += 1

        # Add track to tracks dictionary
        if name is None:
            if file_type is None:
                name = file
            else:
                name = file_type + " " + str(self.type_count[file_type])

        self.tracks.append({name: track_dict})

    def add_hlines(self,
                   y_values: Iterable[int | float],
                   overlay_previous: Literal["share-y", "no"] = "share-y",
                   **kwargs: Any):
        """Add horizontal lines to the previous plot.

        Parameters
        ----------
        y_values : list of int or float
            List of y values to plot horizontal lines at.
        overlay_previous : str, default "share-y"
            Whether to plot the lines on the same y-axis as the previous plot ("share-y") or on a new y-axis ("no").
        **kwargs : arguments
            Additional arguments to be passed to pyGenomeTracks track configuration, for example `title="My lines"`.
        """

        y_values = [str(y) for y in y_values]

        d = {"hlines": {"y_values": ", ".join(y_values), "overlay_previous": overlay_previous, "title": ""}}
        d["hlines"].update(kwargs)
        d["hlines"]["file_type"] = "hlines"

        self.tracks.append(d)

    def add_hline(self,
                  height: int | float = 1,
                  line_width: int | float = 2,
                  **kwargs: Any):
        """Add a horizontal line between tracks, not within a track.

        Can be used to visually separate tracks.

        Parameters
        ----------
        height : int, default 1
            Height of the track with the line in the middle.
        line_width : int, default 2
            Width of the line.
        **kwargs : arguments
            Additional arguments to be passed to pyGenomeTracks track configuration, for example `title="A line"`.
        """

        d = {}
        d["height"] = height
        d["line_width"] = line_width
        d["show_data_range"] = kwargs.get("show_data_range", False)  # default is False
        d["overlay_previous"] = "no"

        self.add_hlines([1], min_value=0, max_value=2, **d)  # line is in the middle of the track

    def add_spacer(self,
                   height: int | float = 1):
        """Add a spacer between tracks.

        Parameters
        ----------
        height : int, default 1
            Height of the spacer track.
        """
        d = {"spacer": {"height": height}}
        self.tracks.append(d)

    def add_xaxis(self,
                  height: int | float = 1,
                  **kwargs: Any):
        """Add the x-axis to the plot.

        Parameters
        ----------
        height : int, default 1
            Height of the x-axis track.
        **kwargs : arguments
            Additional arguments to be passed to pyGenomeTracks track configuration.
        """

        d = {"height": height}
        d.update(kwargs)
        self.tracks.append({"x-axis": d})

    def _predict_type(self, file: str) -> str | None:
        """Predict the file type from the file ending or the contents of the file.

        Parameters
        ----------
        file : str
            Path of the file to be plotted.

        Returns
        -------
        str | None
            String of predicted file type or None if the file type could not be predicted.
        """

        if file.endswith(".bed"):
            return "bed"

        elif file.endswith(".bw"):
            return "bigwig"

        elif file.endswith(".gtf"):
            return "gtf"

        else:
            logger.warning(f"Could not predict file type for '{file}'. pyGenometracks will try to predict the file type. For more control, please specify 'file_type' manually in '.add_track'.")

    def _create_config_str(self) -> str:
        """Create configuration string based on tracks list.

        Returns
        -------
        config_str : str
            String containing the configuration file content
        """

        config_str = ""
        for d in self.tracks:
            track_name = list(d.keys())[0]
            track_dict = d[track_name]

            config_str += f"[{track_name}]\n"
            for key, value in track_dict.items():
                config_str += f"{key} = {value}\n"
            config_str += "\n"

        return config_str

    def _write_config(self, config_file: Optional[str] = None) -> str:
        """Write the configuration file to disk.

        Parameters
        ----------
        config_file : str, default None
            Path to the configuration file to create. If None, a temporary file will be created in the system's temp directory.

        Returns
        -------
        config_file : str
            Path to the configuration file.
        """

        if config_file is None:
            config_file = os.path.join(tempfile.gettempdir(), next(tempfile._get_candidate_names()))

        config_str = self._create_config_str()

        with open(config_file, "w") as f:
            f.write(config_str)

        return config_file

    def show_plot(self):
        """Display the plot."""

        if self.output is None:
            raise ValueError("No output file was created. Run GenomeTracks.plot() first.")

        if utils._is_notebook():
            from IPython.display import Image, IFrame, display

            if self.output.endswith(".png"):
                display(Image(filename=self.output))
            elif self.output.endswith(".pdf"):
                display(IFrame(self.output))
        else:
            import matplotlib.image as mpimg

            if self.output.endswith(".png"):
                img = mpimg.imread(self.output)
                plt.imshow(img)
                plt.axis('off')
                plt.show()
            else:
                logger.warning("Only .png files can be shown in the console.")

    def show_config(self):
        """Show the current configuration file as a string."""

        config_str = self._create_config_str()
        print(config_str)

    def plot(self,
             region: str,
             output: Optional[str] = "genometracks.png",
             config_file: Optional[str] = None,
             title: Optional[str] = None,
             show: bool = True,
             dpi: int = 300,
             **kwargs: Any):
        """
        Plot the final GenomeTracks plot based on the collected tracks.

        Runs pyGenomeTracks with the configuration file and the given parameters, and saves the output to the given file.

        Parameters
        ----------
        region : str
            Region to plot, e.g. "chr1:1000000-2000000".
        output : str, default "genometracks.png"
            Path to the output file.
        config_file : str, default None
            Path to the configuration file to create. If None, a temporary file will be created in the system's temp directory.
        title : str, default None
            Title of the plot. If None, no title will be shown.
        show : bool, default True
            If the function is run in a jupyter notebook, 'show' controls whether to show the plot at the end of the function run.
        dpi : int, default 300
            DPI of the plot.
        **kwargs : arguments
            Additional arguments to be passed to pyGenomeTracks, for example `trackLabelFraction=0.2`.

        Raises
        ------
        ValueError
            If the pyGenomeTracks command fails.
        """

        kwargs["title"] = f"'{title}'" if title is not None else None
        kwargs["dpi"] = dpi

        # create the .ini file
        ini_file = self._write_config(config_file=config_file)

        # Build command
        pgtracks_path = utils.get_binary_path("pyGenomeTracks")
        cmd = f"{pgtracks_path} --tracks {ini_file} --region {region} --outFileName {output} "

        # Add additional kwargs
        for key, value in kwargs.items():
            if value is not None:  # title might be None
                cmd += f" --{key} {value} "

        # Run pygenometracks
        logger.debug(f"Running command: '{cmd}'")
        try:
            subprocess.run(cmd, shell=True, check=True)
        except subprocess.CalledProcessError as e:
            self.output = None  # reset output in case it was previously plotted
            raise ValueError(f"Error while running pyGenomeTracks: {e.output}")

        # Remove config file
        if config_file is None:  # config_file was created by _write_config
            os.remove(ini_file)

        # Show in notebook
        if show:
            self.output = output
            self.show_plot()
