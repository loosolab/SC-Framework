import os
import matplotlib.pyplot as plt
import sctoolbox.plotting as pl

# Prevent figures from being shown, we just check that they are created
plt.switch_backend("Agg")

tracks_folder = os.path.join(os.path.dirname(__file__), 'data', 'tracks')


def test_genometracks():
    """Test genometracks plotting."""

    G = pl.GenomeTracks()

    # Add bigwig tracks
    G.add_track(f"{tracks_folder}/bigwig1.bw", color="red")
    G.add_track(f"{tracks_folder}/bigwig2.bw", color="blue", orientation="inverted")

    # Add hlines to previous bigwig track
    G.add_hlines([100, 200], color="red")
    G.add_hlines([250], color="blue", line_style="dashed")

    # Add links
    G.add_track(f"{tracks_folder}/links.arcs", orientation="inverted", height=2)

    # Add one line between tracks
    G.add_hline()

    # Add .bed-file regions
    G.add_track(f"{tracks_folder}/tad_classification.bed", title="bed")
    G.add_track(f"{tracks_folder}/tad_classification.bed", color="Reds", title="bed colored by score column")

    # Add vlines and highlight
    G.add_track(f"{tracks_folder}/vlines.bed", file_type="vlines")
    G.add_track(f"{tracks_folder}/vhighlight.bed", file_type="vhighlight")

    # Add a spacer
    G.add_spacer()

    # Add genes
    G.add_track(f"{tracks_folder}/genes.gtf", gene_rows=5)

    # Add x-axis
    G.add_spacer()
    G.add_xaxis()

    assert len(G.tracks) == 14

    # Plot
    G.plot(region="X:3000000-3500000", output="genometrack_X.png", trackLabelFraction=0.2)

    assert os.path.isfile("genometrack_X.png")

    # Remove file
    os.remove("genometrack_X.png")
