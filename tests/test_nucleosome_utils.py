import pytest
import os
import numpy as np
import scanpy as sc
import sctoolbox.atac as atac
import sctoolbox.nucleosome_utils as nu
from scipy.signal import find_peaks

# ------------------------ Fixtures ------------------------ #


@pytest.fixture
def count_table():
    fragments = os.path.join(os.path.dirname(__file__), 'data', 'atac', 'mm10_atac_fragments.bed')
    return atac._insertsize_from_fragments(fragments, barcodes=None)


@pytest.fixture
def disturbed_sine(freq=3.1415 * 2):
    in_array = np.linspace(0, freq, 1000)
    sine_wave = np.sin(in_array)
    in_array = np.linspace(0, 500, 1000)
    disturbance = np.sin(in_array)
    scaled_disturbance = disturbance / 10
    disturbed_sine = sine_wave + scaled_disturbance

    return disturbed_sine, sine_wave


@pytest.fixture
def stack_sines(disturbed_sine):

    sines = []
    disturbed_sine_waves = []
    for i in range(10):
        disturbed_sine_wave, sine_wave = disturbed_sine
        sines.append(sine_wave)
        disturbed_sine_waves.append(disturbed_sine_wave)

    sines = np.array(sines)
    disturbed_sine_waves = np.array(disturbed_sine_waves)

    return sines, disturbed_sine_waves


@pytest.fixture
def modulation():
    """This fixture creates a modulation curve"""
    def gaussian(x, mu, sig):  # Gaussian function
        return np.exp(-np.power(x - mu, 2.) / (2 * np.power(sig, 2.)))

    curves = []
    x_values = np.linspace(-3, 3, 1000)
    for mu, sig in [(-1, 0.5), (0.2, 0.3), (-1, 3)]:  # Gaussian curves with different means and standard deviations
        curves.append(gaussian(x_values, mu, sig))

    curves[1] = curves[1] / 1  # Peak 1
    curves[1] = curves[1] / 5  # Peak 2
    curves[2] = curves[2] / 5  # Bias
    sum_c = np.sum(curves, axis=0)  # Sum of the curves

    return sum_c


@pytest.fixture
def fragment_distributions():
    testdata = np.loadtxt(os.path.join(os.path.dirname(__file__), 'data', 'atac', 'nucleosomal_score.csv'), delimiter=None)

    return testdata


@pytest.fixture
def density_reference():
    densities = np.loadtxt(os.path.join(os.path.dirname(__file__), 'data', 'atac', 'densities.txt'), delimiter=None)

    return densities


@pytest.fixture
def adata():
    """ Fixture for an AnnData object. """
    adata = sc.read_h5ad(os.path.join(os.path.dirname(__file__), 'data', 'atac', 'mm10_atac.h5ad'))
    return adata


@pytest.fixture
def bamfile():
    """ Fixture for an Bamfile.  """
    bamfile = os.path.join(os.path.dirname(__file__), 'data', 'atac', 'mm10_atac.bam')
    return bamfile


# ------------------------ Tests ------------------------ #

def test_moving_average(disturbed_sine):
    """
    Test that the moving average function works as expected, by comparing a smoothed disturbed sine wave to the original
    and inspecting the difference.

    """
    disturbed_sine_wave, sine_wave = disturbed_sine
    smoothed_sine = nu.moving_average(disturbed_sine_wave, n=10)

    diff_smooth = np.sum(abs(sine_wave - smoothed_sine))
    diff_disturbed = np.sum(abs(sine_wave - disturbed_sine_wave))

    print("\t")
    print("smoothed difference: " + str(diff_smooth))
    print("disturbed difference: " + str(diff_disturbed))
    assert diff_smooth < 15


def test_multi_ma(stack_sines):
    """
    Test that the multi_ma function works as expected by comparing a smoothed disturbed sine wave to the original
    """
    sine_stack, dist_stack = stack_sines
    smoothed = nu.multi_ma(dist_stack)

    diff_ori = abs(sine_stack - dist_stack)
    diff_smooth = abs(sine_stack - smoothed)

    sum_ori = np.sum(diff_ori, axis=1)
    sum_smooth = np.sum(diff_smooth, axis=1)

    print("\t")
    print("smoothed difference: " + str(sum_smooth))
    print("disturbed difference: " + str(sum_ori))

    assert np.all(sum_smooth < 15)


def test_scale(count_table):
    """ Test that the scale function works as expected by checking that the max value is 1 and the min value is 0 """
    table = count_table

    dist = table[[c for c in table.columns if isinstance(c, int)]]
    dists_arr = dist.to_numpy()
    dists_arr = np.nan_to_num(dists_arr)

    scaled = nu.scale(dists_arr)
    scaled_single = nu.scale(dists_arr[0])

    assert np.max(scaled) == 1
    assert np.min(scaled) == 0

    assert np.max(scaled_single) == 1
    assert np.min(scaled_single) == 0


def test_calc_densities(fragment_distributions, density_reference):
    """ Test that the calc_densities function works as expected by asserting known values """
    densities = nu.calc_densities(fragment_distributions)

    assert np.array_equal(densities, density_reference)


def test_call_peaks_worker(modulation):
    """ Test that the call_peaks_worker function works as expected """
    peaks = nu.call_peaks_worker(modulation)

    assert peaks[0] == 333


def test_call_peaks(stack_sines):
    """ Test that the call_peaks function works as expected """
    sine_stack, dist_stack = stack_sines  # get the stack of sines
    peaks = nu.call_peaks(sine_stack)

    assert len(peaks) == 10
    assert np.all(np.vstack(peaks) == 250)


def test_filter_peaks(disturbed_sine):
    """ Test that the filter_peaks function works as expected """
    peaks = np.array([50, 250, 400, 500, 999])
    disturbed_sine_wave, sine_wave = disturbed_sine

    filtered_peaks = nu.filter_peaks(peaks, sine_wave, peaks_thr=0.75, operator="bigger")
    filtered_peaks_smaller = nu.filter_peaks(peaks, sine_wave, peaks_thr=0.75, operator="smaller")

    assert len(filtered_peaks) == 1
    assert filtered_peaks[0] == 250

    assert len(filtered_peaks_smaller) == 4
    assert np.all(filtered_peaks_smaller == np.array([50, 400, 500, 999]))


def test_momentum_diff(modulation):
    """
    Test that the momentum_diff function works as expected, by modeling overlapping gaussian curves and checking that
    the function finds the correct peaks.
    """
    sum_c = modulation  # get the sum of the gaussian curves

    peaks_raw, _ = find_peaks(sum_c, height=0.1)  # Find peaks in the sum of the curves

    mom, a, b = nu.momentum_diff(sum_c, remove=0, shift=50, smooth=False)  # Calculate the momentum difference

    mom_peaks, _ = find_peaks(mom, height=0.1)  # Find peaks in the momentum difference

    assert len(peaks_raw) == 1
    assert len(mom_peaks) == 2


def test_add_adapters():
    """Test the add_adapters function"""
    input = np.array([[1, 2, 3, 4, 5, 6, 7, 8, 9], [1, 2, 3, 4, 5, 6, 7, 8, 9]])

    added = nu.add_adapters(input, shift=10, smooth=False)  # add adapters to the input array

    # check for 10 zeros added to the beginning and end of each row
    assert np.all(added[:, :10] == 0)


def test_cross_point_shift(modulation):
    """This test checks that the cross_point_shift function works as expected"""
    sum_c = modulation  # get the sum of the gaussian curves
    mom, a, b = nu.momentum_diff(sum_c, remove=0, shift=50, smooth=False)  # Calculate the momentum difference
    peaks, _ = find_peaks(mom, height=0.1)  # Find peaks in the momentum difference

    # get the cross point shift
    shifted_peaks = nu.cross_point_shift(peaks, reference=mom, convergence=0.07)

    assert ((mom[shifted_peaks] <= 0.08).all())  # check that the shifted peaks are below the convergence threshold


def test_single_cwt_ov(modulation):
    """Tests the single_cwt_ov function"""
    features = [modulation]
    coef, filtered_peaks = nu.single_cwt_ov(features, shift=0, sample=0, freq=4)

    assert len(filtered_peaks) == 2


def test_score_by_momentum(fragment_distributions):
    """Tests the score_by_cwt function, by scoring data of different quality from high to low"""
    testdata = fragment_distributions
    testdata = nu.scale(testdata)  # scale
    testdata = nu.multi_ma(testdata)  # smooth
    scores = nu.score_by_momentum(testdata, plotting=False)  # score

    assert scores[0] > scores[1]
    assert scores[1] > scores[2]


def test_score_by_cwt(fragment_distributions):
    """Tests the score_by_cwt function, by scoring data of different quality from high to low"""
    testdata = fragment_distributions
    testdata = nu.scale(testdata)
    scores = nu.score_by_cwt(testdata, plotting=False)

    assert scores[0] > scores[1]
    assert scores[1] > scores[2]


def test_density_plot(density_reference, fragment_distributions):
    """ Tests the density_plot function """
    scaled = nu.scale(fragment_distributions)
    ax = nu.density_plot(scaled, density_reference)

    ax_type = type(ax).__name__

    assert ax_type.startswith("Axes")


def test_plot_single_momentum_ov(fragment_distributions):
    """ Tests the plot_single_momentum_ov function """
    scaled = nu.scale(fragment_distributions)

    mom, shift_l, shift_r = nu.momentum_diff(scaled, remove=0, shift=50, smooth=False)  # Calculate the momentum difference

    peaks = nu.call_peaks(mom, n_threads=8)  # Find peaks in the momentum difference

    peaks = nu.filter_peaks(peaks,
                            reference=mom,
                            peaks_thr=0.03,
                            operator='bigger')

    fig, axes = nu.plot_single_momentum_ov(peaks,
                                           mom,
                                           data=scaled,
                                           shift_l=shift_l,
                                           shift_r=shift_r,
                                           sample_n=0,
                                           shift=0,
                                           remove=0)

    fig_type = type(fig).__name__
    ax1_type = type(axes[0]).__name__
    ax2_type = type(axes[1]).__name__
    ax3_type = type(axes[2]).__name__

    assert fig_type.startswith("Figure")
    assert ax1_type.startswith("Axes")
    assert ax2_type.startswith("Axes")
    assert ax3_type.startswith("Axes")


def test_plot_wavl_ov(fragment_distributions):
    """ Tests the plot_single_momentum_ov function """
    plot_sample = 0
    scaled = nu.scale(fragment_distributions)

    shifted_peaks, wav_features, coefs = nu.wrap_cwt(data=scaled,
                                                     adapter=250,
                                                     wavelet='gaus1',
                                                     scales=16,
                                                     n_threads=4,
                                                     peaks_thr=0.05)

    fig, axes = nu.plot_wavl_ov(wav_features[plot_sample],
                                shifted_peaks[plot_sample],
                                [coefs[plot_sample]],
                                freq=0,
                                plot_peaks=True,
                                perform_cross_point_shift=True,
                                convergence=0)

    fig_type = type(fig).__name__
    ax1_type = type(axes[0]).__name__
    ax2_type = type(axes[1]).__name__
    ax3_type = type(axes[2]).__name__

    assert fig_type.startswith("Figure")
    assert ax1_type.startswith("Axes")
    assert ax2_type.startswith("Axes")
    assert ax3_type.startswith("Axes")


def test_add_insertsize_metrics(adata, bamfile):
    """ check that the add_insertsize_metrics function works as expected """

    adata = nu.add_insertsize_metrics(adata, bamfile, plotting=False)

    assert 'nucleosomal_score_momentum' in adata.obs.columns
    assert 'nucleosomal_score_cwt' in adata.obs.columns
    assert 'genome_counts' in adata.obs.columns
