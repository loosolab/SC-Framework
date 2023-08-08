"""Tools for scATAC nucleosome analysis."""
import numpy as np
import matplotlib.pyplot as plt
import pywt
import multiprocessing as mp
from scipy.signal import find_peaks
from scipy.signal import fftconvolve
import sctoolbox.tools as tools

import sctoolbox.utils.decorator as deco
from sctoolbox._settings import settings
logger = settings.logger


def moving_average(series, n=10):
    """
    Move average filter to smooth out data.

    This implementation ensures that the smoothed data has no shift and
    local maxima remain at the same position.

    Parameters
    ----------
    series : array
        Array of data to be smoothed.
    n : int, default 10
        Number of steps to the left and right of the current step to be averaged.

    Returns
    -------
    numpy.array:
        Smoothed array
    """

    list(series)
    smoothed = []
    for i in range(len(series)):  # loop over all steps
        sumPerStep = 0
        if i > n and i <= (len(series) - n):  # main phase
            for j in range(-n, n):
                sumPerStep += series[i + j]
            smoothed.append(sumPerStep / (n * 2))
        elif i > (len(series) - n):  # end phase
            smoothed.append(series[i])
        elif i <= n:  # init phase
            smoothed.append(series[i])

    smoothed = np.array(smoothed)

    return smoothed


def multi_ma(series, n=2, window_size=10, n_threads=8):
    """
    Multiprocessing wrapper for moving average filter.

    Parameters
    ----------
    series : np.ndarray
        Array of data to be smoothed.
    n : int, default 2
        Number of times to apply the filter
    window_size : int, default 10
        Number of steps to the left and right of the current step to be averaged.
    n_threads : int, default 8
        Number of threads to be used for multiprocessing.

    Returns
    -------
    np.ndarray
        array of smoothed array
    """

    # smooth
    for i in range(n):

        smooth_series = []
        # init pool
        pool = mp.Pool(n_threads)
        jobs = []
        # loop over chunks

        for dist in series:
            job = pool.apply_async(moving_average, args=(dist, window_size))
            jobs.append(job)
        pool.close()

        # collect results
        for job in jobs:
            smooth_series.append(job.get())

        series = np.array(smooth_series)

    return series


def scale(series_arr):
    """
    Scale a series array to a range of 0 to 1.

    Parameters
    ----------
    series_arr : np.ndarray
        Array of data to be scaled 1D or 2D

    Notes
    -----
    If the array is 2D, the scaling is done on axis=1.

    Returns
    -------
    np.ndarray
        Scaled array
    """

    if len(series_arr.shape) == 1:
        max_v = np.max(series_arr)
        scaled_arr = series_arr / max_v

        return scaled_arr

    elif len(series_arr.shape) == 2:
        # Scale as sum off all features within cell match one
        maxis = np.max(series_arr, axis=1)  # total features per cell
        scaled_arr = np.divide(series_arr.T, maxis).T

        return scaled_arr


def calc_densities(features):
    """
    Calculate the density of a feature array, for each feature.

    The density is stored in a matrice of size (n_features, 1000).

    Parameters
    ----------
    features : np.ndarray
        Array of features to calculate the density for

    Returns
    -------
    np.ndarray
        Array of densities
    """

    # calculate densities for a binned grid X,y of size original bins / 1000
    densities = []
    for i in range(0, len(features[0])):
        column = features[:, i]
        scaled_1000 = np.around(column * 1000).astype(int)
        gradient = np.bincount(scaled_1000, minlength=1001)
        densities.append(gradient)
    densities = np.array(densities)

    return densities


def call_peaks(data, n_threads=4, distance=50, width=10):
    """
    Find peaks for multiple arrays at once.

    Parameters
    ----------
    data : np.ndarray
        Array of arrays to find peaks in (2D).
    n_threads : int, default 4
        Number of threads to be used for multiprocessing.
    distance : int, default 50
        Minimum distance between peaks.
    width : int, default 10
        Minimum width of peaks.

    Notes
    -----
    Multiprocessing wrapper for scipy.signal.find_peaks.

    Returns
    -------
    np.ndarray
        Array of peaks (index of data)
    """

    peaks = []

    pool = mp.Pool(n_threads)
    jobs = []

    for array in data:
        job = pool.apply_async(call_peaks_worker, args=(array, distance, width))
        jobs.append(job)
    pool.close()

    # collect results
    for job in jobs:
        peak_list = job.get()
        peaks.append(peak_list)

    # peaks = np.array(peaks)

    return peaks


def call_peaks_worker(array, distance=50, width=10):
    """
    Worker function for multiprocessing of scipy.signal.find_peaks.

    Parameters
    ----------
    array : np.ndarray
        Array of data to find peaks in.
    distance : int, default 50
        Minimum distance between peaks.
    width : int, default 10
        Minimum width of peaks.

    Returns
    -------
    np.ndarray
        Array of peaks (index of data)
    """

    peaks, _ = find_peaks(array, distance=distance, width=width)

    return peaks


def filter_peaks(peaks, reference, peaks_thr, operator='bigger'):
    """
    Filter peaks based on a reference array and a threshold.

    Parameters
    ----------
    peaks : np.ndarray
        Array of peaks to be filtered.
    reference : np.ndarray
        Array of reference values (e.g. data were peaks were found).
    peaks_thr : float
        Threshold for filtering.
    operator : str, default 'bigger'
        Operator for filtering. Options ['bigger', 'smaller'].

    Returns
    -------
    np.ndarray
        Filtered array of peaks
    """

    filtered_peaks = []

    if operator == "bigger":
        if len(reference.shape) == 1:
            filtered_peaks = peaks[np.where(reference[peaks] >= peaks_thr)]
        if len(reference.shape) == 2:
            for i, index in enumerate(peaks):
                filtered_peaks.append(index[np.where(reference[i, index] >= peaks_thr)])

    if operator == "smaller":
        if len(reference.shape) == 1:
            filtered_peaks = peaks[np.where(reference[peaks] <= peaks_thr)]
        if len(reference.shape) == 2:
            for i, index in enumerate(peaks):
                filtered_peaks.append(index[np.where(reference[i, index] <= peaks_thr)])

    return filtered_peaks


# ////////////////////////// Momentum \\\\\\\\\\\\\\\\\\\\\\\\\\\

def momentum_diff(data, remove=150, shift=80, smooth=True):
    """
    Calculate the momentum of a series by subtracting the original series with a shifted version of itself.

    Parameters
    ----------
    data : np.ndarray
        Array of data to calculate the momentum for.
    remove : int, default 150
        Number of samples to remove from the beginning of the series.
    shift : int, default 80
        Number of samples to shift the series.
    smooth : bool, default True
        Smooth the momentum series.

    Returns
    -------
    tuple of len 3 containing np.ndarrays
        Index 1: np.array containg the momentum
        Index 2: np.array containg the shifted data a (data[:-shift])
        Index 3: np.array containg the shifted data b (data[shift:])
    """

    if len(data.shape) == 1:
        shifted_data = data[remove:]
        a = shifted_data[:-shift]
        b = shifted_data[shift:]
        momentum = a - b
    if len(data.shape) == 2:
        shifted_data = data[:, remove:]
        a = shifted_data[:, :-shift]
        b = shifted_data[:, shift:]
        momentum = a - b

    if smooth:
        momentum = multi_ma(momentum, n=1, window_size=10, n_threads=8)

    return momentum, a, b


def score_by_momentum(data,
                      shift=80,
                      remove=100,
                      sample_to_inspect=0,
                      peaks_thr=0.03,
                      period=160,
                      penalty_scale=100,
                      plotting=True):
    """
    Calculate momentum and score cells based on the number of peaks and the distance between them.

    Parameters
    ----------
    data : np.ndarray
        Array of data to calculate the momentum for.
    shift : int, default 80
        Number of samples to shift the series.
    remove : int, default 100
        Number of samples to remove from the beginning of the series.
    sample_to_inspect : int, default 0
        Index of sample to inspect as reference.
    peaks_thr : float, default 0.03
        Threshold for filtering peaks.
    period : int, default 160
        expected peak period.
    penalty_scale : int, default 100
        penalty factor for each peak that is not in the expected period.
    plotting : bool, default True
        Plot the momentum and the peaks of the reference sample.

    Returns
    -------
    np.ndarray
        Array of scores
    """

    logger.info('calculate momentum...')
    momentum, shift_l, shift_r = momentum_diff(data=data, remove=remove, shift=shift)
    logger.info('find peaks...')
    peaks = call_peaks(momentum, n_threads=8)
    logger.info('filter peaks...')
    peaks = filter_peaks(peaks,
                         reference=momentum,
                         peaks_thr=peaks_thr,
                         operator='bigger')

    if plotting:
        logger.info('plot single cell...')
        plot_single_momentum_ov(peaks=peaks,
                                momentum=momentum,
                                data=data,
                                shift_l=shift_l,
                                shift_r=shift_r,
                                sample_n=sample_to_inspect,
                                shift=shift,
                                remove=remove)

    logger.info('calc scores...')
    scores = []
    for i in range(len(peaks)):
        peak_list = peaks[i]
        single_momentum = momentum[i]

        if len(peak_list) == 0:
            score = 0
        elif len(peak_list) == 1:
            score = single_momentum[peak_list[0]] / 100
        elif len(peak_list) > 1:
            corrected_scores = []
            for j in range(1, len(peak_list)):
                amplitude = single_momentum[peak_list[j - 1]] * 2

                diff = peak_list[j] - peak_list[j - 1]
                corrected_score = amplitude - (abs(diff - period) / penalty_scale)
                if corrected_score < 0:
                    corrected_score = 0

                corrected_scores.append(corrected_score)

            score = float(np.sum(np.array(corrected_scores))) + 0

        scores.append(score)

    scores = np.array(scores)

    if plotting:
        fig, ax = plt.subplots()
        ax.hist(np.sort(scores), bins=100, log=True)
        ax.set_title('Scores')
        ax.set_xlabel('Score')
        ax.set_ylabel('Abundance')

    return scores


# //////////////////////////// CWT \\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\

def add_adapters(features, shift=250, smooth=False, window_size=30):
    """
    Add adapters to the beginning of the features array for transient oscillations.

    Parameters
    ----------
    features : np.ndarray
        Array of features.
    shift : int, default 250
        Length of the adapter.
    smooth : bool, default False
        Smooth the features after adding the adapters.
    window_size : int, default 30
        Window size for smoothing.

    Returns
    -------
    np.ndarray
        Array of features with adapters
    """

    if shift != 0:
        for i in range(shift):
            features = np.insert(features, 0, 0, axis=1)

    if smooth:
        features = multi_ma(features, n=1, window_size=window_size)

    return features


def cross_point_shift(peaks, reference, convergence=0.01):
    """
    Cross point shift peaks to the left to the first point where the reference is below the convergence threshold.

    Parameters
    ----------
    peaks : np.ndarray
        Array of peaks.
    reference : np.ndarray
        Array of reference.
    convergence : float, default 0.01
        Convergence threshold.

    Returns
    -------
    np.ndarray
        Array of corrected peaks
    """

    corrected_peaks = []
    latest = 0
    for peak in peaks:
        shift = 0
        for gradient in reversed(reference[latest:peak]):
            if gradient <= convergence:
                corrected_peaks.append(peak - shift)
                break
            else:
                shift += 1

        latest = peak
    corrected_peaks = np.delete(corrected_peaks, np.where(np.array(corrected_peaks) > len(reference)))

    return corrected_peaks


def half_wave_shift(peaks, scale):
    """
    Shift the peaks to the left by half a wavelength.

    Parameters
    ----------
    peaks : np.ndarray
        Array of peaks.
    scale : int
        Scale of the wavelet.

    Returns
    -------
    np.ndarray
        Array of corrected peaks.

    """

    freq = pywt.scale2frequency('gaus1', scale)
    shift = 1 / freq
    peaks - shift

    return peaks


def single_cwt_ov(features,
                  shift=250,
                  sample=0,
                  freq=4,
                  peaks_thr=0.5,
                  perform_cross_point_shift=True,
                  convergence=0.1,
                  plotting=True):
    """
    Apply Continues Wavelet Transformation (CWT) to a single sample and plot the results.

    Parameters
    ----------
    features : np.ndarray
        Array of arrays of the fragment length distribution.
    shift : int, default 250
        Number of samples to shift the series (length of the adapter).
    sample : int, default 0
        Index of the sample to inspect.
    freq : int, default 4
        Frequency to inspect.
    peaks_thr : float, default 0.5
        Threshold for filtering peaks.
    perform_cross_point_shift : bool, default True
        Perform cross point shift.
    convergence : float, default 0.1
        Convergence threshold.
    plotting : bool, default True
        If True plot CWT.

    Returns
    -------
    np.ndarray
        Array of coefficients
    """

    feature = features[sample]
    if shift != 0:
        for i in range(shift):
            feature = np.insert(feature, 0, 0)

    wavelet = "gaus1"
    scales = np.array([2 ** x for x in range(1, 10)])

    coef, freqs = pywt.cwt(feature, scales, wavelet)

    peaks, _ = find_peaks(coef[freq], distance=50, width=10)

    filtered_peaks = filter_peaks(peaks, reference=coef[freq], peaks_thr=peaks_thr, operator='bigger')

    if plotting:
        plot_wavl_ov(feature,
                     filtered_peaks,
                     coef, freq=freq,
                     plot_peaks=True,
                     perform_cross_point_shift=perform_cross_point_shift,
                     convergence=convergence)

    return coef, filtered_peaks


def mp_cwt(features, wavelet='gaus1', scales=16, n_threads=8):
    """
    Multiprocess Continues Wavelet Transformation (CWT).

    Parameters
    ----------
    features : np.ndarray
        Array of arrays of the fragment length distribution.
    wavelet : str, default 'gaus1'
        Wavelet to use.
    scales : int / or array of ints, default 16
        Scales for the CWT.
    n_threads : int, default 8
        Number of threads to use.

    Returns
    -------
    np.ndarray
        Array of coefficients
    """

    coef_arr = []
    # init pool
    pool = mp.Pool(n_threads)
    jobs = []

    # loop over chunks
    for feature in features:
        job = pool.apply_async(cwt_worker, args=(feature, wavelet, scales))
        jobs.append(job)
    pool.close()

    # collect results
    for job in jobs:
        coef_arr.append(job.get()[0])

    coef_arr = np.array(coef_arr)

    return coef_arr


def cwt_worker(feature, wavelet="gaus1", scales=16):
    """
    Perform the CWT on a single feature.

    Parameters
    ----------
    feature : np.array
        Array of the fragment length distribution.
    wavelet : str, default 'gaus1'
        Wavelet to use.
    scales : int / or array of ints, default 16
        Scales for the CWT.

    Notes
    -----
    Worker function for mp_cwt().

    Returns
    -------
    np.array
        Array of coefficients
    """

    coef, freqs = pywt.cwt(feature, scales, wavelet)

    return coef


def wrap_cwt(data,
             adapter=250,
             wavelet='gaus1',
             scales=16,
             n_threads=8,
             peaks_thr=0.1,
             convergence=0.01):
    """
    Find peaks in multiple fragment length distributions.

    Using CWT and scipy.signal.find_peaks performed on a single frequency.
    Peaks are filtered by a threshold and shifted to the left to the first point
    where the reference is below the convergence threshold.

    Parameters
    ----------
    data : np.array
        Array of arrays of the fragment length distributions.
    adapter : int, default 250
        Number of zeros to attach to the left of the series (length of the adapter).
    wavelet : str, default 'gaus1'
        Wavelet to use.
    scales : int, default 16
        Scale for the CWT.
    n_threads : int, default 8
        Number of threads to use.
    peaks_thr : float, default 0.1
        Threshold for filtering peaks.
    convergence : float, default 0.01
        Convergence point.

    Returns
    -------
    tuple of length 3:
        1: peaks
        2: wav features
        3: coeficients
    """

    wav_features = add_adapters(data, shift=adapter)
    coefs = mp_cwt(wav_features, wavelet=wavelet, scales=scales, n_threads=n_threads)
    peaks = call_peaks(coefs, n_threads=n_threads)
    peaks = filter_peaks(peaks, reference=coefs, peaks_thr=peaks_thr, operator='bigger')

    cp_shifted = []
    for i in range(len(peaks)):
        cp_shifted.append(cross_point_shift(peaks[i], reference=coefs[i], convergence=convergence))

    # remove adapter
    # peaks = np.array(peaks)  # Remove as fix for unit testing
    # shifted_peaks = peaks - adapter
    shifted_peaks = [x - adapter for x in peaks]

    nn_peaks = []
    for peak_list in shifted_peaks:
        peak_list[np.where(peak_list < 0)] = 0
        nn_peaks.append(peak_list)

    wav_features = wav_features[:, adapter:]
    coefs = coefs[:, adapter:]

    return nn_peaks, wav_features, coefs


def score_by_cwt(data,
                 plot_sample=0,
                 plotting=True,
                 adapter=250,
                 wavelet='gaus1',
                 scales=35,
                 peaks_thr=0.05,
                 penalty_scale=100,
                 period=160,
                 n_threads=8):
    """
    Calculate scores for each cell using CWT.

    The score is calculated as the sum of the peak amplitudes in the coefficient array
    and corrected by the peak - peak distance given by the argument period and a scaling factor.

    Parameters
    ----------
    data : np.array
        Array of arrays of the fragment length distributions.
    plot_sample : int, default 0
        Index of the sample to plot.
    plotting : bool, default True
        Plot the sample.
    adapter : int, default 250
        Number of zeros to attach to the left of the series (length of the adapter).
    wavelet : str, default 'gaus1'
        Wavelet to use.
    scales : int, default 35
        Scale for the CWT.
    peaks_thr : float, default 0.05
        Threshold for filtering peaks.
    penalty_scale : float, default 100
        Scaling factor for the penalty.
    period : int, default 160
        Period of the peaks.
    n_threads : int, default 8
        Number of threads to use.

    Returns
    -------
    np.array
        Array of scores
    """

    logger.info('performing CWT on: ' + str(len(data)) + ' cells')
    logger.info('using wavelet type: ' + wavelet)
    logger.info('with scale: ' + str(scales))
    shifted_peaks, wav_features, coefs = wrap_cwt(data=data,
                                                  adapter=adapter,
                                                  wavelet=wavelet,
                                                  scales=scales,
                                                  n_threads=n_threads,
                                                  peaks_thr=peaks_thr)

    if plotting:
        logger.info('plotting single cell...')
        plot_wavl_ov(wav_features[plot_sample],
                     shifted_peaks[plot_sample],
                     [coefs[plot_sample]], freq=0,
                     plot_peaks=True,
                     perform_cross_point_shift=True,
                     perform_half_wave_shift=False,
                     scale=scales,
                     convergence=0)

    logger.info('calculate scores...')

    scores = []
    for i in range(len(shifted_peaks)):
        peak_list = shifted_peaks[i]
        coef = coefs[i]

        if len(peak_list) == 0:
            score = 0
        elif len(peak_list) == 1:
            score = 0
        elif len(peak_list) > 1:
            corrected_scores = []
            for j in range(1, len(peak_list)):
                amplitude = coef[peak_list[j]]

                diff = peak_list[j] - peak_list[j - 1]
                corrected_score = amplitude - (abs(diff - period) / penalty_scale)
                if corrected_score < 0:
                    corrected_score = 0

                corrected_scores.append(corrected_score)
            score = float(np.sum(np.array(corrected_scores))) + 0

        scores.append(score)

    scores = np.array(scores)

    if plotting:
        fig, ax = plt.subplots()
        ax.hist(np.sort(scores), bins=50, log=True)
        ax.set_title('Scores')
        ax.set_xlabel('Score')
        ax.set_ylabel('Abundance')

    return scores


# ///////////////////////// Plotting \\\\\\\\\\\\\\\\\\\\\\\\\\\\

def density_plot(scaled, densities):
    """
    Plot the density of the fragment length distributions.

    Parameters
    ----------
    scaled : np.array
        Array of arrays of the scaled fragment length distributions
    densities : np.array
        2D array of the densities

    Returns
    -------
    matplotlib.axes.Axes
        Axes object of density plot.
    """

    # plot density
    normalized = np.log2(densities)  # normalize log2
    rotated = np.rot90(normalized, k=3)  # rotate 90'
    rotated = np.flip(rotated, axis=1)
    stretch = len(rotated[0]) / len(rotated[:, 0])  # calculate stretch for good visibility
    mean = scaled.sum(axis=0) / len(scaled)
    scaled_mean = mean * 1000

    fig, ax = plt.subplots()
    ax.set_title('Fragment Size Density Plot')
    ax.set_xlabel('Fragment Length', color='blue')
    ax.set_ylabel('Abundance')
    ax.imshow(rotated[:-1, :], cmap='viridis', interpolation='nearest', aspect=stretch)
    ax.plot(scaled_mean, color="red", markersize=1)
    plt.gca().invert_yaxis()
    # plt.show()  # plots are shown automatically in jupyter notebook

    return ax


def plot_single_momentum_ov(peaks,
                            momentum,
                            data,
                            shift_l,
                            shift_r,
                            sample_n=0,
                            shift=80,
                            remove=150):
    """
    Plot the momentum of a single sample with found peaks and the original data.

    Parameters
    ----------
    peaks : np.array
        Array of arrays of the found peaks.
    momentum : np.array
        Array of arrays of the momentum.
    data : np.array
        Array of arrays of the fragment length distributions.
    shift_l : np.array
        Array of arrays of the left shifts.
    shift_r : np.array
        Array of arrays of the right shifts.
    sample_n : int, default 0
        Index of the sample to plot.
    shift : int, default 80
        Shift to apply to the peaks to plot with the original data.
    remove : int, default 150
        Number of bases removed from the left of the fragment length distribution.

    Returns
    -------
    tuple of length 2:
        1: matplotlib figure
        2: list (legnth 3) of matplotlib.axes.Axes objects
    """
    single_m = momentum[sample_n]
    single_d = data[sample_n]
    sample_peaks = peaks[sample_n]

    a = shift_l[sample_n]
    b = shift_r[sample_n]

    points_x = sample_peaks
    points_y = single_m[sample_peaks]

    points_x_corrected = sample_peaks - int(shift / 2)

    points_ori_x = points_x_corrected + remove
    points_ori_y = single_d[points_ori_x]

    fig, (ax1, ax2, ax3) = plt.subplots(3, 1)
    ax1.set_title('sample: ' + str(sample_n))
    ax1.set_ylabel('Amplitude')
    ax1.set_xlabel('Fragment Length - ' + str(remove) + 'bp-shift', color='blue')
    ax1.plot(a)
    ax1.plot(b)

    ax2.set_ylabel('Momentum')
    ax2.set_xlabel('Fragment Length - ' + str(remove) + 'bp-shift', color='blue')
    ax2.plot(single_m)
    ax2.scatter(points_x, points_y, color='red', zorder=2)

    ax3.set_ylabel('scaled abundance')
    ax3.set_xlabel('Fragment Length', color='blue')
    ax3.plot(single_d)
    ax3.scatter(points_ori_x, points_ori_y, color='red', zorder=2)

    return fig, [ax1, ax2, ax3]


def plot_wavl_ov(feature,
                 peaks,
                 coef,
                 freq=6,
                 plot_peaks=True,
                 perform_cross_point_shift=True,
                 perform_half_wave_shift=True,
                 scale=35,
                 convergence=0.1):
    """
    Plot the original data, the wavelet transformation and the found peaks as an overview.

    Parameters
    ----------
    feature : np.array
        Array of arrays of the fragment length distributions.
    peaks : np.array
        Array of arrays of the found peaks.
    coef : np.array
        Array of coefficients of the wavelet transformation.
    freq : int, default 6
        Index of the frequency to plot.
    plot_peaks : bool, default True
        If true, the found peaks are plotted.
    perform_cross_point_shift : bool, default True
        If true, the found peaks are shifted to the cross point.
    perform_half_wave_shift : bool, default True
        If true, the found peaks are shifted to the left by half a wavelength.
    scale : int, default 35
        Scale of the wavelet. Required by half_wave_shift.
    convergence : float, default 0.1
        Convergence value for the cross point shift.

    Returns
    -------
    tuple of length 2:
        1: matplotlib figure
        2: list (legnth 3) of matplotlib.axes.Axes objects
    """

    # index frequence of interest
    coef_freq = coef[freq]

    if len(peaks) != 0:
        # define the new points to add
        x_values_freq = peaks
        y_values_freq = coef_freq[peaks]

        # define points for the original abundances
        if perform_cross_point_shift:
            peaks = cross_point_shift(peaks, reference=coef[freq], convergence=convergence)

        if perform_half_wave_shift:
            peaks = half_wave_shift(peaks, scale)
        x_values = peaks
        y_values = feature[peaks]
    else:
        plot_peaks = False

    '''Plots the close prices, returns in a single plot and the wavelet transformation in a plot below'''
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 10))

    ax1.set_title('Frequency Chart')
    ax1.plot(coef_freq, color='blue')
    ax1.set_ylabel('Amplitude')
    ax1.set_xlabel('Fragment Length', color='blue')
    if plot_peaks:
        ax1.scatter(x_values_freq, y_values_freq, color='red', zorder=2)

    ax2.set_title('Fragment Lengths Chart')
    ax2.plot(feature, color='green')
    ax2.set_xlabel('Fragment Length')
    ax2.set_ylabel('Abundances', color='blue')
    if plot_peaks:
        ax2.scatter(x_values, y_values, color='red', zorder=2)

    ax3.set_title('Continues Wavelet Transformation (CWT)')
    ax3.imshow(coef, aspect='auto', cmap='hot')
    ax3.set_xlabel('Fragment Length')
    ax3.set_ylabel('Frequencies', color='blue')

    fig.tight_layout()
    # plt.show()

    return fig, [ax1, ax2, ax3]


def custome_wavelet(amplitude=1.0,
                    frequency=3.0,
                    phase_shift=np.pi / 0.65,
                    mu=0.0,
                    sigma=150.0,
                    plotting=True):
    """
    Implement a custom wavelet.

    The wavelet is a sine curve multiplied by a Gaussian curve.

    Parameters
    ----------
    amplitude : float, default 1.0
        Amplitude of the sine curve.
    frequency : float, default 3
        Frequency of the sine curve.
    phase_shift : float, default np.pi / 0.65
        Phase shift of the sine curve.
    mu : float, default 0.0
        Mean of the Gaussian.
    sigma : float, default 150.0
        Standard deviation of the Gaussian.
    plotting : bool, default True
        If true, the wavelet is plotted.

    Returns
    -------
    np.array
        Array of the wavelet
    """

    center = amplitude  # Center of the sine wave

    wing_size = 300
    # Create an array of x values
    x = np.linspace(-wing_size, wing_size, wing_size * 2)

    # Compute the centered sine curve values for each x
    sine_curve = amplitude * np.sin(2 * np.pi * frequency * x + phase_shift) + center

    # Compute the Gaussian values for each x
    gaussian = (1 / (sigma * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x - mu) / sigma) ** 2)
    wavelet = sine_curve * gaussian

    if plotting:
        fig, ax = plt.subplots()
        ax.plot(wavelet)
        ax.set_title('Wavelet')
        ax.set_xlabel('Interval')
        ax.set_ylabel('Amplitude')

    return wavelet


def custome_cwt(data, mode='convolve', plot_wavl=False):
    """
    Get custom implementation of the continuous wavelet transformation.

    Parameters
    ----------
    data : np.array
        Array of arrays of the fragment length distributions.
    mode : str, default 'concolve'
        Mode of the convolution. Either 'convolve' or 'fftconvolve'.
    plot_wavl : bool, default False
        If true, the wavelet is plotted.

    Returns
    -------
    np.array
        Array of convolved data.
    """

    # Get the wavelet
    wavelet = custome_wavelet(plotting=plot_wavl)

    # convolve with the data
    convolved_data = []
    for cell in data:
        if mode == 'convolve':
            convolved_data.append(np.convolve(cell, wavelet, mode='same'))
        elif mode == 'fftconvolve':
            convolved_data.append(fftconvolve(data, wavelet, mode='same'))

    return np.array(convolved_data)


def score_by_ct_cwt(data,
                    plot_wavl=False,
                    n_threads=12,
                    peaks_thr=0.01,
                    operator='bigger',
                    plotting_mask=False,
                    plotting_ov=True,
                    sample=0):
    """
    Get score by custom continuous wavelet transformation and score mask.

    Parameters
    ----------
    data : np.array
        Array of arrays of the fragment length distributions.
    plot_wavl : bool, default False
        If true, the wavelet is plotted.
    n_threads : int, default 12
        Number of threads to use for the peak calling.
    peaks_thr : float, default 0.01
        Threshold for the peak calling.
    operator : str, default 'bigger'
        Operator to use for the peak calling. Either 'bigger' or 'smaller'.
    plotting_mask : bool, default False
        If true, the score mask is plotted.
    plotting_ov : bool, default True
        If true, the overlay of the score mask and the convolved data is plotted.
    sample : int, default 0
        Index of the sample to plot.

    Returns
    -------
    np.array
        Array of scores for each sample
    """

    convolved_data = custome_cwt(data, plot_wavl=plot_wavl)

    peaks = call_peaks(convolved_data, n_threads=n_threads)

    filtered_peaks = filter_peaks(peaks, reference=convolved_data, peaks_thr=peaks_thr, operator=operator)

    score_mask = build_score_mask(plotting=plotting_mask)
    scores = []
    for i, peak_list in enumerate(filtered_peaks):
        conv = convolved_data[i]

        if len(peak_list) == 0:
            score = 0
        elif len(peak_list) == 1:
            score = conv[peak_list[0]] * score_mask[0][peak_list[0]]

        elif len(peak_list) > 1:
            score = 0
            for j in range(1, len(peak_list)):
                if j <= 3:
                    score += conv[peak_list[j]] * score_mask[j][peak_list[j]]
                else:
                    score += conv[peak_list[j]] * score_mask[3][peak_list[j]]

        scores.append(score)

    if plotting_ov:
        plot_custom_cwt(convolved_data, data, filtered_peaks, scores=scores, sample_n=sample)

    return scores


def build_score_mask(plotting=True, mu_list=[42, 200, 360, 550], sigma_list=[25, 35, 45, 25]):
    """
    Build a score mask for the score by custom continuous wavelet transformation.

    Mask is a sum of 4 Gaussian curves with mu and sigma specified
    for the expected peak positions and deviations.

    Parameters
    ----------
    plotting : bool, default True
        If true, the score mask is plotted.
    mu_list : list, default [42, 200, 360, 550]
        List of mu values for the Gaussian curves.
    sigma_list : list, default [25, 35, 45, 25]
        List of sigma values for the Gaussian curves.

    Returns
    -------
    np.array
        Array of the score mask
    """

    # Create an array of x values
    x = np.linspace(0, 1000, 1000)
    gaussians = []
    for mu, sigma in zip(mu_list, sigma_list):
        gaussians.append(scale((1 / (sigma * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x - mu) / sigma) ** 2)))

    gaussians = np.array(gaussians)

    if plotting:
        fig, ax = plt.subplots()

        ax.plot(gaussians[0])
        ax.plot(gaussians[1])
        ax.plot(gaussians[2])
        ax.plot(gaussians[3])
        ax.set_title('score-mask')
        ax.set_xlabel('position')
        ax.set_ylabel('scoring')

    return gaussians


def gauss(x, mu, sigma):
    """
    Calculate the values of the Gaussian function for a given x, mu and sigma.

    Parameters
    ----------
    x : array
        x values
    mu : float
        mu value
    sigma : float
        sigma value

    Returns
    -------
    float
        Value of the Gaussian function for the given x, mu and sigma.
    """

    gaussian = (1 / (sigma * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x - mu) / sigma) ** 2)

    return gaussian


def plot_custom_cwt(convolved_data, data, peaks, scores, sample_n=0):
    """
    Plot the overlay of the convolved data, the peaks and the score mask.

    Parameters
    ----------
    convolved_data : np.array
        Array of the convolved data.
    data : np.array
        Array of the original data.
    peaks : np.array
        Array of the peaks.
    scores : np.array
        Array of the scores.
    sample_n : int, defualt 0
        Index of the sample to plot.
    """

    single_m = convolved_data[sample_n]
    single_d = data[sample_n]
    sample_peaks = peaks[sample_n]

    points_x = sample_peaks
    points_y = single_m[sample_peaks]

    points_y_data = single_d[sample_peaks]

    fig, (ax1, ax2, ax3) = plt.subplots(3, 1)

    ax1.set_ylabel('convolution of sample: ' + str(sample_n))
    ax1.set_xlabel('fragment length', color='blue')
    ax1.plot(single_m)
    ax1.scatter(points_x, points_y, color='red', zorder=2)

    ax2.set_ylabel('scaled abundance of sample: ' + str(sample_n))
    ax2.set_xlabel('fragment length', color='blue')
    ax2.plot(single_d)
    ax2.scatter(points_x, points_y_data, color='red', zorder=2)

    ax3.set_ylabel('scores all')
    ax3.set_xlabel('fragment length', color='blue')
    ax3.hist(scores, bins=100, log=True)


@deco.log_anndata
def add_insertsize_metrics(adata,
                           bam=None,
                           fragments=None,
                           barcode_col=None,
                           barcode_tag="CB",
                           regions=None,
                           use_momentum=True,
                           use_cwt=True,
                           use_ct_cwt=True,
                           peaks_thr_mom=0.03,
                           peaks_thr_cwt=0.05,
                           peaks_thr_ct_cwt=0.01,
                           wavl_scale=35,
                           plotting=True,
                           plot_sample=0,
                           n_threads=8):
    """
    Add insert size metrics to an AnnData object.

    This function can either take a bam file or a fragments file as input.
    If both are provided, an error is raised. If none are provided, an error is raised.
    Nucleosomal signal can either calculated using the momentum method or
    the continuous wavelet transformation (CWT).

    Parameters
    ----------
    adata : anndata.AnnData
        AnnData object to add the insert size metrics to.
    bam : str, default None
        Path to bam file.
    fragments : str, default None
        Path to fragments file.
    barcode_col : str, default None
        Name of the column in the adata.obs dataframe that contains the barcodes.
    barcode_tag : str, default 'CB'
        Name of the tag in the bam file that contains the barcodes.
    regions : str, default None
        Path to bed file containing regions to calculate insert size metrics for.
    use_momentum : bool, default None
        If true, nucleosomal signal is calculated using the momentum method.
    use_cwt : bool, default True
        If true, nucleosomal signal is calculated using the CWT method.
    use_ct_cwt : bool, default True
        If true, nucleosomal signal is calculated using the custom CWT method.
    peaks_thr_mom : float, default 0.03
        Threshold for the momentum method.
    peaks_thr_cwt : float, default 0.05
        Threshold for the CWT method.
    peaks_thr_ct_cwt : float, default 0.01
        Threshold for the custom CWT method.
    wavl_scale : float, default 35
        Scale of the wavelet
    plotting : bool, default True
        If true, plots are generated.
    plot_sample : int, default 0
        Index of the sample to plot.
    n_threads : int, default 8
        Number of threads.

    Returns
    -------
    anndata.AnnData
        AnnData object with the insert size metrics added to the adata.obs dataframe.

    Raises
    ------
    ValueError:
        If bam and fragment parameter is not None.
    """

    adata_barcodes = adata.obs.index.tolist() if barcode_col is None else adata.obs[barcode_col].tolist()

    if bam is not None and fragments is not None:
        raise ValueError("Please provide either a bam file or a fragments file - not both.")

    elif bam is not None:
        count_table = tools._insertsize_from_bam(bam, barcode_tag=barcode_tag, regions=regions, barcodes=adata_barcodes)

    elif fragments is not None:
        count_table = tools._insertsize_from_fragments(fragments, barcodes=adata_barcodes)

    dist = count_table[[c for c in count_table.columns if isinstance(c, int)]]
    dists_arr = dist.to_numpy()
    dists_arr = np.nan_to_num(dists_arr)

    # scale the data
    scaled_ori = scale(dists_arr)

    # plot the densityplot of the fragment length distribution
    if plotting:
        logger.info("plotting density...")
        densities = calc_densities(scaled_ori)
        density_plot(scaled_ori, densities)

    if use_momentum:
        # prepare the data to be used for the momentum method
        # smooth the data
        logger.info("smoothing data...")
        smooth = multi_ma(dists_arr, n=2, window_size=10)
        # scale the data
        scaled = scale(smooth)

        # calculate scores using the momentum method
        logger.info("calculating scores using the momentum method...")
        momentum_scores = score_by_momentum(data=scaled,
                                            shift=80,
                                            remove=100,
                                            sample_to_inspect=plot_sample,
                                            peaks_thr=peaks_thr_mom,
                                            period=160,
                                            penalty_scale=100,
                                            plotting=plotting)

    if use_cwt:
        # calculate scores using the continues wavelet transformation
        logger.info("calculating scores using the continues wavelet transformation...")
        cwt_scores = score_by_cwt(data=scaled_ori,
                                  plot_sample=plot_sample,
                                  plotting=plotting,
                                  adapter=250,
                                  wavelet='gaus1',
                                  scales=wavl_scale,
                                  n_threads=n_threads,
                                  peaks_thr=peaks_thr_cwt,
                                  penalty_scale=100,
                                  period=160)

    if use_ct_cwt:
        logger.info("calculating scores using the custom continues wavelet transformation...")
        ct_cwt_scores = score_by_ct_cwt(data=scaled_ori,
                                        plot_wavl=False,
                                        n_threads=n_threads,
                                        peaks_thr=peaks_thr_ct_cwt,
                                        operator='bigger',
                                        plotting_mask=False,
                                        plotting_ov=True,
                                        sample=plot_sample)

    # select total inserts count and mean from count table
    inserts_table = count_table[[c for c in count_table.columns if isinstance(c, str)]]

    if use_momentum:
        inserts_table['nucleosomal_score_momentum'] = momentum_scores

    if use_cwt:
        inserts_table['nucleosomal_score_cwt'] = cwt_scores

    if use_ct_cwt:
        inserts_table['nucleosomal_score_ct_cwt'] = ct_cwt_scores

    adata.obs = adata.obs.join(inserts_table)

    if use_momentum:
        adata.obs['nucleosomal_score_momentum'] = adata.obs['nucleosomal_score_momentum'].fillna(0)

    if use_cwt:
        adata.obs['nucleosomal_score_cwt'] = adata.obs['nucleosomal_score_cwt'].fillna(0)

    if use_ct_cwt:
        adata.obs['nucleosomal_score_ct_cwt'] = adata.obs['nucleosomal_score_ct_cwt'].fillna(0)

    adata.obs.rename(columns={'insertsize_count': 'genome_counts'}, inplace=True)

    return adata


if __name__ == "__main__":

    print('This script is not meant to be run directly. Please import it into a Jupyter notebook.')
    print("Done!")
