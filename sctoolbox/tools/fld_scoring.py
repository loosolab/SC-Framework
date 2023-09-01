"""Tools for scATAC nucleosome analysis."""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import tqdm
import multiprocessing as mp
from scipy.signal import find_peaks
from scipy.signal import fftconvolve
from typing import Tuple
import anndata
import sctoolbox.tools as tools
import sctoolbox.plotting as plotting

import sctoolbox.utils.decorator as deco
from sctoolbox._settings import settings
logger = settings.logger


def moving_average(series, n=10) -> np.array:
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
    np.array
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


def multi_ma(series, n=2, window_size=10, n_threads=8) -> np.ndarray:
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


def scale(series_arr) -> np.ndarray:
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


# ////////////////////////////////// Peak calling \\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\


def call_peaks(data, n_threads=4, distance=50, width=10) -> np.ndarray:
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


def call_peaks_worker(array, distance=50, width=10) -> np.ndarray:
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


def filter_peaks(peaks, reference, peaks_thr, operator='bigger') -> np.ndarray:
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

# //////////////////////////////////////// Scoring \\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\


def distances_score(peaks, momentum, period, penalty_scale):
    """
    Calculate a score based on the distances between peaks.

    Parameters
    ----------
    peaks : np.ndarray
        Array of peaks.
    momentum : np.ndarray
        Array of momentum values.
    period : int
        expected distances
    penalty_scale : int
        scale parameter for the penalty

    Returns
    -------
    np.ndarray
        Array of scores
    """

    scores = []
    # loop over all cells
    for i in range(len(peaks)):
        # get peaks and momentum for single cell
        peak_list = peaks[i]
        single_momentum = momentum[i]

        # calculate score
        if len(peak_list) == 0:
            score = 0
        elif len(peak_list) == 1: # if only one peak, score is the momentum at that peak divided by 100
            score = single_momentum[peak_list[0]] / 100
        elif len(peak_list) > 1: # if more than one peak
            corrected_scores = []
            for j in range(1, len(peak_list)): # loop over all peaks
                amplitude = single_momentum[peak_list[j - 1]] * 2 # amplitude is the momentum at the previous peak

                diff = peak_list[j] - peak_list[j - 1] # difference between the current and previous peak
                corrected_score = amplitude - (abs(diff - period) / penalty_scale) # corrected score
                if corrected_score < 0:
                    corrected_score = 0

                corrected_scores.append(corrected_score) # append corrected score to list

            score = float(np.sum(np.array(corrected_scores))) + 0 # sum all corrected scores

        scores.append(score) # append score to list

    return scores


def score_mask(peaks, convolved_data, plot=False, save=False):
    """
    compute a score for each sample based on the convolved data and the peaks multiplied by a score mask.

    Parameters
    ----------
    peaks : np.array
        Array of arrays of the peaks.
    convolved_data : np.array
        Array of arrays of the convolved data.
    plot : bool, default False
        If true, the score mask is plotted.
    save : bool, default False
        If true, the score mask is saved as a .png file.

    Returns
    -------
    np.array
        Array of scores for each sample
    """

    # build score mask
    score_mask = build_score_mask(plot=plot, save=save)

    scores = []
    # loop over all cells
    for i, peak_list in enumerate(peaks):
        conv = convolved_data[i]

        # calculate score
        # if no peaks, score is 0
        if len(peak_list) == 0:
            score = 0
        # if only one peak, score is the convolved data at that peak multiplied by the score mask
        elif len(peak_list) == 1:
            score = conv[peak_list[0]] * score_mask[0][peak_list[0]]

        # if more than one peak, score is the sum of the convolved data at each peak multiplied by the score mask
        elif len(peak_list) > 1:
            score = 0
            for j in range(1, len(peak_list)):
                if j <= 3:
                    score += conv[peak_list[j]] * score_mask[j][peak_list[j]]
                else:
                    score += conv[peak_list[j]] * score_mask[3][peak_list[j]]

        scores.append(score)

    return np.array(scores)


def build_score_mask(plot=True,
                     save=False,
                     mu_list=[50, 200, 350, 550],
                     sigma_list=[25, 35, 45, 25]) -> np.array:
    """
    Build a score mask for the score by custom continuous wavelet transformation.

    Mask is a sum of 4 Gaussian curves with mu and sigma specified
    for the expected peak positions and deviations.

    Parameters
    ----------
    plot : bool, default True
        If true, the score mask is plotted.
    save : bool, default False
        If true, the score mask is saved as png
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

    if plot:
        fig, ax = plt.subplots()

        ax.plot(gaussians[0])
        ax.plot(gaussians[1])
        ax.plot(gaussians[2])
        ax.plot(gaussians[3])
        ax.set_title('Score-Mask')
        ax.set_xlabel('Position')
        ax.set_ylabel('Scoring')

        if save:
            plotting._save_figure('score_mask')

    return gaussians


def gauss(x, mu, sigma) -> float:
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


# //////////////////////////// Differential Quotient (Momentum) \\\\\\\\\\\\\\\\\\\\\\\\\\\\\

def momentum_diff(data, remove=150, shift=80, smooth=True) -> Tuple[np.array, np.array, np.array]:
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
    Tuple[np.array, np.array, np.array]
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
                      plot=True,
                      score_by_distances=True,
                      score_by_mask=False,
                      save=False,
                      figure_name='MOM_Scores') -> np.ndarray:
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
    plot : bool, default True
        Plot the momentum and the peaks of the reference sample.
    score_by_distances : bool, default True
        Score cells based on the distance between peaks.
    score_by_mask : bool, default False
        Score cells based on a mask.
    save : bool, default False
        Save the figure.
    figure_name : str, default 'MOM_Scores'
        Name of the figure to save.

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

    if plot:
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

    if score_by_distances:
        scores = distances_score(peaks, momentum, period, penalty_scale)
    if score_by_mask:
        scores = score_mask(peaks, momentum, plot=plot, save=save)

    scores = np.array(scores)

    if plot:
        fig, ax = plt.subplots()
        ax.hist(np.sort(scores), bins=100, log=True)
        ax.set_title('Scores')
        ax.set_xlabel('Score')
        ax.set_ylabel('Number of cells')

        if save:
            plotting._save_figure(figure_name)

    return scores

# //////////////////////////// wavelet transformation \\\\\\\\\\\\\\\\\\\\\\\\\\\\\


def cos_wavelet(wavelength=100,
                amplitude=1.0,
                phase_shift=0,
                mu=0.0,
                sigma=0.4,
                plot=False,
                save=False,
                figure_name='cos_wavelet') -> np.ndarray:
    """
    Build a cosine wavelet. The wavelet is a cosine curve multiplied by a Gaussian curve.

    Parameters
    ----------
    wavelength : int, default 100
        Wavelength of the cosine curve.
    amplitude : float, default 1.0
        Amplitude of the cosine curve.
    phase_shift : int, default 0
        Phase shift of the cosine curve.
    mu : float, default 0.0
        Mean of the Gaussian curve.
    sigma : float, default 0.4
        Standard deviation of the Gaussian curve.
    plot : bool, default False
        Plot the wavelet.
    save : bool, default False
        Save the plot.
    figure_name : str, default 'cos_wavelet'
        Name of the figure to save.

    Returns
    -------
    np.ndarray
        Array of the wavelet.
    """

    # Scale the wavelength and sigma with the scale
    wavl_scale = int(wavelength * 1.5)
    sigma = sigma * wavl_scale  # This ensures sigma is scaled with scale
    frequency = 1.5 / wavl_scale # This ensures the frequency is scaled with scale

    # Create an array of x values
    x = np.linspace(-wavl_scale, wavl_scale, wavl_scale * 2)

    # Compute the centered sine curve values for each x
    sine_curve = amplitude * np.cos(2 * np.pi * frequency * x + phase_shift)

    # Compute the Gaussian values for each x
    gaussian = (1 / (sigma * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x - mu) / sigma) ** 2)
    wavelet = sine_curve * gaussian

    if plot:
        fig, ax = plt.subplots()
        ax.plot(wavelet)
        ax.set_title('Wavelet')
        ax.set_xlabel('Interval')
        ax.set_ylabel('Amplitude')

        if save:
            plotting._save_figure(figure_name)

        # Optionally, to show the figure
        plt.show()

    return wavelet


def get_wavelets(wavelengths, sigma=0.4) -> list:
    """
    Get a list of wavelets.

    Parameters
    ----------
    wavelengths : list
        List of wavelengths for the wavelets.
    sigma : float, default 0.4
        Standard deviation of the Gaussian curve.

    Returns
    -------
    list
        List of wavelets.
    """
    wavelets = []
    for wavelength in wavelengths:
        wavelet = cos_wavelet(wavelength=wavelength,
                              amplitude=1.0,
                              phase_shift=0,
                              mu=0.0,
                              sigma=sigma,
                              plot=False)
        wavelets.append(wavelet)

    return wavelets


def wavelet_transformation(data, wavelets) -> np.ndarray:
    """
    Get wavelet transformations of the fragment length distributions.

    Parameters
    ----------
    data : np.array
        Array of the fragment length distributions.
    wavelets : list
        List of wavelets.

    Returns
    -------
    np.ndarray
        Array of the wavelet transformations.
    """

    convolved_data = []
    for wavelet in wavelets:
        convolved_data.append(np.convolve(data, wavelet, mode='same'))

    convolved_data = np.array(convolved_data)

    return convolved_data


# TODO multithreading
def wavelet_transform_fld(dists_arr, wavelengths=None, sigma=0.4) -> np.ndarray:
    """
    Get wavelet transformations of the fragment length distributions.

    Parameters
    ----------
    dists_arr : np.array
        Array of arrays of the fragment length distributions.
    wavelengths : list
        List of wavelengths for the wavelets.
    sigma : float, default 0.4
        Standard deviation of the Gaussian curve.
    """

    # Set default wavelengths
    if wavelengths is None:
        wavelengths = np.arange(5, 250, 5).astype(int)

    # Get wavelets
    wavelets = get_wavelets(wavelengths, sigma=sigma)

    dataset_convolution = []
    # Process each cell with the wavelet transformation
    for cell in tqdm(dists_arr, desc="Processing cells"):
        dataset_convolution.append(wavelet_transformation(cell, wavelets))

    dataset_convolution = np.array(dataset_convolution)

    return dataset_convolution


def custom_conv(data, wavelength=150, sigma=0.4, mode='convolve', plot_wavl=False) -> np.array:
    """
    Get custom implementation of a wavelet transformation based convolution.

    Parameters
    ----------
    data : np.array
        Array of arrays of the fragment length distributions.
    wavelength : int, default 150
        Wavelength of the wavelet.
    sigma : float, default 0.4
        Standard deviation of the Gaussian curve.
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
    wavelet = cos_wavelet(wavelength=wavelength,
                amplitude=1.0,
                phase_shift=0,
                mu=0.0,
                sigma=sigma,
                plot=plot_wavl)

    # convolve with the data
    convolved_data = []
    for cell in data:
        if mode == 'convolve':
            convolved_data.append(np.convolve(cell, wavelet, mode='same'))
        elif mode == 'fftconvolve':
            convolved_data.append(fftconvolve(data, wavelet, mode='same'))

    return np.array(convolved_data)


def score_by_conv(data,
                  wavelength=150,
                  sigma=0.4,
                  plot_wavl=False,
                  n_threads=12,
                  peaks_thr=0.01,
                  operator='bigger',
                  plot_mask=False,
                  plot_ov=True,
                  save=False,
                  sample=0) -> np.array:
    """
    Get a score by a continues wavelet transformation based convolution of the distribution with a single wavelet and score mask.

    Parameters
    ----------
    data : np.array
        Array of arrays of the fragment length distributions.
    wavelength : int, default 150
        Wavelength of the wavelet.
    sigma : float, default 0.4
        Standard deviation of the Gaussian curve.
    plot_wavl : bool, default False
        If true, the wavelet is plotted.
    n_threads : int, default 12
        Number of threads to use for the peak calling.
    peaks_thr : float, default 0.01
        Threshold for the peak calling.
    operator : str, default 'bigger'
        Operator to use for the peak calling. Either 'bigger' or 'smaller'.
    plot_mask : bool, default False
        If true, the score mask is plotted.
    plot_ov : bool, default True
        If true, the overlay of the score mask and the convolved data is plotted.
    save : bool, default False
        If true, the figure is saved.
    sample : int, default 0
        Index of the sample to plot.

    Returns
    -------
    np.array
        Array of scores for each sample
    """

    convolved_data = custom_conv(data, wavelength=wavelength, sigma=sigma, plot_wavl=plot_wavl)

    peaks = call_peaks(convolved_data, n_threads=n_threads)

    filtered_peaks = filter_peaks(peaks, reference=convolved_data, peaks_thr=peaks_thr, operator=operator)

    scores = score_mask(peaks, convolved_data, plot=plot_mask)

    if plot_ov:
        plot_custom_conv(convolved_data, data, filtered_peaks, scores=scores, sample_n=sample, save=save, figure_name='convolution_overview')

    return scores


# //////////////////////////////// plotting \\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\

def density_plot(count_table, max_abundance=600, target_height=1000, save=False, figure_name='density_plot', colormap='jet') -> matplotlib.axes.Axes:
    """
    Plot the density of the fragment length distribution over all cells.

    The density is calculated by binning the abundances of the fragment lengths.

    Parameters
    ----------
    count_table : np.ndarray
        Array of arrays of the fragment length distributions
    max_abundance : int, default 600
        Maximal abundance of a fragment length of a cell (for better visability)
    target_height : int, default 1000
        Target height of the plot
    save : bool, default False
        If true, the plot is saved.
    figure_name : str, default 'density_plot'
        Name of the figure to save.
    colormap : str, default 'jet'
        Color map of the plot.

    Returns
    -------
    matplotlib.axes.Axes
        Axes of the plot
    """
    count_table = count_table
    # handle 0,1 min/max scaled count_table
    if count_table.dtype != 'int64':
        if np.max(count_table) > 1:
            rounded = (np.round(count_table)).astype('int64')
            count_table = rounded
        else:
            #count_table = unscale(count_table)
            count_table = (count_table * 1000).astype('int64')
    # get the maximal abundance of a fragment length over all cells
    max_value = np.max(np.around(count_table).astype(int))
    # Init empty densities list
    densities = []
    # loop over all fragment lengths from 0 to 1000
    for i in range(0, len(count_table[0])):
        column = count_table[:, i]
        # round abundances to be integers, that they are countable
        rounded = np.around(column).astype(int)
        # count the abundance of the abundances with boundaries 0 to maximal abundance
        gradient = np.bincount(rounded, minlength=max_value + 1)
        densities.append(gradient)
    densities = np.array(densities)

    # Log normalization + 1 to avoid log(0)
    densities_log = np.log1p(densities)

    # Transpose the matrix
    densities = densities_log.T

    # get the section of interest
    densities = densities[:max_abundance]

    # calculate the mean of the FLD
    mean = count_table.sum(axis=0) / len(count_table)

    # Stretch or compress densities' y-axis to the target height
    num_rows = densities.shape[0]
    old_y = np.linspace(0, num_rows - 1, num_rows)
    new_y = np.linspace(0, num_rows - 1, 1000)

    # Interpolate the densities along the y-axis
    densities_interpolated = np.array([np.interp(new_y, old_y, densities[:, i]) for i in range(densities.shape[1])]).T

    # scaling factor for mean
    scaling_factor = len(new_y) / len(old_y)

    # Apply the scaling factor to the mean values
    mean_interpolated = mean * scaling_factor

    # Initialize subplots
    fig, ax = plt.subplots()

    # Display the image
    im = ax.imshow(densities_interpolated, aspect='auto', origin="lower", cmap=colormap)

    # Plot additional data
    ax.plot(mean_interpolated, color="red", markersize=1)

    # Set labels and title
    ax.set_title('Fragment Length Density Plot')
    ax.set_xlabel('Fragment Length', color='blue')
    ax.set_ylabel('Number of Fragments', color='blue')

    # Adjust y-ticks to show original scale
    ax.set_yticks(np.linspace(0, target_height - 1, 6))
    ax.set_yticklabels(np.linspace(0, num_rows - 1, 6).astype(int))

    # Add colorbar to the plot
    fig.colorbar(im, ax=ax, label='Density (log scale)')

    if save:
        plotting._save_figure(figure_name)

    plt.show()

    return ax


def plot_wavelet_transformation(convolution,
                                wavelengths,
                                fld=None,
                                save=False,
                                figure_name='wavelet_transformation'):
    """
    Plot the wavelet transformation of the fragment length distribution.
    if fld is not None, the fragment length distribution is plotted as well.

    Parameters
    ----------
    convolution : np.ndarray
        Wavelet transformation of the fragment length distribution
    wavelengths : np.ndarray
        Wavelengths of the wavelet transformation
    fld : np.ndarray, default None
        Fragment length distribution
    save : bool, default False
        If true, the plot is saved.
    figure_name : str, default 'wavelet_transformation'
        Name of the figure to save.

    Returns
    -------
    matplotlib.axes.Axes
        Axes of the plot
    """

    xmin = 0
    xmax = convolution.shape[1]
    ymin, ymax = wavelengths[0], wavelengths[-1]

    if fld is not None:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10,5))

        ax1.set_title('Fragment Length Distribution')
        ax1.set_xlabel('Fragment Length (bp)')
        ax1.set_ylabel('Number of Fragments')
        ax1.plot(fld)

        img = ax2.imshow(convolution, aspect='auto', cmap='jet', extent=[xmin, xmax, ymax, ymin])
        # Adding a colorbar to ax2
        cbar = fig.colorbar(img, ax=ax2)
        cbar.set_label('Fit')

        ax2.set_title('Wavelet Transformation')
        ax2.set_xlabel('Fragment Length (bp)')
        ax2.set_ylabel('Wavelength (bp)')
        ax2.grid(color='white', linestyle='--', linewidth=0.5)

        plt.tight_layout()

    else:
        # Create a figure and set the size
        plt.imshow(convolution, aspect='auto', cmap='jet', extent=[xmin, xmax, ymax, ymin])
        plt.colorbar(label='Fit')
        plt.xlabel('Fragment Length (bp)')
        plt.ylabel('Wavelength (bp)')
        plt.title('Wavelet Transformation')

        plt.grid(color='white', linestyle='--', linewidth=0.5)

    # Save the figure
    if save:
        plotting._save_figure(figure_name)

    plt.show()


def plot_single_momentum_ov(peaks,
                            momentum,
                            data,
                            shift_l,
                            shift_r,
                            sample_n=0,
                            shift=80,
                            remove=150,
                            save=False,
                            figure_name='momentum_overview') -> Tuple[plt.figure, list[matplotlib.axes.Axes]]:
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
    save : bool, default False
        If true, the plot is saved.
    figure_name : str, default 'momentum_overview'
        Name of the figure to save.

    Returns
    -------
    Tuple[plt.figure, list[matplotlib.axes.Axes]]
        Tuple at index 1: matplotlib figure
        Tuple at index 2: list (legnth 3) of matplotlib.axes.Axes objects
    """
    single_m = momentum[sample_n]
    single_d = data[sample_n]
    sample_peaks = peaks[sample_n]

    a = shift_l[sample_n]
    b = shift_r[sample_n]

    points_x = sample_peaks
    points_y = single_m[sample_peaks]

    points_ori_x = points_x + remove
    points_ori_y = single_d[points_ori_x]

    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 10))
    ax1.set_title('Shifted Distributions a and b')
    ax1.set_ylabel('Number of Fragments')
    ax1.set_xlabel('Fragment Length - ' + str(remove) + 'bp-shift', color='blue')
    ax1.plot(a)
    ax1.plot(b)

    ax2.set_title('Momentum')
    ax2.set_ylabel('Momentum')
    ax2.set_xlabel('Fragment Length - ' + str(remove) + 'bp-shift', color='blue')
    ax2.plot(single_m)
    ax2.scatter(points_x, points_y, color='red', zorder=2)

    ax3.set_title('FLD with Peaks')
    ax3.set_ylabel('Number of Fragments')
    ax3.set_xlabel('Fragment Length', color='blue')
    ax3.plot(single_d)
    ax3.scatter(points_ori_x, points_ori_y, color='red', zorder=2)

    plt.tight_layout()

    if save:
        plotting._save_figure(figure_name)

    plt.show()

    return fig, [ax1, ax2, ax3]


def plot_custom_conv(convolved_data, data, peaks, scores, sample_n=0, save=False, figure_name='momentum_overview') -> list[matplotlib.axes.Axes]:
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
    save : bool, default False
        If true, the plot is saved.
    figure_name : str, default 'momentum_overview'
        Name of the figure to save.
    """

    single_m = convolved_data[sample_n]
    single_d = data[sample_n]
    sample_peaks = peaks[sample_n]

    points_x = sample_peaks
    points_y = single_m[sample_peaks]

    points_y_data = single_d[sample_peaks]

    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 10))

    ax1.set_title("Convolution: " + str(sample_n) + " - Sample")
    ax1.set_ylabel('Convolution Fit')
    ax1.set_xlabel('Fragment Length', color='blue')
    ax1.plot(single_m)
    ax1.scatter(points_x, points_y, color='red', zorder=2)

    ax2.set_title('Fragment Length Distribution')
    ax2.set_ylabel('Number of Fragments')
    ax2.set_xlabel('Fragment Length', color='blue')
    ax2.plot(single_d)
    ax2.scatter(points_x, points_y_data, color='red', zorder=2)

    ax3.set_title('Scores')
    ax3.set_ylabel('Number of Cells')
    ax3.set_xlabel('Fragment Length', color='blue')
    ax3.hist(scores, bins=100, log=True)

    plt.tight_layout()

    if save:
        plotting._save_figure(figure_name)

    plt.show()

    return ax1, ax2, ax3

# ///////////////////////////////////////// final wrapper \\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\

@deco.log_anndata
def add_fld_metrics(adata,
                    bam=None,
                    fragments=None,
                    barcode_col=None,
                    barcode_tag="CB",
                    regions=None,
                    use_momentum=True,
                    use_conv=True,
                    peaks_thr_mom=3,
                    peaks_thr_conv=1,
                    wavelength=150,
                    sigma=0.4,
                    plot=True,
                    save_plots=False,
                    plot_sample=0,
                    n_threads=12) -> anndata.AnnData:
    """
    Add insert size metrics to an AnnData object.

    This function can either take a bam file or a fragments file as input.
    If both are provided, an error is raised. If none are provided, an error is raised.
    Nucleosomal signal can either calculated using the momentum method (differential quotient) or
    the wavelet transformation based convolution method.

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
    use_conv : bool, default None
        If true, nucleosomal signal is calculated using the convolution method.
    peaks_thr_mom : float, default 3
        Threshold for the momentum method.
    peaks_thr_conv : float, default 1
        Threshold for the convolution method.
    wavelength : int, default 150
        Wavelength for the convolution method.
    sigma : float, default 0.4
        Sigma for the convolution method.
    plot : bool, default True
        If true, plots are generated.
    save_plots : bool, default False
        If true, plots are saved.
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
    # remove all rows containing only 0
    filtered_dist = dist.loc[~(dist == 0).all(axis=1)]
    # extract available barcodes
    barcodes = filtered_dist.index.to_numpy()
    # get numpy array for calculation
    dists_arr = filtered_dist.to_numpy()

    # plot the densityplot of the fragment length distribution
    if plot:
        logger.info("plotting density...")
        density_plot(dists_arr, max_abundance=600, save=save_plots)

    if use_momentum:
        # prepare the data to be used for the momentum method
        # smooth the data
        logger.info("smoothing data...")
        smooth = multi_ma(dists_arr, n=2, window_size=10)

        # calculate scores using the momentum method
        logger.info("calculating scores using the momentum method...")
        momentum_scores = score_by_momentum(data=smooth,
                                            shift=80,
                                            remove=0,
                                            sample_to_inspect=plot_sample,
                                            peaks_thr=peaks_thr_mom,
                                            period=160,
                                            penalty_scale=100,
                                            plot=plot,
                                            save=save_plots)

    if use_conv:
        logger.info("calculating scores using the custom continues wavelet transformation...")
        conv_scores = score_by_conv(data=dists_arr,
                                    wavelength=wavelength,
                                    sigma=sigma,
                                    plot_wavl=plot,
                                    n_threads=n_threads,
                                    peaks_thr=peaks_thr_conv,
                                    operator='bigger',
                                    plot_mask=plot,
                                    plot_ov=plot,
                                    save=save_plots,
                                    sample=plot_sample)

    # create a dataframe with the scores and match the barcodes
    inserts_df = pd.DataFrame(index=barcodes)

    if use_momentum:
        inserts_df['fld_score_momentum'] = momentum_scores

    if use_conv:
        inserts_df['fld_score_conv'] = conv_scores

    adata.obs = adata.obs.join(inserts_df)

    if use_momentum:
        adata.obs['fld_score_momentum'] = adata.obs['fld_score_momentum'].fillna(0)

    if use_conv:
        adata.obs['fld_score_conv'] = adata.obs['fld_score_conv'].fillna(0)

    # adata.obs.rename(columns={'insertsize_count': 'genome_counts'}, inplace=True)

    return adata
