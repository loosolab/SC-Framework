"""Tools for scATAC nucleosome analysis."""
import numpy as np
import pandas as pd
import scanpy as sc
import matplotlib.pyplot as plt
import matplotlib
import tqdm
import multiprocessing as mp
from scipy.signal import find_peaks
from scipy.signal import fftconvolve

from beartype.typing import Optional, Literal, SupportsFloat
from beartype import beartype
import numpy.typing as npt

import sctoolbox.tools as tools  # add_insertsize()
import sctoolbox.plotting as plotting  # save_figure()

import sctoolbox.utils.decorator as deco
from sctoolbox._settings import settings
logger = settings.logger


@beartype
def moving_average(series: npt.ArrayLike,
                   n: int = 10) -> npt.ArrayLike:
    """
    Move average filter to smooth out data.

    This implementation ensures that the smoothed data has no shift and
    local maxima remain at the same position.

    Parameters
    ----------
    series : npt.ArrayLike
        Array of data to be smoothed.
    n : int, default 10
        Number of steps to the left and right of the current step to be averaged.

    Returns
    -------
    npt.ArrayLike
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


@beartype
def multi_ma(series: npt.ArrayLike,
             n: int = 2,
             window_size: int = 10,
             n_threads: int = 8) -> npt.ArrayLike:
    """
    Multiprocessing wrapper for moving average filter.

    Parameters
    ----------
    series : npt.ArrayLike
        Array of data to be smoothed.
    n : int, default 2
        Number of times to apply the filter
    window_size : int, default 10
        Number of steps to the left and right of the current step to be averaged.
    n_threads : int, default 8
        Number of threads to be used for multiprocessing.

    Returns
    -------
    npt.ArrayLike
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


@beartype
def scale(series_arr: npt.ArrayLike) -> npt.ArrayLike:
    """
    Scale a series array to a range of 0 to 1.

    Parameters
    ----------
    series_arr : npt.ArrayLike
        Array of data to be scaled 1D or 2D

    Notes
    -----
    If the array is 2D, the scaling is done on axis=1.

    Returns
    -------
    npt.ArrayLike
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


@beartype
def call_peaks(data: npt.ArrayLike,
               n_threads: int = 4,
               distance: int = 50,
               width: int = 10) -> npt.ArrayLike:
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
    npt.ArrayLike
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


@beartype
def call_peaks_worker(array: npt.ArrayLike,
                      distance: int = 50,
                      width: int = 10) -> npt.ArrayLike:
    """
    Worker function for multiprocessing of scipy.signal.find_peaks.

    Parameters
    ----------
    array : npt.ArrayLike
        Array of data to find peaks in.
    distance : int, default 50
        Minimum distance between peaks.
    width : int, default 10
        Minimum width of peaks.

    Returns
    -------
    npt.ArrayLike
        Array of peaks (index of data)
    """

    peaks, _ = find_peaks(array, distance=distance, width=width)

    return peaks


@beartype
def filter_peaks(peaks: npt.ArrayLike,
                 reference: npt.ArrayLike,
                 peaks_thr: SupportsFloat,
                 operator: Literal['bigger', 'smaller'] = 'bigger') -> npt.ArrayLike:
    """
    Filter peaks based on a reference array and a threshold.

    Parameters
    ----------
    peaks : npt.ArrayLike
        Array of peaks to be filtered.
    reference : npt.ArrayLike
        Array of reference values (e.g. data were peaks were found).
    peaks_thr : float
        Threshold for filtering.
    operator : str, default 'bigger'
        Operator for filtering. Options ['bigger', 'smaller'].

    Returns
    -------
    npt.ArrayLike
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


@beartype
def distances_score(peaks: npt.ArrayLike,
                    momentum: npt.ArrayLike,
                    period: int,
                    penalty_scale: int) -> npt.ArrayLike:
    """
    Calculate a score based on the distances between peaks.

    Parameters
    ----------
    peaks : npt.ArrayLike
        Array of peaks.
    momentum : npt.ArrayLike
        Array of momentum values.
    period : int
        expected distances
    penalty_scale : int
        scale parameter for the penalty

    Returns
    -------
    npt.ArrayLike
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
        elif len(peak_list) == 1:  # if only one peak, score is the momentum at that peak divided by 100
            score = single_momentum[peak_list[0]] / 100
        elif len(peak_list) > 1:  # if more than one peak
            corrected_scores = []
            for j in range(1, len(peak_list)):  # loop over all peaks
                amplitude = single_momentum[peak_list[j - 1]] * 2  # amplitude is the momentum at the previous peak

                diff = peak_list[j] - peak_list[j - 1]  # difference between the current and previous peak
                corrected_score = amplitude - (abs(diff - period) / penalty_scale)  # corrected score
                if corrected_score < 0:
                    corrected_score = 0

                corrected_scores.append(corrected_score)  # append corrected score to list

            score = float(np.sum(np.array(corrected_scores))) + 0  # sum all corrected scores

        scores.append(score)  # append score to list

    return scores


@beartype
def score_mask(peaks: npt.ArrayLike,
               convolved_data: npt.ArrayLike,
               plot: bool = False,
               save: bool = False) -> npt.ArrayLike:
    """
    Compute a score for each sample based on the convolved data and the peaks multiplied by a score mask.

    Parameters
    ----------
    peaks : npt.ArrayLike
        Array of arrays of the peaks.
    convolved_data : npt.ArrayLike
        Array of arrays of the convolved data.
    plot : bool, default False
        If true, the score mask is plotted.
    save : bool, default False
        If true, the score mask is saved as a .png file.

    Returns
    -------
    npt.ArrayLike
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


@beartype
def build_score_mask(plot: bool = True,
                     save: bool = False,
                     mu_list: list[int] = [42, 200, 360, 550],
                     sigma_list: list[int] = [25, 35, 45, 25]) -> npt.ArrayLike:
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
    mu_list : list[int], default [42, 200, 360, 550]
        List of mu values for the Gaussian curves.
    sigma_list : list[int], default [25, 35, 45, 25]
        List of sigma values for the Gaussian curves.

    Returns
    -------
    npt.ArrayLike
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


@beartype
def gauss(x: npt.ArrayLike,
          mu: float,
          sigma: float) -> float:
    """
    Calculate the values of the Gaussian function for a given x, mu and sigma.

    Parameters
    ----------
    x : npt.ArrayLike
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

# //////////////////////////// wavelet transformation \\\\\\\\\\\\\\\\\\\\\\\\\\\\\


@beartype
def cos_wavelet(wavelength: int = 100,
                amplitude: float = 1.0,
                phase_shift: int = 0,
                mu: float = 0.0,
                sigma: float = 0.4,
                plot: bool = False,
                save: bool = False,
                figure_name: str = 'cos_wavelet') -> npt.ArrayLike:
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
    npt.ArrayLike
        Array of the wavelet.
    """

    # Scale the wavelength and sigma with the scale
    wavl_scale = int(wavelength * 1.5)
    sigma = sigma * wavl_scale  # This ensures sigma is scaled with scale
    frequency = 1.5 / wavl_scale  # This ensures the frequency is scaled with scale

    # Create an array of x values
    x = np.linspace(-wavl_scale, wavl_scale, wavl_scale * 2)

    # Compute the centered sine curve values for each x
    cosine_curve = amplitude * np.cos(2 * np.pi * frequency * x + phase_shift)

    # Compute the Gaussian values for each x
    gaussian = (1 / (sigma * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x - mu) / sigma) ** 2)
    wavelet = cosine_curve * gaussian

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


@beartype
def get_wavelets(wavelengths: list[int],
                 sigma: float = 0.4) -> list[npt.ArrayLike]:
    """
    Get a list of wavelets.

    Parameters
    ----------
    wavelengths : list[int]
        List of wavelengths for the wavelets.
    sigma : float, default 0.4
        Standard deviation of the Gaussian curve.

    Returns
    -------
    list[npt.ArrayLike]
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


@beartype
def wavelet_transformation(data: npt.ArrayLike,
                           wavelets: list[npt.ArrayLike]) -> npt.ArrayLike:
    """
    Get wavelet transformations of the fragment length distributions.

    Parameters
    ----------
    data : npt.ArrayLike
        Array of the fragment length distributions.
    wavelets : list[npt.ArrayLike]
        List of wavelets.

    Returns
    -------
    npt.ArrayLike
        Array of the wavelet transformations.
    """

    convolved_data = []
    for wavelet in wavelets:
        convolved_data.append(np.convolve(data, wavelet, mode='same'))

    convolved_data = np.array(convolved_data)

    return convolved_data


# TODO multithreading
@beartype
def wavelet_transform_fld(dists_arr: npt.ArrayLike,
                          wavelengths: Optional[list[int]] = None,
                          sigma: float = 0.4) -> npt.ArrayLike:
    """
    Get wavelet transformations of the fragment length distributions.

    Parameters
    ----------
    dists_arr : npt.ArrayLike
        Array of arrays of the fragment length distributions.
    wavelengths : list[int], default None
        List of wavelengths for the wavelets.
    sigma : float, default 0.4
        Standard deviation of the Gaussian curve.

    Returns
    -------
    npt.ArrayLike
        Array of arrays of the wavelet transformations.
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


@beartype
def custom_conv(data: npt.ArrayLike,
                wavelength: int = 150,
                sigma: float = 0.4,
                mode: str = 'convolve',
                plot_wavl: bool = False) -> npt.ArrayLike:
    """
    Get custom implementation of a wavelet transformation based convolution.

    Parameters
    ----------
    data : npt.ArrayLike
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
    npt.ArrayLike
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


@beartype
def score_by_conv(data: npt.ArrayLike,
                  wavelength: int = 150,
                  sigma: float = 0.4,
                  plot_wavl: bool = False,
                  n_threads: int = 12,
                  peaks_thr: SupportsFloat = 0.01,
                  operator: str = 'bigger',
                  plot_mask: bool = False,
                  plot_ov: bool = True,
                  save: bool = False,
                  sample: int = 0) -> npt.ArrayLike:
    """
    Get a score by a continues wavelet transformation based convolution of the distribution with a single wavelet and score mask.

    Parameters
    ----------
    data : npt.ArrayLike
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
    npt.ArrayLike
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

@beartype
def density_plot(count_table: npt.ArrayLike,
                 max_abundance: int = 600,
                 target_height: int = 1000,
                 save: bool = False,
                 figure_name: str = 'density_plot',
                 colormap: str = 'jet',
                 ax: Optional[matplotlib.axes.Axes] = None,
                 fig: Optional[matplotlib.figure.Figure] = None) -> npt.ArrayLike:
    """
    Plot the density of the fragment length distribution over all cells.

    The density is calculated by binning the abundances of the fragment lengths.

    Parameters
    ----------
    count_table : npt.ArrayLike
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
    ax : matplotlib.axes.Axes, default None
        Axes to plot on.
    fig : matplotlib.figure.Figure, default None
        Figure to plot on.

    Returns
    -------
    npt.ArrayLike
        Axes and figure of the plot.
    """
    count_table = count_table
    # handle 0,1 min/max scaled count_table
    if count_table.dtype != 'int64':
        if np.max(count_table) > 1:
            rounded = (np.round(count_table)).astype('int64')
            count_table = rounded
        else:
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
    if ax is None:
        main_plot = True
        fig, ax = plt.subplots()
    else:
        main_plot = False

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
    if fig is not None:
        fig.colorbar(im, ax=ax, label='Density (log scale)')

    if main_plot:
        if save:
            plotting._save_figure(figure_name)

        plt.show()

    figure = np.array([ax, fig])

    return figure


@beartype
def plot_wavelet_transformation(convolution: npt.ArrayLike,
                                wavelengths: npt.ArrayLike,
                                fld: Optional[npt.ArrayLike] = None,
                                save: bool = False,
                                figure_name: str = 'wavelet_transformation') -> npt.ArrayLike:
    """
    Plot the wavelet transformation of the fragment length distribution.

    If fld is not None, the fragment length distribution is plotted as well.

    Parameters
    ----------
    convolution : npt.ArrayLike
        Wavelet transformation of the fragment length distribution
    wavelengths : npt.ArrayLike
        Wavelengths of the wavelet transformation
    fld : npt.ArrayLike, default None
        Fragment length distribution
    save : bool, default False
        If true, the plot is saved.
    figure_name : str, default 'wavelet_transformation'
        Name of the figure to save.

    Returns
    -------
    npt.ArrayLike
        Axes of the plot
    """

    xmin = 0
    xmax = convolution.shape[1]
    ymin, ymax = wavelengths[0], wavelengths[-1]

    if fld is not None:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))

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

    axes = np.array([ax1, ax2])

    return axes


@beartype
def plot_custom_conv(convolved_data: npt.ArrayLike,
                     data: npt.ArrayLike,
                     peaks: npt.ArrayLike,
                     scores: npt.ArrayLike,
                     sample_n: int = 0,
                     save: bool = False,
                     figure_name: str = 'overview') -> npt.ArrayLike:
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
    sample_n : int, default 0
        Index of the sample to plot.
    save : bool, default False
        If true, the plot is saved.
    figure_name : str, default 'overview'
        Name of the figure to save.

    Returns
    -------
    npt.ArrayLike
        Axes of the plot
    """

    single_m = convolved_data[sample_n]
    single_d = data[sample_n]
    sample_peaks = peaks[sample_n]

    points_x = sample_peaks
    points_y = single_m[sample_peaks]

    points_y_data = single_d[sample_peaks]

    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 10))

    ax1.set_title("Convolution: ")
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
    ax3.set_xlabel('Score', color='blue')
    ax3.hist(scores, bins=100, log=True)

    plt.tight_layout()

    if save:
        plotting._save_figure(figure_name)

    plt.show()

    axes = np.array([ax1, ax2, ax3])

    return axes


# ///////////////////////////////////////// final wrapper \\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\

@beartype
@deco.log_anndata
def add_fld_metrics(adata: sc.AnnData,
                    bam: Optional[str] = None,
                    fragments: Optional[str] = None,
                    barcode_col: Optional[str] = None,
                    barcode_tag: str = "CB",
                    regions: Optional[str] = None,
                    peaks_thr_conv: SupportsFloat = 1,
                    wavelength: int = 150,
                    sigma: float = 0.4,
                    plot: bool = True,
                    save_plots: bool = False,
                    plot_sample: int = 0,
                    n_threads: int = 12) -> sc.AnnData:
    """
    Add insert size metrics to an AnnData object.

    This function can either take a bam file or a fragments file as input.
    If both are provided, an error is raised. If none are provided, an error is raised.
    Nucleosomal signal can either calculated using the momentum method (differential quotient) or
    the wavelet transformation based convolution method.

    Parameters
    ----------
    adata : sc.AnnData
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
    n_threads : int, default 12
        Number of threads.

    Returns
    -------
    sc.AnnData
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

    inserts_df['fld_score'] = conv_scores

    adata.obs = adata.obs.join(inserts_df)

    adata.obs['fld_score'] = adata.obs['fld_score'].fillna(0)

    # adata.obs.rename(columns={'insertsize_count': 'genome_counts'}, inplace=True)

    return adata
