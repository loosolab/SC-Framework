# Larger parts of the code used here come from the repository of the Applied Data Analysis module 2022/2023

# Import libraries
import numpy as np
import logging
import pandas as pd
import math
import time


def add_chromatin_conditions(adata, fragment_file, bins=30, penalty=200):
    """
    This method adds the chromatin conditions to the anndata object.
    :param adata: anndata object.
    :param fragment_file: path to the fragment file.
    :param bins: resolution of the calculated distribution further calculations based on.
    :param penalty: penalty for the score calculation.
    """

    # load data
    df = load_data(fragment_file, bins=bins, penalty=penalty)

    # add data to anndata object
    adata = add_df_to_adata(adata, df)

    return adata


def load_data(path: str, bins=30, penalty=200):
    """
    This method creates an dataframe with cell barcode as index and colums
    for fragment lengths ('Fragments'), fragment count ('Fragment-Count'),
    fragment length distribution ('Distribution'), maxima position ('Maxima'),
    maxima count ('Maxima-Count') und einen quality score ('Score').
    :param path: path to a .bed file.
    :param bins: resolution of the calculated distribution further calculations based on.
    :return: dataframe with the colums specified above.
    """

    time_point_1 = time.time()

    # loding fragment_file and creating dataframe with fragments, fragment_count, mean and median
    print('loading fragments...')
    fragments = read_fragment_file(path)
    df = create_dataframe(fragments)
    df['Fragment-Count'] = [len(x) for x in df['Fragments']]

    time_point_2 = time.time()
    print("Time for loading fragments: ", str(time_point_2 - time_point_1))

    # calculate distribution in predifined bins and add it to the dataframe
    print('calculate distribution...')
    df['Distribution'] = get_distribution(df, bins=bins)

    time_point_3 = time.time()
    print("Time for calculating distribution: ", str(time_point_3 - time_point_2))

    # calculate maxima and their count and add them to the dataframe
    print('calculate maxima...')
    df['Maxima'] = get_maxima(df)
    df['Maxima-Count'] = [len(x) for x in df['Maxima']]

    time_point_4 = time.time()
    print("Time for calculating maxima: ", str(time_point_4 - time_point_3))

    # calculate score and add it to the dataframe
    print('calculate score...')
    get_score(df, bins=bins, penalty=penalty)

    time_point_5 = time.time()
    print("Time for calculating score: ", str(time_point_5 - time_point_4))

    return df


def read_fragment_file(abs_path: str):
    """
    This method reads a fragment file (.bed) and returns
    a dictionary with the cellbarcodes as keys and the
    computed fragment lengths as the corresponding values.
    The fragment lengths are stored as a list.
    Returns None when file extension is not .bed.
    :param abs_path: (absolute) path to the .bed file.
    :return: dictionary (keys=cellbarcode/values=list of fragment lengths)
    """

    # Check if the file extension is .bed
    if abs_path[-4:] != ".bed":
        logging.warning("provided file does not have '.bed' extension:\t"+abs_path)
        return None

    # Get file from disk
    fragment_file = open(abs_path, "r")

    # Create dictionary
    frag_dictionary = {}

    # Read file
    for line in fragment_file:

        # Split current line
        line_values = line.split()

        # Get cell barcode
        cellbarcode = line_values[3]

        # Get both fragments
        start = int(line_values[1])
        stop = int(line_values[2])

        # Check if cellbarcode key is in dictionary
        if cellbarcode in frag_dictionary:
            # Append list of fragment lengths
            frag_dictionary[cellbarcode].append(calculate_fragment_length(start, stop))
        else:
            # Create new Key/Value (Cellbarcode/Fragment length list)
            frag_dictionary[cellbarcode] = [calculate_fragment_length(start, stop)]

    # Close open access to file
    fragment_file.close()

    # Return dictionary
    return frag_dictionary


def create_dataframe(frag_dictionary: dict, tissue=""):
    """
    This method creates a new dataframe object with the values
    taken from the input dictionary. The resulting object only
    contains the cellbarcode from the dictionary as an index and
    the corresponding mean/median values as well as the fragment lists
    from the provided frag_dictionary.
    :param frag_dictionary: dictionary from which the object is created from.
    :param tissue: string of the belonging tissue.
    :return: dataframe object with cellbarcodes, means, medians and fragment length lists.
    """

    # Create empty dictionary
    mean_median_dictionary = {}

    # Iterate over keys in frag_dictionary
    if tissue == "":
        for cellbarcode in frag_dictionary:
            mean_median_dictionary[cellbarcode] = {}
            mean_median_dictionary[cellbarcode]["Mean-Fragment-Length"] = calculate_mean(frag_dictionary[cellbarcode])
            mean_median_dictionary[cellbarcode]["Median-Fragment-Length"] = calculate_median(frag_dictionary[cellbarcode])
            mean_median_dictionary[cellbarcode]["Fragments"] = frag_dictionary[cellbarcode]
    else:
        for cellbarcode in frag_dictionary:
            mean_median_dictionary[tissue+"+"+cellbarcode] = {}
            mean_median_dictionary[tissue+"+"+cellbarcode]["Mean-Fragment-Length"] = calculate_mean(frag_dictionary[cellbarcode])
            mean_median_dictionary[tissue+"+"+cellbarcode]["Median-Fragment-Length"] = calculate_median(frag_dictionary[cellbarcode])
            mean_median_dictionary[cellbarcode]["Fragments"] = frag_dictionary[cellbarcode]

    # Transform dictionary into dataframe
    data_frame = pd.DataFrame(mean_median_dictionary).T

    # Return dataframe
    return data_frame


def get_distribution(df, bins=30):
    """
    This method scales a given dataframe with a column of fragment lengths and binnes them
    to get the y-values of the distribution.
    :param df: dataframe with a column for fragment lengths.
    :param bins: resolution of the calculated distribution further calculations based on.
    """

    # calculate list of bin_indexes over the range of fragment lengths
    min_value = min(df['Fragments'].apply(min))
    max_value = max(df['Fragments'].apply(max))
    value_range = max_value - min_value
    bin_scale = value_range / bins
    bin_index = [x for x in np.arange(min_value, max_value, bin_scale)]

    # digitize fragment length vor every dataframe index into predefined bins
    distribution = []
    for i in df['Fragments']:
        inds = np.digitize(i, bin_index)

        # count the elements in every bin to get the y-value every point in the distribution
        y_values = []
        for x in range(len(bin_index)):
            y_values.append(np.count_nonzero(inds == x + 1))
        distribution.append(y_values)
    return distribution


def get_maxima(df, distribution='Distribution'):
    """
    This method calculates the local maxima for every index in a dataframe.
    :param df: dataframe with a kind of distribution column
    :param distribution: column in a dataframe with arrays of values to calculate the maximas from
    """

    # itterate over whole dataframe
    maxima = []
    for i in df[distribution]:

        # filter empty cells in the dataframe
        if i is np.nan:
            maxima.append(np.nan)
        else:
            # calculate local maxima
            maxima.append(calculate_maxima(i))

    return maxima


def get_score(df, bins=30, penalty=200):
    """
    Computes a score for each row of the dataframe and adds
    a corresponding column "Score" containing each individual score.
    The dataframe needs to have a column "Distribution" with
    lists of numerical values to enable a reliable scoring.
    If a score of a row could not be calculated, the value in
    the new cell will be NaN.
    :param dataframe: the dataframe to be extended
    """

    # Check if column "Distribution" exists
    if "Distribution" not in df.columns:
        print("Dataframe does not have a column Distribution!")
        return

    # Check if column "Score" exists, if not add it
    if "nucleosomal-score" not in df.columns:
        df["nucleosomal-score"] = np.NaN

    # Calculate bin_size
    min_frag = min(df['Fragments'].apply(min))
    max_frag = max(df['Fragments'].apply(max))
    bin_size = (max_frag - min_frag) / bins

    # Create empty score list
    score_list = []

    # Iterate over dataframe rows
    for index, frame in df.iterrows():

        # Compute Score for this row
        score = calculate_score(calculate_maxima(df["Distribution"][index]), min_frag, bin_size, bins=bins)

        # add penalty for fragment count
        if df['Fragment-Count'][index] < penalty:
            score = score + (penalty - df['Fragment-Count'][index])
        score = score * (1 / np.log(df['Fragment-Count'][index]))

        # Append score list with computed value
        score_list.append(score)

    # Set the values of the column "Score" in the dataframe
    df["nucleosomal-score"] = score_list
    return score_list


def add_df_to_adata(adata, df, remove_tmp=True):
    """
    This method adds a dataframe to an anndata object.
    :param adata: anndata object to add the dataframe to.
    :param df: dataframe to be added.
    """
    if remove_tmp:
        df.pop('Maxima')
        df.pop('Maxima-Count')
    # add dataframe to anndata object
    adata.obs = adata.obs.join(df)
    adata.obs = adata.obs.fillna(0)

    # convert tables to string
    adata.obs['Fragments'] = adata.obs['Fragments'].astype(str)
    adata.obs['Distribution'] = adata.obs['Distribution'].astype(str)
    adata.obs['Maxima'] = adata.obs['Maxima'].astype(str)

    return adata
############################################################################################
# calculation
############################################################################################

def calculate_fragment_length(start: int, stop: int):
    """
    This method simply calculates the fragment length of the
    two specified start/stop base pairs.
    :param start: start position of the fragment (bp).
    :param stop: stop position of the fragment (bp).
    :return: length of the fragment.
    """
    return abs(start-stop)

def calculate_maxima(value_list):
    """
    This method computes all local maxima in a given list of numerical
    values using a "sliding window" to filter false-positive local maxima
    resulting from noisy data. This window has a fixed size of 5.
    A list of all indices of such local maxima found in the provided
    value list is then returned.
    :param value_list: list of numerical values
    :return: list of indices of all local maxima
    """

    # Create local maxima list
    local_maxima = []

    # Store length of distribution parameter
    distr_len = len(value_list)

    # Iterate over distribution
    for index, count in enumerate(value_list):

        # Create "empty" sliding window
        window = np.array([None, None, None, None, None])

        # Set window
        if index - 2 >= 0:
            window[0] = value_list[index - 2]
        if index - 1 >= 0:
            window[1] = value_list[index - 1]
        if index >= 0:
            window[2] = value_list[index]
        if index + 1 <= distr_len - 1:
            window[3] = value_list[index + 1]
        if index + 2 <= distr_len - 1:
            window[4] = value_list[index + 2]

        # Check if value in window is local maxima
        poss_maxima = window[2]
        is_maxima = False
        # If window[1] is None, then window[0] is also None
        if window[1] is None:
            if poss_maxima > window[3] and poss_maxima > window[4]:
                is_maxima = True
        # Check if only window[0] is None
        elif window[0] is None:
            if window[1] < poss_maxima and poss_maxima > window[3] and poss_maxima > window[4]:
                is_maxima = True
        # Check if window[4] is not None, then window[3] must also be not None
        # This is the usual case
        elif window[4] is not None:
            if window[0] < poss_maxima and window[1] < poss_maxima and poss_maxima > window[3] and poss_maxima > window[4]:
                is_maxima = True

        # Check if possible maxima is a real local maxima
        # If yes, then append it to local_maxima list
        if is_maxima:
            local_maxima.append(index)

    # Return list of all local maxima
    return local_maxima


def calculate_score(peaks, min_frag, bin_size, bins=30, period=160):
    """
    Computes the average difference between the provided peak indices and
    the specified period. The calculation is divided into three cases:

    1. The peak list is empty:
        -> Return (positive) infinity
    2. The peak list contains just one peak:
        -> Return absolute difference between peak and period+min_frag
    3. The peak list contains more than one peak:
        -> Return average difference between all peaks w.r.t. specified period

    Hence, the closer this value conforms to 0 the better, making 0 the best possible
    value.
    :param peaks: list of peaks indices
    :param period: period in which the peaks should occur
    :param min_frag: length of the smallest fragment
    :param bin_size: size of each bin
    :param bins: number of bins the peak indices will be mapped to
    """

    # Create list for later computation
    temp = range(0, bins + 1)

    # Create empty list for computed locations
    peak_bins = []


    for i in temp:
        if temp.index(i) in peaks:
            peak_bins.append(i * bin_size + min_frag)

    # Compare Difference
    diff = 0

    # Check for 0 peaks
    if len(peaks) == 0:
        return np.inf

    # Check for 1 peak
    if len(peaks) == 1:
        peak_diff = abs(peak_bins[0] - (period + min_frag))
        return round(peak_diff, 2)

    # More than 1 peak
    for index in range(len(peak_bins) - 1):
        bin_diff = abs(peak_bins[index] - peak_bins[index + 1])
        diff += abs(bin_diff - period)

    diff /= len(peaks) - 1

    return round(diff, 2)


def calculate_mean(value_list: list, decimal_places=2):
    """
    This method computes the mean value out of a list
    of values. The result is rounded to two decimal places.
    :param value_list: list of values.
    :param decimal_places: rounded to these decimal places.
    :return: mean of the given list of values.
    """

    # Create variable to be returned
    mean = 0

    # Iterate over value list
    for length in value_list:
        mean += length

    mean = round(mean / len(value_list), decimal_places)

    return mean


def calculate_median(value_list: list):
    """
    This method computes the median value out of a list
    of values.
    :param value_list: list of values.
    :return: median of the given list of values.
    """

    # Create sorted copy of list
    sorted_list = sorted(value_list)

    # Check if list length is even or odd
    if is_even(sorted_list) is False:
        # return middle value
        middle_index = int(math.ceil(len(sorted_list)/2))
        return sorted_list[middle_index-1]
    else:
        # Get the two values above/below middle index
        middle_index = int(len(sorted_list)/2)

        below = sorted_list[middle_index-1]
        above = sorted_list[middle_index]

        # return median value
        return (above+below)/2

def is_even(value_list: list):
    """
    This method checks if the length of a list is even.
    :param value_list: list of values.
    :return: True, if the length of the list is even, odd otherwise.
    """
    if len(value_list) % 2 == 0:
        return True
    return False

if __name__ == '__main__':

    import time
    import episcanpy as epi

    #fragnment_file = "/mnt/workspace/jdetlef/data/bamfiles/sorted_esophagus_muscularis_146_0.01_fragments_sorted.bed"
    fragnment_file = "/mnt/workspace/jdetlef/data/bamfiles/cropped_esophagus_146_fragments.bed"
    out_file = "/mnt/workspace/jdetlef/processed_data/Esophagus_146_0.01.h5ad"
    h5ad_file = "/mnt/workspace/jdetlef/processed_data/Esophagus_146_0.01/annotation/anndata/Esophagus_146_0.01.h5ad"
    adata = epi.read_h5ad(h5ad_file)
    # get start time
    start = time.time()

    # print start time
    print("Start: " + str(start))

    # load data and calc score
    #df = load_data(fragnment_file)
    adata = add_chromatin_conditions(adata, fragnment_file)

    # get end time
    end = time.time()

    #adata.obs = adata.obs.fillna(0)
    adata.write(out_file)

    # print time
    print("Time: " + str(end - start))

    print("finished")