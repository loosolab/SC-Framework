"""AMULET: ATAC-seq MULtiplet Estimation Tool integration for sctoolbox.

Doublet detection for single-cell ATAC-seq data based on the observation
that diploid cells should have at most 2 reads overlapping at any genomic locus.

This reimplements core functions from the AMULET tool and integrates them
into the sctoolbox framework.

Reference: https://doi.org/10.1186/s13059-021-02469-x
"""

import numpy as np
import pandas as pd
import scanpy as sc
import gzip
from pathlib import Path
from scipy import stats
from statsmodels.stats.multitest import multipletests

from beartype import beartype
from beartype.typing import Optional, Tuple, List, Dict

import sctoolbox.utils.decorator as deco
from sctoolbox._settings import settings

logger = settings.logger


# ==============================================================================
# UTILITY FUNCTIONS (from peakoverlap.py)
# ==============================================================================


def _get_chr_start_sorted(
    data: np.ndarray,
    chr_idx: int = 0,
    start_idx: int = 1
) -> Dict[str, np.ndarray]:
    """Return dict mapping chromosome to sorted [start, original_index] array.

    Creates an index structure for efficient binary search in overlap detection.

    Parameters
    ----------
    data : np.ndarray
        Array of shape (n_peaks, n_features) containing peak/region data.
        Must have chromosome at chr_idx and start position at start_idx.
    chr_idx : int, default 0
        Column index for chromosome in data array.
    start_idx : int, default 1
        Column index for start position in data array.

    Returns
    -------
    Dict[str, np.ndarray]
        Dictionary mapping each chromosome to a 2D array where:
        - Column 0: start positions (sorted ascending)
        - Column 1: indices into original data array

    Notes
    -----
    Source: peakoverlap.py → getChrStartSorted() (lines 56-88)
    Time complexity: O(n log n) due to sorting
    """
    all_chr = np.unique(data[:, chr_idx])
    rv = dict()
    for i in range(len(all_chr)):
        idx = np.where(data[:, chr_idx] == all_chr[i])[0]
        chr_data = data[idx, :]
        sorted_idx = np.argsort(chr_data[:, start_idx], kind="mergesort")
        rv[all_chr[i]] = np.concatenate(
            (
                np.transpose(chr_data[sorted_idx, start_idx][np.newaxis]),
                np.transpose(idx[sorted_idx][np.newaxis])
            ),
            axis=1
        )
    return rv


def _get_overlapping_regions(
    chrom: str,
    start: int,
    end: int,
    chr_start_sorted: Dict[str, np.ndarray],
    data: np.ndarray,
    end_idx: int = 2
) -> Tuple[int, ...]:
    """Binary search to find all regions overlapping a given position.

    Parameters
    ----------
    chrom : str
        Chromosome of the query region.
    start : int
        Start position of the query region.
    end : int
        End position of the query region.
    chr_start_sorted : Dict[str, np.ndarray]
        Chromosome-sorted index from _get_chr_start_sorted().
    data : np.ndarray
        Original data array containing the regions to search.
    end_idx : int, default 2
        Column index for end position in data array.

    Returns
    -------
    Tuple[int, ...]
        Tuple of indices into data array for all overlapping regions.

    Notes
    -----
    Source: peakoverlap.py → getOverlappingRegions() (lines 91-151)
    Time complexity: O(log n) for binary search + O(k) for k overlapping regions
    Uses position format (1-based, fully-closed intervals).
    """
    try:
        start_sorted = chr_start_sorted[chrom]
    except KeyError:
        start_sorted = []

    s = 0
    e = len(start_sorted)

    # Binary search to find starting position
    while (e - s) > 1:
        mi = int(s + ((e - s) / 2))
        m_start = start_sorted[mi, 0]
        if m_start < start:
            s = mi
        elif m_start > start:
            e = mi
        else:
            s = mi
            e = mi

    # Scan until starts are greater than end
    rv = []
    idx = s
    while idx < len(start_sorted) and end > start_sorted[idx, 0]:
        didx = int(start_sorted[idx, 1])
        c_start = start_sorted[idx, 0]
        c_end = data[didx, end_idx]
        # Position format comparison (1-based, fully-closed)
        if start <= c_end and end >= c_start:
            rv.append(didx)
        idx += 1

    return tuple(rv)


def _get_union_peaks(
    datasets: Tuple[np.ndarray, ...],
    chr_idx: int = 0,
    start_idx: int = 1,
    end_idx: int = 2
) -> np.ndarray:
    """Merge overlapping regions across datasets into union set.

    Parameters
    ----------
    datasets : Tuple[np.ndarray, ...]
        Tuple of arrays, each with shape (n_peaks, n_features).
        Each array must have chromosome, start, and end columns.
    chr_idx : int, default 0
        Column index for chromosome.
    start_idx : int, default 1
        Column index for start position.
    end_idx : int, default 2
        Column index for end position.

    Returns
    -------
    np.ndarray
        Array of merged regions with columns [chromosome, start, end].

    Notes
    -----
    Source: peakoverlap.py → getUnionPeaks() (lines 341-358)
    Overlapping regions are merged into single regions spanning their union.
    """
    combined_data = np.concatenate(datasets)
    sorted_locations = _get_chr_start_sorted(combined_data)
    rv = []

    for cur_chr in sorted_locations:
        locations = sorted_locations[cur_chr]
        cur_loci = combined_data[int(locations[0, 1]), :3].copy()

        for i in range(1, len(locations)):
            next_loci = combined_data[int(locations[i, 1]), :3]
            if next_loci[1] > cur_loci[2]:
                # No overlap - add current and start new
                rv.append([cur_chr, cur_loci[1], cur_loci[2]])
                cur_loci = next_loci.copy()
            else:
                # Overlap - extend current region
                cur_loci[2] = max(cur_loci[2], next_loci[2])

        # Add final region for this chromosome
        rv.append([cur_chr, cur_loci[1], cur_loci[2]])

    return np.array(rv, dtype=object)


def _get_overlap_index(
    data: np.ndarray,
    peakset: np.ndarray,
    chr_idx: int = 0,
    start_idx: int = 1,
    end_idx: int = 2,
    set_chr_idx: int = 0,
    set_start_idx: int = 1,
    set_end_idx: int = 2
) -> np.ndarray:
    """Return boolean vector indicating overlap with a set of peaks.

    Parameters
    ----------
    data : np.ndarray
        Array of regions to test for overlap.
    peakset : np.ndarray
        Array of reference peaks to check against.
    chr_idx : int, default 0
        Chromosome column index for data.
    start_idx : int, default 1
        Start column index for data.
    end_idx : int, default 2
        End column index for data.
    set_chr_idx : int, default 0
        Chromosome column index for peakset.
    set_start_idx : int, default 1
        Start column index for peakset.
    set_end_idx : int, default 2
        End column index for peakset.

    Returns
    -------
    np.ndarray
        Boolean vector indicating whether each region in data overlaps peakset.

    Notes
    -----
    Source: peakoverlap.py → getOverlapIndex() (lines 154-202)
    Time complexity: O(n log n)
    """
    sorted_consensus = _get_chr_start_sorted(peakset, set_chr_idx, set_start_idx)
    rv = np.zeros(len(data), dtype=bool)

    for i in range(len(data)):
        cur_chr = data[i, chr_idx]
        cur_start = data[i, start_idx]
        cur_end = data[i, end_idx]
        if len(_get_overlapping_regions(cur_chr, cur_start, cur_end,
                                        sorted_consensus, peakset, set_end_idx)) > 0:
            rv[i] = True

    return rv


def _get_overlap_count(
    count_dataset: np.ndarray,
    datasets: Tuple[np.ndarray, ...],
    chr_idx: int = 0,
    start_idx: int = 1,
    end_idx: int = 2
) -> Tuple[np.ndarray, np.ndarray]:
    """Count how many peaks overlap with each dataset.

    Parameters
    ----------
    count_dataset : np.ndarray
        Array of regions to count overlaps for.
    datasets : Tuple[np.ndarray, ...]
        Tuple of reference peak arrays to check against.
    chr_idx : int, default 0
        Chromosome column index.
    start_idx : int, default 1
        Start column index.
    end_idx : int, default 2
        End column index.

    Returns
    -------
    overlap_vector : np.ndarray
        Vector of shape (n_peaks,) with count of datasets overlapping each peak.
    overlap_matrix : np.ndarray
        Matrix of shape (n_peaks, n_datasets) indicating overlap with each dataset.

    Notes
    -----
    Source: peakoverlap.py → getOverlapCount() (lines 204-246)
    Time complexity: O(m * n log n) where m = number of datasets
    """
    overlap_vector = np.zeros(len(count_dataset))
    overlap_matrix = np.zeros((len(count_dataset), len(datasets)))

    for i in range(len(datasets)):
        cur_v = _get_overlap_index(
            count_dataset, datasets[i], chr_idx, start_idx, end_idx,
            chr_idx, start_idx
        ).astype(int)
        overlap_matrix[:, i] = cur_v
        overlap_vector = overlap_vector + cur_v

    return overlap_vector, overlap_matrix


# ==============================================================================
# CORE OVERLAP DETECTION (from FragmentFileOverlapCounter.py)
# ==============================================================================


def _get_overlaps(
    reads: List[List],
    expected_overlap: int
) -> List[List]:
    """Find genomic regions where >expected_overlap reads overlap.

    Uses a running sum algorithm to efficiently detect regions where more
    reads overlap than expected for a diploid cell.

    Parameters
    ----------
    reads : List[List]
        List of reads, each as [chromosome, start, end].
        All reads must be from the same chromosome.
    expected_overlap : int
        Expected maximum number of overlapping reads (2 for diploid).
        Regions with more overlaps than this are flagged.

    Returns
    -------
    List[List]
        List of overlap regions, each as:
        [chromosome, start, end, min_overlap, max_overlap, starts_str, ends_str]
        where starts_str and ends_str are comma-separated read positions.

    Notes
    -----
    Source: FragmentFileOverlapCounter.py → getOverlaps() (lines 46-106)
    Time complexity: O(n log n) due to sorting
    Calls _assign_reads_within_overlaps() before returning.
    """
    # If there are no reads, we are done
    if len(reads) <= expected_overlap:
        return []

    # Create Overlap Index (1 if starting, -1 if ending) O(n)
    overlap_index = []
    for cur_read in reads:
        overlap_index.append([cur_read[1], 1])
        overlap_index.append([cur_read[2], -1])
    overlap_index = np.array(overlap_index)

    # Sort the overlap index by position O(n log n)
    overlap_index = overlap_index[np.argsort(overlap_index[:, 0]), :]

    index_size = len(overlap_index)

    # Calculate running sum O(n)
    running_sum = [overlap_index[0, 1]]
    running_sum_pos = [overlap_index[0, 0]]
    for i in range(1, index_size):
        prev_i = i - 1
        cur_sum = running_sum[-1] + overlap_index[i, 1]
        if overlap_index[prev_i, 0] == overlap_index[i, 0]:
            # Sum same positions together
            running_sum[-1] = cur_sum
        else:
            # Start a new position
            running_sum.append(cur_sum)
            running_sum_pos.append(overlap_index[i, 0])

    # Detect overlaps > the expected and report regions using the running sum O(n)
    rv = []
    chromosome = reads[0][0]
    within_segment = False
    segment_start = -1
    min_overlap = -1
    max_overlap = -1

    for i in range(len(running_sum)):
        if within_segment:
            if running_sum[i] <= expected_overlap:
                rv.append([chromosome, segment_start, running_sum_pos[i],
                          min_overlap, max_overlap])
                within_segment = False
                segment_start = -1
                min_overlap = -1
                max_overlap = -1
            else:
                max_overlap = max(max_overlap, running_sum[i])
        else:
            if running_sum[i] > expected_overlap:
                segment_start = running_sum_pos[i]
                min_overlap = running_sum[i]
                max_overlap = running_sum[i]
                within_segment = True

    if within_segment:
        rv.append([chromosome, segment_start, running_sum_pos[-1],
                  min_overlap, max_overlap])

    _assign_reads_within_overlaps(rv, reads)

    return rv


def _assign_reads_within_overlaps(
    overlaps: List[List],
    reads: List[List]
) -> None:
    """Assign individual read start/end positions to each overlap region.

    Modifies overlaps in-place by appending comma-separated start and end
    position strings to each overlap entry.

    Parameters
    ----------
    overlaps : List[List]
        List of overlap regions from _get_overlaps(), each as:
        [chromosome, start, end, min_overlap, max_overlap].
        Modified in-place to add starts and ends strings.
    reads : List[List]
        List of reads, each as [chromosome, start, end].

    Notes
    -----
    Source: FragmentFileOverlapCounter.py → assignReadsWithinOverlaps() (lines 108-154)
    This function is critical for repeat filtering, which needs read-level precision.
    """
    num_reads = len(reads)
    ri = 0
    multi_overlaps = []

    for cur_overlap in overlaps:
        cur_ol_start = cur_overlap[1]
        cur_ol_end = cur_overlap[2]
        cur_starts = []
        cur_ends = []

        next_multi_overlaps = []
        for cur_multi_overlap in multi_overlaps:
            cur_read_start = cur_multi_overlap[1]
            cur_read_end = cur_multi_overlap[2]

            if cur_read_end >= cur_ol_start and cur_ol_end >= cur_read_start:
                cur_starts.append(cur_read_start)
                cur_ends.append(cur_read_end)

            if cur_read_end >= cur_ol_end:
                next_multi_overlaps.append(cur_multi_overlap)

        multi_overlaps = next_multi_overlaps

        while ri < num_reads:
            cur_read = reads[ri]
            cur_read_start = cur_read[1]
            cur_read_end = cur_read[2]

            if cur_read_end >= cur_ol_start and cur_ol_end >= cur_read_start:
                cur_starts.append(cur_read_start)
                cur_ends.append(cur_read_end)

            if cur_read_end >= cur_ol_end:
                multi_overlaps.append(cur_read)

            ri += 1

        start_string = ""
        end_string = ""
        for i in range(len(cur_starts)):
            start_string += str(cur_starts[i]) + ","
            end_string += str(cur_ends[i]) + ","

        cur_overlap.append(start_string)
        cur_overlap.append(end_string)


def count_fragment_overlaps( # noqa: C901
    fragments_file: str,
    barcodes: List[str],
    chromosomes: Optional[List[str]] = None,
    expected_overlap: int = 2,
    max_insert_size: int = 900,
    start_bases: int = 0,
    end_bases: int = 0
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Count overlapping regions from fragment file.

    Parses a 10x CellRanger format fragment file and identifies regions
    where more reads overlap than expected for diploid cells.

    Parameters
    ----------
    fragments_file : str
        Path to fragment file (.tsv, .txt, .bed or .tsv.gz, .txt.gz, .bed.gz).
    barcodes : List[str]
        List of cell barcodes to include.
    chromosomes : Optional[List[str]], default None
        List of chromosomes to analyze. If None, uses chr1-chr22.
    expected_overlap : int, default 2
        Expected maximum number of overlapping reads (2 for diploid).
    max_insert_size : int, default 900
        Maximum fragment size to include (bp).
    start_bases : int, default 0
        Bases to add to start position.
    end_bases : int, default 0
        Bases to add to end position.

    Returns
    -------
    overlaps_df : pd.DataFrame
        DataFrame with columns: chr, start, end, cell_id, min_overlap,
        max_overlap, starts, ends. The 'starts' and 'ends' columns contain
        comma-separated individual read positions.
    summary_df : pd.DataFrame
        DataFrame with columns: cell_id, n_valid_reads, n_overlaps, barcode,
        n_total_reads.

    Raises
    ------
    ValueError
        Raises if the fragment file format is not recognized.

    Notes
    -----
    Source: FragmentFileOverlapCounter.py → findOverlaps() (lines 213-354)
    """
    # Default chromosomes (autosomes)
    if chromosomes is None:
        chromosomes = [f"chr{i}" for i in range(1, 23)]

    chr_dict = {chrom: True for chrom in chromosomes}

    # Set up file reader
    path = Path(fragments_file)
    suffixes = path.suffixes  # e.g. ['.bed', '.gz']

    if suffixes in ([".txt", ".gz"], [".tsv", ".gz"], [".bed", ".gz"]):
        fragment_reader = gzip.open(path, "rt")
    elif suffixes in ([".txt"], [".tsv"], [".bed"]):
        fragment_reader = open(path, "r")
    else:
        raise ValueError(
            "Fragment file must be *.txt, *.tsv, *.bed or their .gz equivalents"
        )

    # Set up barcode maps
    bc_map = dict()
    previous_reads = dict()
    previous_ends = dict()
    overlap_counts = dict()
    valid_reads_per_cell = dict()
    reads_per_cell = dict()

    for barcode in barcodes:
        bc_map[barcode] = []
        previous_reads[barcode] = []
        previous_ends[barcode] = -1
        overlap_counts[barcode] = 0
        valid_reads_per_cell[barcode] = 0
        reads_per_cell[barcode] = 0

    # Collect all overlaps
    all_overlaps = []
    chromosome_set = ""

    # Loop through all reads to detect overlaps
    for cur_line in fragment_reader:
        if cur_line.strip().startswith("#"):
            continue

        split = cur_line.split("\t")

        cur_chr = split[0]
        cur_start = int(split[1]) + start_bases
        cur_end = int(split[2]) + end_bases
        cur_barcode = split[3].strip()

        cur_location = [cur_chr, cur_start, cur_end]

        insert_size = cur_end - cur_start

        if cur_barcode not in bc_map:
            continue

        reads_per_cell[cur_barcode] = reads_per_cell[cur_barcode] + 1

        if insert_size > max_insert_size:
            continue  # skip when the insert size is greater than the limit

        if cur_chr not in chr_dict:
            continue  # skip reads from chromosomes not in the provided list

        valid_reads_per_cell[cur_barcode] = valid_reads_per_cell[cur_barcode] + 1

        if cur_chr != chromosome_set:
            # Finish checking remaining overlaps and reset the running lists
            # This change indicates we are done with the chromosome set

            # Loop through all barcodes
            for prev_barcode in previous_reads.keys():
                # 1 - find overlaps with remaining lists
                if len(previous_reads[prev_barcode]) > 0:
                    cur_overlaps = _get_overlaps(previous_reads[prev_barcode],
                                                 expected_overlap)
                    # 2 - add overlaps to collection
                    for overlap in cur_overlaps:
                        all_overlaps.append(overlap + [prev_barcode])
                    # 3 - Update overlap count for each
                    overlap_counts[prev_barcode] = (
                        overlap_counts[prev_barcode] + len(cur_overlaps)
                    )

                # Reset lists
                previous_reads[prev_barcode] = []
                previous_ends[prev_barcode] = -1

            chromosome_set = cur_chr

        prev_end = previous_ends[cur_barcode]

        if prev_end < cur_start:
            # There can be no more overlaps for this segment.
            # Therefore, check for overlaps meeting criteria and start a new list.

            # 1: Find the overlaps
            cur_overlaps = _get_overlaps(previous_reads[cur_barcode], expected_overlap)

            # 2: Add overlaps to collection
            for overlap in cur_overlaps:
                all_overlaps.append(overlap + [cur_barcode])

            # 3: Update the overlap count for the current barcode
            overlap_counts[cur_barcode] = overlap_counts[cur_barcode] + len(cur_overlaps)

            # Start a new running list of reads
            previous_reads[cur_barcode] = [cur_location]

        else:
            # Add this read to the running list of overlaps for this barcode
            previous_reads[cur_barcode].append(cur_location)

        # Assign a new endpoint for overlaps
        new_end = max(prev_end, cur_end)
        previous_ends[cur_barcode] = new_end

    # Process remaining reads at end of file
    for prev_barcode in previous_reads.keys():
        if len(previous_reads[prev_barcode]) > 0:
            cur_overlaps = _get_overlaps(previous_reads[prev_barcode], expected_overlap)
            for overlap in cur_overlaps:
                all_overlaps.append(overlap + [prev_barcode])
            overlap_counts[prev_barcode] = overlap_counts[prev_barcode] + len(cur_overlaps)

    fragment_reader.close()

    # Create overlaps DataFrame
    if len(all_overlaps) > 0:
        overlaps_df = pd.DataFrame(
            all_overlaps,
            columns=['chr', 'start', 'end', 'min_overlap', 'max_overlap',
                     'starts', 'ends', 'cell_id']
        )
        # Reorder columns to match original format
        overlaps_df = overlaps_df[['chr', 'start', 'end', 'cell_id',
                                   'min_overlap', 'max_overlap', 'starts', 'ends']]
    else:
        overlaps_df = pd.DataFrame(
            columns=['chr', 'start', 'end', 'cell_id', 'min_overlap',
                     'max_overlap', 'starts', 'ends']
        )

    # Create summary DataFrame
    summary_data = []
    for barcode in barcodes:
        summary_data.append([
            barcode,  # cell_id
            valid_reads_per_cell[barcode],
            overlap_counts[barcode],
            barcode,  # barcode
            reads_per_cell[barcode]
        ])

    summary_df = pd.DataFrame(
        summary_data,
        columns=['cell_id', 'n_valid_reads', 'n_overlaps', 'barcode', 'n_total_reads']
    )

    return overlaps_df, summary_df


# ==============================================================================
# MULTIPLET DETECTION (from AMULET.py)
# ==============================================================================


def _filter_known_repeats( # noqa: C901
    data: np.ndarray,
    repeat_regions: np.ndarray,
    expected_overlap: int
) -> np.ndarray:
    """Filter overlaps using known repeat regions with read-level precision.

    Uses individual read positions (starts/ends columns) to check each read
    against repeat regions. Reads falling within repeats are removed, and
    overlap regions are recalculated from the remaining reads.

    Parameters
    ----------
    data : np.ndarray
        Array of overlaps with columns including starts and ends (as comma-separated
        strings in the last two columns).
    repeat_regions : np.ndarray
        Array of repeat regions with columns [chromosome, start, end].
    expected_overlap : int
        Expected maximum overlap count (for recalculating overlaps).

    Returns
    -------
    np.ndarray
        Filtered overlaps array with columns [chr, start, end, cell_id].

    Notes
    -----
    Source: AMULET.py → getFilteredOverlaps() (lines 106-179)
    This function provides read-level precision for repeat filtering.
    """
    if len(repeat_regions) == 0:
        # No repeats to filter, return first 4 columns
        return data[:, :4] if len(data) > 0 else np.array([])

    rv = []
    sorted_repeats = _get_chr_start_sorted(repeat_regions)

    for cur_overlap in data:
        cur_chr = cur_overlap[0]
        starts = np.array(str(cur_overlap[-2]).split(",")[:-1], dtype=int)
        ends = np.array(str(cur_overlap[-1]).split(",")[:-1], dtype=int)

        observed_loci = dict()

        new_starts = []
        new_ends = []
        for i in range(len(starts)):
            key = str(starts[i]) + "-" + str(ends[i])
            if key not in observed_loci:
                observed_loci[key] = True

                overlap = _get_overlapping_regions(
                    cur_chr, starts[i], ends[i], sorted_repeats, repeat_regions
                )
                if len(overlap) == 0:
                    new_starts.append(starts[i])
                    new_ends.append(ends[i])

        if len(new_starts) < len(starts):
            if len(new_starts) > expected_overlap:
                # Recalculate overlaps

                # starts increment by 1
                counts = np.ones((len(new_starts), 2), dtype=object)
                counts[:, 0] = new_starts

                # ends decrement by 1
                counts2 = -1 * np.ones((len(new_ends), 2), dtype=object)
                counts2[:, 0] = new_ends

                # Combine the counts and sort them
                combined_counts = np.concatenate((counts, counts2))
                count_order = np.argsort(combined_counts[:, 0])
                combined_counts = combined_counts[count_order]

                # Scan through and maintain a running sum
                # when the running sum is > 2, continue until <= 2 and report that overlap
                running_sum = 0
                i = 0
                start_overlap = False
                start_overlap_position = 0

                while i < len(combined_counts):
                    running_sum += combined_counts[i][1]
                    j = i + 1
                    while j < len(combined_counts):
                        if combined_counts[i, 0] == combined_counts[j, 0]:
                            running_sum += combined_counts[j][1]
                            j += 1
                        else:
                            break

                    if not start_overlap and running_sum > expected_overlap:
                        start_overlap = True
                        start_overlap_position = combined_counts[i][0]
                    elif start_overlap and running_sum <= expected_overlap:
                        # append overlap
                        rv.append([cur_chr, start_overlap_position,
                                  combined_counts[i][0], cur_overlap[3]])
                        start_overlap = False

                    i = j

                if start_overlap:
                    rv.append([cur_chr, start_overlap_position,
                              combined_counts[-1][0], cur_overlap[3]])

        else:
            rv.append(list(cur_overlap[:4]))

    rv = np.array(rv, dtype=object) if len(rv) > 0 else np.array([]).reshape(0, 4)
    return rv


def _generate_matrix(
    data: np.ndarray,
    cell_ids: np.ndarray,
    union_overlaps: np.ndarray
) -> Tuple[np.ndarray, Dict[int, str], Dict[int, List]]:
    """Create binary cell × region matrix.

    Parameters
    ----------
    data : np.ndarray
        Array of overlaps with columns [chr, start, end, cell_id].
    cell_ids : np.ndarray
        Array of all cell IDs to include in the matrix.
    union_overlaps : np.ndarray
        Array of union overlap regions with columns [chr, start, end].

    Returns
    -------
    matrix : np.ndarray
        Binary matrix of shape (n_regions, n_cells) indicating which
        cells have overlaps in each region.
    reverse_cell_id_dict : Dict[int, str]
        Dictionary mapping matrix column indices to cell IDs.
    region_info : Dict[int, List]
        Dictionary mapping region indices to overlap information.

    Notes
    -----
    Source: AMULET.py → generateMatrix() (lines 35-65)
    Uses _get_chr_start_sorted() and _get_overlapping_regions() for efficient lookup.
    """
    # Map cell IDs to integers
    cell_id_dict = dict()
    reverse_cell_id_dict = dict()
    for i in range(len(cell_ids)):
        cell_id_dict[cell_ids[i]] = i
        reverse_cell_id_dict[i] = cell_ids[i]

    sorted_union_overlaps = _get_chr_start_sorted(union_overlaps)

    region_info = dict()

    matrix = np.zeros((len(union_overlaps), len(cell_ids)))
    for i in range(len(data)):
        cur_chr = data[i, 0]
        cur_start = data[i, 1]
        cur_end = data[i, 2]
        cell_id = data[i, 3]
        overlap = _get_overlapping_regions(
            cur_chr, cur_start, int(cur_end) + 1,
            sorted_union_overlaps, union_overlaps
        )

        if cell_id in cell_id_dict:
            for oi in overlap:
                matrix[oi, cell_id_dict[cell_id]] = 1

                if oi not in region_info:
                    region_info[oi] = []
                merged_length = union_overlaps[oi][2] - union_overlaps[oi][1] + 1
                length = cur_end - cur_start + 1
                region_info[oi].append([length, length / merged_length])

    return matrix, reverse_cell_id_dict, region_info


def _infer_repeats(
    matrix: np.ndarray,
    union_overlaps: np.ndarray,
    threshold: float
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Identify likely repetitive regions using Poisson test on row sums.

    Regions with significantly high overlap counts across many cells
    are identified as likely repetitive regions.

    Parameters
    ----------
    matrix : np.ndarray
        Binary cell × region matrix from _generate_matrix().
    union_overlaps : np.ndarray
        Array of union overlap regions.
    threshold : float
        FDR-corrected q-value threshold for calling repeats.

    Returns
    -------
    repetitive_regions : np.ndarray
        Array of regions identified as repetitive.
    non_repetitive_regions : np.ndarray
        Array of regions not identified as repetitive.
    probabilities : np.ndarray
        Array with region info and p/q-values for all regions.

    Notes
    -----
    Source: AMULET.py → inferRepeats() (lines 67-82)
    Uses Poisson survival function with mean overlap as expected value.
    """
    row_sum = np.sum(matrix, axis=1)
    rep_probabilities = []
    rep_mean = np.mean(row_sum[:])

    for cur_val in row_sum:
        rep_probabilities.append(stats.poisson.sf(cur_val, rep_mean))

    rep_probabilities = np.array(rep_probabilities)
    corrected_rep_probabilities = multipletests(rep_probabilities, method='fdr_bh')

    rep_regions = union_overlaps[corrected_rep_probabilities[1] < threshold]
    non_rep_regions = union_overlaps[corrected_rep_probabilities[1] >= threshold]

    rep_probas = np.concatenate(
        (
            union_overlaps,
            row_sum[:, np.newaxis],
            rep_probabilities[:, np.newaxis],
            corrected_rep_probabilities[1][:, np.newaxis]
        ),
        axis=1
    )

    return rep_regions, non_rep_regions, rep_probas


def _get_doublets(
    matrix: np.ndarray,
    union_overlaps: np.ndarray,
    reverse_cell_id_dict: Dict[int, str]
) -> np.ndarray:
    """Detect doublets using Poisson test on column sums.

    Cells with significantly more overlap regions than expected are
    identified as potential doublets.

    Parameters
    ----------
    matrix : np.ndarray
        Binary cell × region matrix from _generate_matrix().
    union_overlaps : np.ndarray
        Array of union overlap regions.
    reverse_cell_id_dict : Dict[int, str]
        Dictionary mapping matrix column indices to cell IDs.

    Returns
    -------
    np.ndarray
        Array with columns [cell_id, p_value, q_value] for all cells.

    Notes
    -----
    Source: AMULET.py → getDoublets() (lines 84-102)
    Uses Poisson survival function with mean overlap as expected value.
    FDR correction using Benjamini-Hochberg method.
    """
    col_sum = np.sum(matrix, axis=0)
    doublet_probabilities = []
    doublet_mean = np.mean(col_sum[:])

    for cur_val in col_sum:
        doublet_probabilities.append(stats.poisson.sf(cur_val, doublet_mean))

    doublet_probabilities = np.array(doublet_probabilities)
    corrected_doublet_probabilities = multipletests(
        doublet_probabilities, method='fdr_bh'
    )

    doublets_probas = []
    index = 0
    for cur in corrected_doublet_probabilities[1]:
        doublets_probas.append([
            reverse_cell_id_dict[index],
            doublet_probabilities[index],
            cur
        ])
        index += 1

    return np.array(doublets_probas, dtype=object)


def detect_multiplets(
    overlaps_df: pd.DataFrame,
    summary_df: pd.DataFrame,
    repeat_regions: Optional[np.ndarray] = None,
    q_threshold: float = 0.01,
    q_repeat_threshold: float = 0.01,
    min_overlap: int = 1,
    expected_overlap: int = 2
) -> pd.DataFrame:
    """Full detection pipeline with two-stage repeat filtering.

    Orchestrates the complete AMULET doublet detection pipeline:
    1. Filter known repeat regions (if provided)
    2. Filter by minimum overlap length
    3. Create union of overlap regions
    4. Generate cell × region matrix
    5. Infer additional repetitive regions
    6. Filter inferred repeats
    7. Regenerate matrix without inferred repeats
    8. Run doublet detection

    Parameters
    ----------
    overlaps_df : pd.DataFrame
        DataFrame from count_fragment_overlaps() with overlap data.
    summary_df : pd.DataFrame
        DataFrame from count_fragment_overlaps() with per-cell summary.
    repeat_regions : Optional[np.ndarray], default None
        Array of known repeat regions to filter.
    q_threshold : float, default 0.01
        FDR-corrected q-value threshold for calling multiplets.
    q_repeat_threshold : float, default 0.01
        Q-value threshold for inferring repetitive regions.
    min_overlap : int, default 1
        Minimum overlap length in bp to keep.
    expected_overlap : int, default 2
        Expected maximum overlap count for diploid cells.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns: cell_id, barcode, p_value, q_value,
        doublet_score, predicted_doublet.

    Notes
    -----
    Source: AMULET.py main pipeline (lines 186-226)
    """
    if len(overlaps_df) == 0:
        # No overlaps found - return empty results
        result_df = pd.DataFrame({
            'cell_id': summary_df['cell_id'].values,
            'barcode': summary_df['barcode'].values,
            'p_value': np.ones(len(summary_df)),
            'q_value': np.ones(len(summary_df)),
            'doublet_score': np.zeros(len(summary_df)),
            'predicted_doublet': False
        })
        return result_df

    # Convert overlaps_df to numpy array
    data = overlaps_df.values
    cell_ids = summary_df['cell_id'].values

    # Step 1: Filter known repeat regions
    if repeat_regions is not None and len(repeat_regions) > 0:
        logger.info("Filtering known repeat regions.")
        # Merge repeat regions into union set
        simple_repeats = _get_union_peaks(tuple([repeat_regions]))
        filtered_data = _filter_known_repeats(data, simple_repeats, expected_overlap)
    else:
        # No repeats to filter, extract first 4 columns
        filtered_data = data[:, :4]

    # Step 2: Filter by minimum overlap length
    if len(filtered_data) > 0:
        lengths = np.array([
            int(filtered_data[i, 2]) - int(filtered_data[i, 1]) + 1
            for i in range(len(filtered_data))
        ])
        min_overlap_adjusted = min_overlap - 1
        filtered_data = filtered_data[lengths > min_overlap_adjusted, :]

        num_filtered = len(data) - len(filtered_data)
        if len(data) > 0:
            logger.info(
                f"Number of regions filtered: {num_filtered} "
                f"({100 * num_filtered / len(data):.1f}%)"
            )

    if len(filtered_data) == 0:
        # All overlaps filtered - return empty results
        result_df = pd.DataFrame({
            'cell_id': summary_df['cell_id'].values,
            'barcode': summary_df['barcode'].values,
            'p_value': np.ones(len(summary_df)),
            'q_value': np.ones(len(summary_df)),
            'doublet_score': np.zeros(len(summary_df)),
            'predicted_doublet': False
        })
        return result_df

    logger.info("Detecting multiplets.")

    # Step 3: Generate union of overlaps
    union_overlaps = _get_union_peaks(tuple([filtered_data]))

    # Step 4: Generate initial matrix
    matrix, reverse_cell_id_dict, _ = _generate_matrix(
        filtered_data, cell_ids, union_overlaps
    )

    # Step 5: Infer repetitive regions
    repetitive, non_repetitive, _ = _infer_repeats(
        matrix, union_overlaps, q_repeat_threshold
    )

    # Step 6: Filter inferred repetitive regions
    if len(repetitive) > 0:
        rep_filter_index, _ = _get_overlap_count(filtered_data, tuple([repetitive]))
        rep_filtered_data = filtered_data[rep_filter_index == 0, :]
    else:
        rep_filtered_data = filtered_data

    # Step 7: Regenerate matrix without inferred repeats
    if len(rep_filtered_data) > 0:
        rep_filtered_union_overlaps = _get_union_peaks(tuple([rep_filtered_data]))
        rep_filtered_matrix, reverse_cell_id_dict2, _ = _generate_matrix(
            rep_filtered_data, cell_ids, rep_filtered_union_overlaps
        )
    else:
        # No data left after filtering
        result_df = pd.DataFrame({
            'cell_id': summary_df['cell_id'].values,
            'barcode': summary_df['barcode'].values,
            'p_value': np.ones(len(summary_df)),
            'q_value': np.ones(len(summary_df)),
            'doublet_score': np.zeros(len(summary_df)),
            'predicted_doublet': False
        })
        return result_df

    # Step 8: Run doublet detection
    doublets_with_prob = _get_doublets(
        rep_filtered_matrix, rep_filtered_union_overlaps, reverse_cell_id_dict2
    )

    # Create result DataFrame
    summary_dict = dict()
    for i in range(len(summary_df)):
        summary_dict[summary_df.iloc[i]['cell_id']] = summary_df.iloc[i]

    results = []
    for i in range(len(doublets_with_prob)):
        cell_id = doublets_with_prob[i, 0]
        p_value = float(doublets_with_prob[i, 1])
        q_value = float(doublets_with_prob[i, 2])

        # Get barcode from summary
        barcode = summary_dict[cell_id]['barcode']

        # Doublet score is -log10(q_value), capped at 0
        doublet_score = -np.log10(max(q_value, 1e-300))

        results.append({
            'cell_id': cell_id,
            'barcode': barcode,
            'p_value': p_value,
            'q_value': q_value,
            'doublet_score': doublet_score,
            'predicted_doublet': q_value < q_threshold
        })

    result_df = pd.DataFrame(results)

    return result_df


# ==============================================================================
# FRAMEWORK INTEGRATION
# ==============================================================================


@deco.log_anndata
@beartype
def estimate_doublets_amulet(
    adata: sc.AnnData,
    fragments_file: str,
    chromosomes: Optional[List[str]] = None,
    repeat_filter: Optional[str] = None,
    expected_overlap: int = 2,
    max_insert_size: int = 900,
    q_threshold: float = 0.01,
    q_repeat_threshold: float = 0.01,
    min_overlap: int = 1,
    groupby: Optional[str] = None,
    inplace: bool = True
) -> Optional[sc.AnnData]:
    """
    Estimate doublet cells using AMULET algorithm.

    AMULET detects multiplets in single-cell ATAC-seq data based on the
    observation that diploid cells should have at most 2 reads overlapping
    at any genomic locus. Cells with significantly more overlapping loci
    than expected are called as multiplets.

    Parameters
    ----------
    adata : sc.AnnData
        Annotated data matrix with cell barcodes in obs.index.
    fragments_file : str
        Path to fragment file (10x CellRanger format: .tsv.gz).
    chromosomes : Optional[List[str]], default None
        List of chromosomes to analyze. Default: autosomes (chr1-22).
    repeat_filter : Optional[str], default None
        Path to BED file of repetitive regions to exclude.
    expected_overlap : int, default 2
        Expected maximum read overlap in singlets (2 for diploid).
    max_insert_size : int, default 900
        Maximum fragment size to include (bp).
    q_threshold : float, default 0.01
        FDR-corrected q-value threshold for calling multiplets.
    q_repeat_threshold : float, default 0.01
        Q-value threshold for inferring additional repetitive regions.
    min_overlap : int, default 1
        Minimum overlap length in bp to keep.
    groupby : Optional[str], default None
        Column in adata.obs to process samples separately.
    inplace : bool, default True
        Whether to modify adata in place.

    Returns
    -------
    Optional[sc.AnnData]
        If inplace=False, returns modified AnnData. Otherwise None.

    Notes
    -----
    Adds the following to adata:
    - adata.obs["doublet_score"]: Float score (higher = more likely doublet)
    - adata.obs["predicted_doublet"]: Boolean prediction
    - adata.uns["amulet"]: Algorithm metadata

    Reference: https://doi.org/10.1186/s13059-021-02469-x
    """
    if not inplace:
        adata = adata.copy()

    # Load repeat regions if provided
    repeat_regions = None
    if repeat_filter is not None:
        logger.info(f"Loading repeat filter from: {repeat_filter}")
        repeat_df = pd.read_csv(repeat_filter, sep="\t", header=None)
        repeat_regions = repeat_df.values[:, 0:3]

    # Get barcodes from adata
    all_barcodes = list(adata.obs.index)

    # Initialize result columns
    adata.obs["doublet_score"] = 0.0
    adata.obs["predicted_doublet"] = False

    # Store metadata
    adata.uns["amulet"] = {
        "q_threshold": q_threshold,
        "q_repeat_threshold": q_repeat_threshold,
        "expected_overlap": expected_overlap,
        "max_insert_size": max_insert_size,
        "min_overlap": min_overlap,
        "fragments_file": fragments_file
    }

    if groupby is not None:
        # Process each group separately
        groups = adata.obs[groupby].unique()
        logger.info(f"Processing {len(groups)} groups from '{groupby}'")

        for group in groups:
            logger.info(f"Processing group: {group}")
            group_mask = adata.obs[groupby] == group
            group_barcodes = list(adata.obs.index[group_mask])

            # Count overlaps for this group
            overlaps_df, summary_df = count_fragment_overlaps(
                fragments_file=fragments_file,
                barcodes=group_barcodes,
                chromosomes=chromosomes,
                expected_overlap=expected_overlap,
                max_insert_size=max_insert_size
            )

            # Detect multiplets
            result_df = detect_multiplets(
                overlaps_df=overlaps_df,
                summary_df=summary_df,
                repeat_regions=repeat_regions,
                q_threshold=q_threshold,
                q_repeat_threshold=q_repeat_threshold,
                min_overlap=min_overlap,
                expected_overlap=expected_overlap
            )

            # Update adata with results
            for _, row in result_df.iterrows():
                barcode = row['cell_id']
                if barcode in adata.obs.index:
                    adata.obs.loc[barcode, "doublet_score"] = row['doublet_score']
                    adata.obs.loc[barcode, "predicted_doublet"] = row['predicted_doublet']

    else:
        # Process all cells together
        logger.info(f"Processing {len(all_barcodes)} cells")

        # Count overlaps
        overlaps_df, summary_df = count_fragment_overlaps(
            fragments_file=fragments_file,
            barcodes=all_barcodes,
            chromosomes=chromosomes,
            expected_overlap=expected_overlap,
            max_insert_size=max_insert_size
        )

        # Detect multiplets
        result_df = detect_multiplets(
            overlaps_df=overlaps_df,
            summary_df=summary_df,
            repeat_regions=repeat_regions,
            q_threshold=q_threshold,
            q_repeat_threshold=q_repeat_threshold,
            min_overlap=min_overlap,
            expected_overlap=expected_overlap
        )

        # Update adata with results
        for _, row in result_df.iterrows():
            barcode = row['cell_id']
            if barcode in adata.obs.index:
                adata.obs.loc[barcode, "doublet_score"] = row['doublet_score']
                adata.obs.loc[barcode, "predicted_doublet"] = row['predicted_doublet']

    # Log summary
    n_doublets = adata.obs["predicted_doublet"].sum()
    n_total = len(adata.obs)
    logger.info(
        f"AMULET detected {n_doublets} doublets out of {n_total} cells "
        f"({100 * n_doublets / n_total:.1f}%)"
    )

    if not inplace:
        return adata
