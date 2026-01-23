"""Test suite for AMULET module.

Tests all functions from the AMULET implementation:
- Utility functions from peakoverlap.py
- Core overlap detection from FragmentFileOverlapCounter.py
- Multiplet detection from AMULET.py
- Framework integration
"""

import pytest
import numpy as np
import pandas as pd
import scanpy as sc
import gzip

# Import module under test
import sctoolbox.tools.amulet as amulet


# ==============================================================================
# FIXTURES
# ==============================================================================


@pytest.fixture
def sample_region_data():
    """Create sample region data for testing utility functions."""
    # Format: [chromosome, start, end, ...]
    data = np.array([
        ["chr1", 100, 200],
        ["chr1", 150, 250],
        ["chr1", 300, 400],
        ["chr2", 100, 200],
        ["chr2", 500, 600],
    ], dtype=object)
    return data


@pytest.fixture
def sample_reads():
    """Create sample reads for testing overlap detection."""
    # Simple case: 3 overlapping reads on chr1
    reads = [
        ["chr1", 100, 200],
        ["chr1", 150, 250],
        ["chr1", 180, 280],
    ]
    return reads


@pytest.fixture
def sample_reads_no_overlap():
    """Create non-overlapping reads."""
    reads = [
        ["chr1", 100, 150],
        ["chr1", 200, 250],
    ]
    return reads


@pytest.fixture
def sample_fragment_file(tmp_path):
    """Create a temporary fragment file for testing."""
    fragment_content = """# Comment line
chr1	100	200	BARCODE1	1
chr1	150	250	BARCODE1	1
chr1	180	280	BARCODE1	1
chr1	300	400	BARCODE1	1
chr1	100	200	BARCODE2	1
chr1	110	210	BARCODE2	1
chr1	120	220	BARCODE2	1
chr1	130	230	BARCODE2	1
chr2	500	600	BARCODE1	1
chr2	100	200	BARCODE3	1
"""
    fragment_path = tmp_path / "fragments.tsv"
    fragment_path.write_text(fragment_content)
    return str(fragment_path)


@pytest.fixture
def sample_fragment_file_gz(tmp_path):
    """Create a compressed fragment file for testing."""
    fragment_content = """chr1	100	200	BARCODE1	1
chr1	150	250	BARCODE1	1
chr1	180	280	BARCODE1	1
"""
    fragment_path = tmp_path / "fragments.tsv.gz"
    with gzip.open(fragment_path, 'wt') as f:
        f.write(fragment_content)
    return str(fragment_path)


@pytest.fixture
def sample_repeat_regions():
    """Create sample repeat regions for testing."""
    return np.array([
        ["chr1", 150, 180],
        ["chr2", 100, 150],
    ], dtype=object)


@pytest.fixture
def sample_repeat_bed_file(tmp_path):
    """Create a temporary BED file with repeat regions."""
    bed_content = """chr1	150	180
chr2	100	150
"""
    bed_path = tmp_path / "repeats.bed"
    bed_path.write_text(bed_content)
    return str(bed_path)


@pytest.fixture
def sample_adata(sample_fragment_file):
    """Create a sample AnnData object for testing."""
    # Create simple count matrix
    n_cells = 3
    n_features = 10
    X = np.random.randint(0, 10, size=(n_cells, n_features))

    adata = sc.AnnData(X)
    adata.obs.index = ["BARCODE1", "BARCODE2", "BARCODE3"]
    adata.obs["sample"] = ["A", "A", "B"]

    return adata


@pytest.fixture
def sample_overlaps_df():
    """Create sample overlaps DataFrame for testing."""
    data = {
        'chr': ['chr1', 'chr1', 'chr1'],
        'start': [100, 300, 100],
        'end': [200, 400, 200],
        'cell_id': ['CELL1', 'CELL1', 'CELL2'],
        'min_overlap': [3, 3, 4],
        'max_overlap': [3, 3, 4],
        'starts': ['100,150,180,', '300,350,380,', '100,110,120,130,'],
        'ends': ['200,250,280,', '400,450,480,', '200,210,220,230,']
    }
    return pd.DataFrame(data)


@pytest.fixture
def sample_summary_df():
    """Create sample summary DataFrame for testing."""
    data = {
        'cell_id': ['CELL1', 'CELL2', 'CELL3'],
        'n_valid_reads': [100, 150, 80],
        'n_overlaps': [2, 1, 0],
        'barcode': ['CELL1', 'CELL2', 'CELL3'],
        'n_total_reads': [120, 180, 100]
    }
    return pd.DataFrame(data)


# ==============================================================================
# UTILITY FUNCTION TESTS (from peakoverlap.py)
# ==============================================================================


class TestGetChrStartSorted:
    """Tests for _get_chr_start_sorted function."""

    def test_basic_sorting(self, sample_region_data):
        """Test that data is sorted by chromosome and start position."""
        result = amulet._get_chr_start_sorted(sample_region_data)

        # Should have two chromosomes
        assert "chr1" in result
        assert "chr2" in result

        # chr1 should have 3 entries sorted by start
        chr1_data = result["chr1"]
        assert len(chr1_data) == 3
        # Check starts are sorted
        assert chr1_data[0, 0] == 100
        assert chr1_data[1, 0] == 150
        assert chr1_data[2, 0] == 300

    def test_index_mapping(self, sample_region_data):
        """Test that index mapping is correct."""
        result = amulet._get_chr_start_sorted(sample_region_data)

        # First entry in chr1 should map to index 0 in original data
        chr1_data = result["chr1"]
        original_idx = int(chr1_data[0, 1])
        assert sample_region_data[original_idx, 1] == 100

    def test_empty_data(self):
        """Test with empty data."""
        empty_data = np.array([]).reshape(0, 3)
        result = amulet._get_chr_start_sorted(empty_data)
        assert len(result) == 0


class TestGetOverlappingRegions:
    """Tests for _get_overlapping_regions function."""

    def test_find_overlapping(self, sample_region_data):
        """Test finding overlapping regions."""
        chr_start_sorted = amulet._get_chr_start_sorted(sample_region_data)

        # Query region that overlaps with first two chr1 entries
        result = amulet._get_overlapping_regions(
            "chr1", 120, 170, chr_start_sorted, sample_region_data
        )

        # Should find indices 0 and 1 (the first two chr1 entries overlap)
        assert len(result) == 2
        assert 0 in result
        assert 1 in result

    def test_no_overlap(self, sample_region_data):
        """Test query with no overlapping regions."""
        chr_start_sorted = amulet._get_chr_start_sorted(sample_region_data)

        # Query region that doesn't overlap anything
        result = amulet._get_overlapping_regions(
            "chr1", 500, 600, chr_start_sorted, sample_region_data
        )

        assert len(result) == 0

    def test_exact_boundaries(self, sample_region_data):
        """Test with exact boundary overlap."""
        chr_start_sorted = amulet._get_chr_start_sorted(sample_region_data)

        # Query exactly matching first entry
        result = amulet._get_overlapping_regions(
            "chr1", 100, 200, chr_start_sorted, sample_region_data
        )

        # Should find at least the first entry
        assert 0 in result

    def test_nonexistent_chromosome(self, sample_region_data):
        """Test query on non-existent chromosome."""
        chr_start_sorted = amulet._get_chr_start_sorted(sample_region_data)

        result = amulet._get_overlapping_regions(
            "chrX", 100, 200, chr_start_sorted, sample_region_data
        )

        assert len(result) == 0


class TestGetUnionPeaks:
    """Tests for _get_union_peaks function."""

    def test_merge_overlapping(self):
        """Test merging of overlapping regions."""
        data1 = np.array([["chr1", 100, 200]], dtype=object)
        data2 = np.array([["chr1", 150, 250]], dtype=object)

        result = amulet._get_union_peaks((data1, data2))

        # Should merge into single region
        assert len(result) == 1
        assert result[0][0] == "chr1"
        assert result[0][1] == 100
        assert result[0][2] == 250

    def test_non_overlapping_separate(self):
        """Test that non-overlapping regions stay separate."""
        data1 = np.array([["chr1", 100, 200]], dtype=object)
        data2 = np.array([["chr1", 300, 400]], dtype=object)

        result = amulet._get_union_peaks((data1, data2))

        # Should have 2 separate regions
        assert len(result) == 2

    def test_multiple_chromosomes(self):
        """Test with regions on multiple chromosomes."""
        data = np.array([
            ["chr1", 100, 200],
            ["chr2", 100, 200],
        ], dtype=object)

        result = amulet._get_union_peaks((data,))

        # Should have 2 regions on different chromosomes
        assert len(result) == 2


class TestGetOverlapCount:
    """Tests for _get_overlap_count function."""

    def test_count_overlaps(self, sample_region_data):
        """Test counting overlaps with multiple datasets."""
        # Create reference datasets
        dataset1 = np.array([["chr1", 100, 200]], dtype=object)
        dataset2 = np.array([["chr1", 150, 250]], dtype=object)

        overlap_vector, overlap_matrix = amulet._get_overlap_count(
            sample_region_data, (dataset1, dataset2)
        )

        # First chr1 entry (100-200) overlaps with dataset1 (exact) and dataset2 (at 150-200)
        assert overlap_matrix[0, 0] == 1
        assert overlap_matrix[0, 1] == 1
        # Second chr1 entry (150-250) overlaps with both datasets
        assert overlap_matrix[1, 0] == 1
        assert overlap_matrix[1, 1] == 1

        # Check vector sums
        assert overlap_vector[0] == 2  # First entry overlaps both datasets (100-200 overlaps both)
        assert overlap_vector[1] == 2  # Second entry overlaps both datasets


# ==============================================================================
# CORE OVERLAP DETECTION TESTS (from FragmentFileOverlapCounter.py)
# ==============================================================================


class TestGetOverlapsBasic:
    """Tests for _get_overlaps function."""

    def test_simple_overlap(self, sample_reads):
        """Test with simple overlapping reads."""
        result = amulet._get_overlaps(sample_reads, expected_overlap=2)

        # Should detect one overlap region
        assert len(result) >= 1

        # Check chromosome and position
        overlap = result[0]
        assert overlap[0] == "chr1"
        assert overlap[3] >= 3  # min_overlap >= 3
        assert overlap[4] >= 3  # max_overlap >= 3

    def test_overlap_positions(self, sample_reads):
        """Test that overlap positions are correct."""
        result = amulet._get_overlaps(sample_reads, expected_overlap=2)

        # The overlap region should be where all 3 reads overlap
        # Reads: 100-200, 150-250, 180-280
        # Overlap: 180-200 (where all 3 overlap)
        overlap = result[0]
        assert overlap[1] == 180  # start
        assert overlap[2] == 200  # end


class TestGetOverlapsNoOverlap:
    """Tests for _get_overlaps with non-overlapping reads."""

    def test_no_overlap(self, sample_reads_no_overlap):
        """Test with non-overlapping reads."""
        result = amulet._get_overlaps(sample_reads_no_overlap, expected_overlap=2)

        # Should return empty list
        assert len(result) == 0

    def test_exactly_two_reads(self):
        """Test with exactly 2 overlapping reads (threshold edge case)."""
        reads = [
            ["chr1", 100, 200],
            ["chr1", 150, 250],
        ]

        # With expected_overlap=2, exactly 2 overlapping reads should NOT trigger
        result = amulet._get_overlaps(reads, expected_overlap=2)
        assert len(result) == 0


class TestGetOverlapsEdgeCases:
    """Edge case tests for _get_overlaps."""

    def test_three_overlapping_reads(self):
        """Test that 3+ overlapping reads trigger detection."""
        reads = [
            ["chr1", 100, 200],
            ["chr1", 150, 250],
            ["chr1", 175, 275],
        ]

        result = amulet._get_overlaps(reads, expected_overlap=2)

        # Should detect overlap (3 > 2)
        assert len(result) >= 1

    def test_empty_reads(self):
        """Test with empty read list."""
        result = amulet._get_overlaps([], expected_overlap=2)
        assert len(result) == 0

    def test_single_read(self):
        """Test with single read."""
        reads = [["chr1", 100, 200]]
        result = amulet._get_overlaps(reads, expected_overlap=2)
        assert len(result) == 0


class TestAssignReadsWithinOverlaps:
    """Tests for _assign_reads_within_overlaps function."""

    def test_assigns_positions(self, sample_reads):
        """Test that start/end positions are correctly assigned."""
        # First get overlaps
        overlaps = amulet._get_overlaps(sample_reads, expected_overlap=2)

        # Check that starts and ends are added
        assert len(overlaps) > 0
        overlap = overlaps[0]

        # Overlap should have 7 elements: chr, start, end, min, max, starts_str, ends_str
        assert len(overlap) == 7

        # Check comma-separated format
        starts_str = overlap[5]
        ends_str = overlap[6]
        assert "," in starts_str
        assert "," in ends_str

    def test_position_format(self, sample_reads):
        """Test that positions are comma-separated strings."""
        overlaps = amulet._get_overlaps(sample_reads, expected_overlap=2)

        overlap = overlaps[0]
        starts_str = overlap[5]
        ends_str = overlap[6]

        # Parse positions
        starts = [int(x) for x in starts_str.split(",") if x]
        ends = [int(x) for x in ends_str.split(",") if x]

        # Should have same number of starts and ends
        assert len(starts) == len(ends)
        assert len(starts) >= 3  # At least 3 reads


class TestCountFragmentOverlaps:
    """Tests for count_fragment_overlaps function."""

    def test_parse_tsv(self, sample_fragment_file):
        """Test parsing plain TSV file."""
        barcodes = ["BARCODE1", "BARCODE2", "BARCODE3"]

        overlaps_df, summary_df = amulet.count_fragment_overlaps(
            sample_fragment_file, barcodes, chromosomes=["chr1", "chr2"]
        )

        # Check overlaps_df structure
        assert "chr" in overlaps_df.columns
        assert "start" in overlaps_df.columns
        assert "end" in overlaps_df.columns
        assert "cell_id" in overlaps_df.columns
        assert "starts" in overlaps_df.columns
        assert "ends" in overlaps_df.columns

    def test_parse_gzip(self, sample_fragment_file_gz):
        """Test parsing gzipped TSV file."""
        barcodes = ["BARCODE1"]

        overlaps_df, summary_df = amulet.count_fragment_overlaps(
            sample_fragment_file_gz, barcodes, chromosomes=["chr1"]
        )

        # Should parse without error
        assert isinstance(overlaps_df, pd.DataFrame)
        assert isinstance(summary_df, pd.DataFrame)

    def test_barcode_filtering(self, sample_fragment_file):
        """Test that barcode filtering works."""
        # Only include BARCODE1
        barcodes = ["BARCODE1"]

        overlaps_df, summary_df = amulet.count_fragment_overlaps(
            sample_fragment_file, barcodes, chromosomes=["chr1", "chr2"]
        )

        # All overlaps should be from BARCODE1
        if len(overlaps_df) > 0:
            assert all(overlaps_df["cell_id"] == "BARCODE1")

    def test_chromosome_filtering(self, sample_fragment_file):
        """Test that chromosome filtering works."""
        barcodes = ["BARCODE1", "BARCODE2"]

        # Only include chr1
        overlaps_df, summary_df = amulet.count_fragment_overlaps(
            sample_fragment_file, barcodes, chromosomes=["chr1"]
        )

        # All overlaps should be from chr1
        if len(overlaps_df) > 0:
            assert all(overlaps_df["chr"] == "chr1")

    def test_insert_size_filtering(self, sample_fragment_file):
        """Test that insert size filtering works."""
        barcodes = ["BARCODE1", "BARCODE2"]

        # Set very small max insert size
        overlaps_df, summary_df = amulet.count_fragment_overlaps(
            sample_fragment_file, barcodes,
            chromosomes=["chr1"],
            max_insert_size=50  # Very small
        )

        # Should filter out most/all reads
        # Check valid reads count
        assert summary_df["n_valid_reads"].sum() == 0

    def test_starts_ends_columns(self, sample_fragment_file):
        """Test that overlaps_df contains starts and ends columns."""
        barcodes = ["BARCODE1", "BARCODE2"]

        overlaps_df, summary_df = amulet.count_fragment_overlaps(
            sample_fragment_file, barcodes, chromosomes=["chr1"]
        )

        assert "starts" in overlaps_df.columns
        assert "ends" in overlaps_df.columns


# ==============================================================================
# MULTIPLET DETECTION TESTS (from AMULET.py)
# ==============================================================================


class TestFilterKnownRepeats:
    """Tests for _filter_known_repeats function."""

    def test_filters_repeats(self, sample_overlaps_df, sample_repeat_regions):
        """Test that reads in repeat regions are filtered."""
        data = sample_overlaps_df.values

        result = amulet._filter_known_repeats(
            data, sample_repeat_regions, expected_overlap=2
        )

        # Should have filtered some overlaps or reduced their extent
        assert isinstance(result, np.ndarray)

    def test_no_repeats(self, sample_overlaps_df):
        """Test with empty repeat regions."""
        data = sample_overlaps_df.values
        empty_repeats = np.array([]).reshape(0, 3)

        result = amulet._filter_known_repeats(
            data, empty_repeats, expected_overlap=2
        )

        # Should return all overlaps (first 4 columns)
        assert len(result) == len(data)

    def test_recalculates_overlaps(self, sample_overlaps_df, sample_repeat_regions):
        """Test that overlaps are recalculated after filtering reads."""
        data = sample_overlaps_df.values

        result = amulet._filter_known_repeats(
            data, sample_repeat_regions, expected_overlap=2
        )

        # Result should have 4 columns: chr, start, end, cell_id
        if len(result) > 0:
            assert result.shape[1] == 4


class TestGenerateMatrix:
    """Tests for _generate_matrix function."""

    def test_creates_binary_matrix(self, sample_overlaps_df):
        """Test that matrix is binary (0 or 1)."""
        # Create simple union overlaps
        data = sample_overlaps_df[['chr', 'start', 'end', 'cell_id']].values
        cell_ids = np.array(['CELL1', 'CELL2', 'CELL3'])
        union_overlaps = amulet._get_union_peaks((data,))

        matrix, reverse_dict, region_info = amulet._generate_matrix(
            data, cell_ids, union_overlaps
        )

        # Check matrix is binary
        assert set(np.unique(matrix)).issubset({0, 1})

    def test_cell_id_mapping(self, sample_overlaps_df):
        """Test that cell ID mapping is correct."""
        data = sample_overlaps_df[['chr', 'start', 'end', 'cell_id']].values
        cell_ids = np.array(['CELL1', 'CELL2', 'CELL3'])
        union_overlaps = amulet._get_union_peaks((data,))

        matrix, reverse_dict, region_info = amulet._generate_matrix(
            data, cell_ids, union_overlaps
        )

        # Check reverse dict maps indices to cell IDs
        assert reverse_dict[0] == 'CELL1'
        assert reverse_dict[1] == 'CELL2'
        assert reverse_dict[2] == 'CELL3'

    def test_matrix_shape(self, sample_overlaps_df):
        """Test that matrix has correct shape."""
        data = sample_overlaps_df[['chr', 'start', 'end', 'cell_id']].values
        cell_ids = np.array(['CELL1', 'CELL2', 'CELL3'])
        union_overlaps = amulet._get_union_peaks((data,))

        matrix, _, _ = amulet._generate_matrix(data, cell_ids, union_overlaps)

        # Shape should be (n_regions, n_cells)
        assert matrix.shape[0] == len(union_overlaps)
        assert matrix.shape[1] == len(cell_ids)


class TestInferRepeats:
    """Tests for _infer_repeats function."""

    def test_detects_high_overlap_regions(self):
        """Test that high-overlap regions are detected as repetitive."""
        # Create matrix where first region has very high overlap
        matrix = np.array([
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],  # Region 0: overlaps in all cells
            [1, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # Region 1: overlaps in 1 cell
            [0, 1, 0, 0, 0, 0, 0, 0, 0, 0],  # Region 2: overlaps in 1 cell
        ])

        union_overlaps = np.array([
            ["chr1", 100, 200],
            ["chr1", 300, 400],
            ["chr1", 500, 600],
        ], dtype=object)

        rep_regions, non_rep, probas = amulet._infer_repeats(
            matrix, union_overlaps, threshold=0.05
        )

        # First region should be detected as repetitive
        # (it has significantly more overlaps than average)
        assert len(rep_regions) >= 1

    def test_returns_probabilities(self):
        """Test that probabilities are returned."""
        matrix = np.array([[1, 0], [0, 1]])
        union_overlaps = np.array([
            ["chr1", 100, 200],
            ["chr1", 300, 400],
        ], dtype=object)

        rep_regions, non_rep, probas = amulet._infer_repeats(
            matrix, union_overlaps, threshold=0.05
        )

        # Probabilities should have additional columns for stats
        assert probas.shape[1] > union_overlaps.shape[1]


class TestGetDoublets:
    """Tests for _get_doublets function."""

    def test_pvalue_calculation(self):
        """Test that p-values are calculated."""
        matrix = np.array([
            [1, 1, 1, 0, 0],
            [1, 1, 1, 0, 0],
            [1, 1, 1, 0, 0],
        ])

        union_overlaps = np.array([
            ["chr1", 100, 200],
            ["chr1", 300, 400],
            ["chr1", 500, 600],
        ], dtype=object)

        reverse_dict = {0: "CELL1", 1: "CELL2", 2: "CELL3", 3: "CELL4", 4: "CELL5"}

        result = amulet._get_doublets(matrix, union_overlaps, reverse_dict)

        # Should return array with cell_id, p_value, q_value
        assert result.shape[1] == 3
        assert len(result) == 5  # 5 cells

    def test_fdr_correction(self):
        """Test that FDR correction is applied."""
        # Create matrix where some cells have many more overlaps
        matrix = np.array([
            [1, 1, 1, 0, 0],
            [1, 1, 1, 0, 0],
            [1, 1, 1, 0, 0],
            [1, 1, 0, 0, 0],
            [1, 0, 0, 0, 0],
        ])

        union_overlaps = np.array([
            ["chr1", 100, 200],
            ["chr1", 300, 400],
            ["chr1", 500, 600],
            ["chr1", 700, 800],
            ["chr1", 900, 1000],
        ], dtype=object)

        reverse_dict = {0: "CELL1", 1: "CELL2", 2: "CELL3", 3: "CELL4", 4: "CELL5"}

        result = amulet._get_doublets(matrix, union_overlaps, reverse_dict)

        # Q-values should be >= p-values (FDR correction)
        for row in result:
            p_val = float(row[1])
            q_val = float(row[2])
            assert q_val >= p_val or np.isclose(q_val, p_val)


class TestDetectMultiplets:
    """Tests for detect_multiplets function."""

    def test_full_pipeline(self, sample_overlaps_df, sample_summary_df):
        """Test full pipeline orchestration."""
        result_df = amulet.detect_multiplets(
            sample_overlaps_df, sample_summary_df,
            q_threshold=0.01
        )

        # Check required columns
        assert "cell_id" in result_df.columns
        assert "barcode" in result_df.columns
        assert "p_value" in result_df.columns
        assert "q_value" in result_df.columns
        assert "doublet_score" in result_df.columns
        assert "predicted_doublet" in result_df.columns

    def test_qvalue_threshold(self, sample_overlaps_df, sample_summary_df):
        """Test q-value threshold application."""
        result_df = amulet.detect_multiplets(
            sample_overlaps_df, sample_summary_df,
            q_threshold=0.01
        )

        # predicted_doublet should be based on q_value < threshold
        for _, row in result_df.iterrows():
            if row['q_value'] < 0.01:
                assert row['predicted_doublet'] is True
            else:
                assert row['predicted_doublet'] is False

    def test_with_repeat_filter(self, sample_overlaps_df, sample_summary_df, sample_repeat_regions):
        """Test pipeline with repeat filter."""
        result_df = amulet.detect_multiplets(
            sample_overlaps_df, sample_summary_df,
            repeat_regions=sample_repeat_regions,
            q_threshold=0.01
        )

        # Should complete without error
        assert isinstance(result_df, pd.DataFrame)

    def test_empty_overlaps(self, sample_summary_df):
        """Test with empty overlaps DataFrame."""
        empty_overlaps = pd.DataFrame(
            columns=['chr', 'start', 'end', 'cell_id', 'min_overlap',
                     'max_overlap', 'starts', 'ends']
        )

        result_df = amulet.detect_multiplets(
            empty_overlaps, sample_summary_df,
            q_threshold=0.01
        )

        # Should return DataFrame with all cells as non-doublets
        assert len(result_df) == len(sample_summary_df)
        assert not result_df['predicted_doublet'].any()


# ==============================================================================
# FRAMEWORK INTEGRATION TESTS
# ==============================================================================


class TestEstimateDoubletsAmuletBasic:
    """Basic tests for estimate_doublets_amulet function."""

    def test_adds_doublet_score(self, sample_adata, sample_fragment_file):
        """Test that doublet_score is added to adata.obs."""
        amulet.estimate_doublets_amulet(
            sample_adata, sample_fragment_file,
            chromosomes=["chr1", "chr2"]
        )

        assert "doublet_score" in sample_adata.obs.columns

    def test_adds_predicted_doublet(self, sample_adata, sample_fragment_file):
        """Test that predicted_doublet is added to adata.obs."""
        amulet.estimate_doublets_amulet(
            sample_adata, sample_fragment_file,
            chromosomes=["chr1", "chr2"]
        )

        assert "predicted_doublet" in sample_adata.obs.columns

    def test_adds_metadata(self, sample_adata, sample_fragment_file):
        """Test that metadata is added to adata.uns."""
        amulet.estimate_doublets_amulet(
            sample_adata, sample_fragment_file,
            chromosomes=["chr1", "chr2"]
        )

        assert "amulet" in sample_adata.uns
        assert "q_threshold" in sample_adata.uns["amulet"]


class TestEstimateDoubletsAmuletGroupby:
    """Tests for estimate_doublets_amulet with groupby parameter."""

    def test_per_sample_processing(self, sample_adata, sample_fragment_file):
        """Test per-sample processing with groupby."""
        amulet.estimate_doublets_amulet(
            sample_adata, sample_fragment_file,
            chromosomes=["chr1", "chr2"],
            groupby="sample"
        )

        # Should complete without error
        assert "doublet_score" in sample_adata.obs.columns
        assert "predicted_doublet" in sample_adata.obs.columns

    def test_results_combined(self, sample_adata, sample_fragment_file):
        """Test that results are combined correctly across groups."""
        amulet.estimate_doublets_amulet(
            sample_adata, sample_fragment_file,
            chromosomes=["chr1", "chr2"],
            groupby="sample"
        )

        # All cells should have results
        assert not sample_adata.obs["doublet_score"].isna().any()


class TestEstimateDoubletsAmuletInplace:
    """Tests for estimate_doublets_amulet inplace parameter."""

    def test_inplace_true(self, sample_adata, sample_fragment_file):
        """Test inplace=True returns None."""
        result = amulet.estimate_doublets_amulet(
            sample_adata, sample_fragment_file,
            chromosomes=["chr1", "chr2"],
            inplace=True
        )

        assert result is None
        assert "doublet_score" in sample_adata.obs.columns

    def test_inplace_false(self, sample_adata, sample_fragment_file):
        """Test inplace=False returns modified AnnData."""
        result = amulet.estimate_doublets_amulet(
            sample_adata, sample_fragment_file,
            chromosomes=["chr1", "chr2"],
            inplace=False
        )

        assert result is not None
        assert isinstance(result, sc.AnnData)
        assert "doublet_score" in result.obs.columns

        # Original should not be modified
        assert "doublet_score" not in sample_adata.obs.columns


class TestEstimateDoubletsAmuletWithRepeats:
    """Tests for estimate_doublets_amulet with repeat filter."""

    def test_with_repeat_bed(self, sample_adata, sample_fragment_file, sample_repeat_bed_file):
        """Test with repeat filter BED file."""
        amulet.estimate_doublets_amulet(
            sample_adata, sample_fragment_file,
            chromosomes=["chr1", "chr2"],
            repeat_filter=sample_repeat_bed_file
        )

        # Should complete without error
        assert "doublet_score" in sample_adata.obs.columns

    def test_filters_known_repeats(self, sample_adata, sample_fragment_file, sample_repeat_bed_file):
        """Test that known repeats are filtered."""
        # Run with and without repeat filter
        adata_no_filter = sample_adata.copy()
        adata_with_filter = sample_adata.copy()

        amulet.estimate_doublets_amulet(
            adata_no_filter, sample_fragment_file,
            chromosomes=["chr1", "chr2"]
        )

        amulet.estimate_doublets_amulet(
            adata_with_filter, sample_fragment_file,
            chromosomes=["chr1", "chr2"],
            repeat_filter=sample_repeat_bed_file
        )

        # Both should complete (results may differ)
        assert "doublet_score" in adata_no_filter.obs.columns
        assert "doublet_score" in adata_with_filter.obs.columns
