"""Test quality control functions."""

import numpy as np
from scipy.sparse import csr_matrix
import sctoolbox.tools.qc_filter as qc
import scanpy as sc
import numpy as np
import matplotlib.pyplot as plt

# Prevent figures from being shown, we just check that they are created
plt.switch_backend("Agg")


# Define the test function
def test_denoise_data(mocker):
    # Mock the AnnData objects
    adata = sc.AnnData(np.random.rand(100, 10))
    adata_raw = sc.AnnData(np.random.rand(100, 10))
    
    # Set the X attribute to simulate real data
    adata.X = csr_matrix(np.random.rand(100, 10))
    adata_raw.X = csr_matrix(np.random.rand(100, 10))
    
    # Mock the setup_anndata function
    #mocker.patch('setup_anndata')
    
    # Mock the model class and its methods
    mock_model = mocker.patch('scar.model')
    mock_scar_instance = mock_model.return_value
    mock_scar_instance.train.return_value = None
    mock_scar_instance.inference.return_value = None
    mock_scar_instance.native_counts = np.random.rand(100, 10)

    # Mock the _save_figure function
    #mocker.patch('sctoolbox.tools.qc_filter._save_figure')

    # Mock the logger
    mock_logger = mocker.patch('sctoolbox.tools.qc_filter.logger')

    # Call the function
    result = qc.denoise_data(adata, adata_raw, feature_type='Gene Expression', epochs=10, prob=0.99, save=None, verbose=False)

    # Assertions
    assert isinstance(result, sc.AnnData)
    assert result.X.shape == adata.X.shape

    # Ensure the logger methods were called
    assert mock_logger.info.call_count == 3
