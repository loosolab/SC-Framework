import pytest
import anndata
import pandas as pd
import numpy as np
import sctoolbox.analyser as an

@pytest.fixture
def test_adata_vr():
    adata = anndata.AnnData(pd.DataFrame({"a": pd.Series(1, index=list(range(2)),dtype="float32"),
                                          "b": pd.Series(1, index=list(range(2)), dtype="float32")},
                                          index=["a","b","c","d"]))

    variance_ratio = np.array([0.08749175, 0.03792242, 0.02643393, 0.02533128, 0.0180286 ,
                               0.01313281, 0.01159428, 0.01045423, 0.0083562 , 0.00675285,
                               0.00628902, 0.00555756, 0.00518746, 0.00414544, 0.00378045,
                               0.00332042, 0.00329669, 0.00297686, 0.00284822, 0.00258963,
                               0.00246014, 0.00222408, 0.00212752, 0.0019508 , 0.00190878,
                               0.00179127, 0.00175954, 0.00161796, 0.00157837, 0.00154598,
                               0.00149027, 0.00147236, 0.00141266, 0.00132336, 0.00126029,
                               0.00122388, 0.0011817 , 0.0011381 , 0.00108894, 0.00107301,
                               0.00103355, 0.00102861, 0.00098351, 0.00093241, 0.00090659,
                               0.0008869 , 0.00086826, 0.00086241, 0.00082759, 0.000824])

    adata.uns["pca"] = {"variance_ratio": variance_ratio}
    return adata

def test_define_PC(test_adata_vr):
    print(test_adata_vr)
