import sctoolbox.creators as creator
import anndata
import pytest
import pandas as pd

@pytest.fixture
def test_adata():
    adata = anndata.AnnData(pd.DataFrame({"a": pd.Series(1, index=list(range(2)),dtype="float32"),
                                          "b": pd.Series(1, index=list(range(2)), dtype="float32")},
                                          index=["a","b","c","d"]))
    return adata

@pytest.mark.parametrize("key,value",[(1,2), ("test", "string"), ("test", 1), (1,"test"), ("asdf", [1,2,3])])
def test_build_infor_wo_infoprocess(test_adata, key, value):
    adata = creator.build_infor(test_adata,key,value)
    assert adata.uns["infoprocess"][key] == value

@pytest.mark.parametrize("key,value",[(1,2), ("test", "string"), ("test", 1), (1,"test"), ("asdf", [1,2,3])])
def test_build_infor_w_infoprocess(test_adata, key, value):
    test_adata.uns["infoprocess"]={}
    adata = creator.build_infor(test_adata,key,value)
    assert adata.uns["infoprocess"][key] == value