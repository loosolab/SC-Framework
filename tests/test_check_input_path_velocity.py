import pytest
import sctoolbox
from sctoolbox.checker import *



#def input_path_velocity(MAINPATH, tenX, DTYPE): #Check if the main directory of solo (MAINPATH) and files exist to assembling the anndata object to make velocyte analysis. tenX is the configuration of samples in the 10X.yml, the DTYPE is the type of Solo data choose (raw or filtered)
#def test_check_input_path_velocity():
    #input parameters (inputdirectory, 10X, datatype)
#                input1 = "/mnt/agnerds/loosolab_SC_RNA_framework/examples/assembling_10_velocity/"
#                input2="10X"
#                input3="raw"
#                print("Test")
#                print(check_input_path_velocity(input1,input2,input3))
                #print path_QUANT

@pytest.mark.parametrize ("path,desc",[("/mnt/agnerds/loosolab_SC_RNA_framework/examples/assembling_10_velocity/","valid"),("/mnt/loosolab_SC_RNA_framework/examples/assembling_10_velocity/","invalid path"),(2,"valid")])
def test_checking_paths(path,desc):
    #path="/mnt/agnerds/loosolab_SC_RNA_framework/examples/assembling_10_velocity/"
    #desc="valid"
    result=checking_paths(path)
    assert result==desc



