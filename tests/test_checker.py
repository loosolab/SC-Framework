import pytest
import sctoolbox
from sctoolbox.checker import *

def test_check_cuts():
	input1 = "100"
	input2 = 1
	input3 = 1000	
	output1 = "valid"
	output2 = "invalid"

	result = check_cuts(input1, input2, input3)

	assert result == output1
