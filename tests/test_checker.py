import sctoolbox.checker as ch

def test_check_cuts():
    """ Test if int representation of ans is in given range. """
    result = ch.check_cuts(ans="100", limi1=1, limit2=1000)

    assert result == "valid"
