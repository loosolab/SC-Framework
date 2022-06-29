import pytest
import sctoolbox.checker as checker

def test_fetch_info_path(tmpdir):
    # Todo mock
    file = tmpdir.join("info.txt")
    checker.write_info_txt("/test/path/", file.strpath)
    result_path = checker.fetch_info_txt(file.strpath)
    assert result_path == '/test/path/'

def test_write_info_txt(tmpdir):
    file = tmpdir.join("info.txt")
    checker.write_info_txt("/test/path/", file.strpath)
    assert file.read() == '/test/path/'