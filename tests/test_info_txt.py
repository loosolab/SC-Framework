from unittest import mock
import builtins
import sctoolbox.checker as checker


@mock.patch.object(builtins, "open", new_callable=mock.mock_open, read_data="/test/path/")
def test_fetch_info_path(tmpdir):
    result_path = checker.fetch_info_txt("test")
    assert result_path == '/test/path/'

def test_write_info_txt(tmpdir):
    file = tmpdir.join("info.txt")
    checker.write_info_txt("/test/path/", file.strpath)
    assert file.read() == '/test/path/'