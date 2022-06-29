
def test_fetch_info_path():
    assert()

def test_write_info_txt(tmpdir):
    file = tmpdir.join("info.txt")
    write_info_txt("/test/path/", file.strpath)
    assert file.read() == '/test/path/'