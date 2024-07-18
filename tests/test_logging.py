"""Test logging functions."""


from sctoolbox._settings import settings
logger = settings.logger


def test_user_logging():
    """Test is logfile is correctly overwritten."""

    settings.log_file = "test.log"
    logger.info("test_info")
    assert "test_info" in open(settings.log_file).read()

    # Set again to the same file
    settings.log_file = "test.log"
    logger.info("test_info2")
    content = open(settings.log_file).read()
    assert "test_info" in content  # check that the first log is still there
    assert "test_info2" in content

    # Set to overwrite the file
    settings.overwrite_log = True
    logger.info("test_info3")
    content = open(settings.log_file).read()
    assert "test_info2" not in content  # previous log was overwritten
    assert "test_info3" in content      # new log is there
