"""Tests for usage of TOBIAS within the sc framework."""

import sctoolbox.tools.tobias as tobias
import yaml


def test_write_TOBIAS_config():
    """Test write_TOBIAS_config success."""

    tobias.write_TOBIAS_config("tobias.yml", bams=["bam1.bam", "bam2.bam"])
    yml = yaml.full_load(open("tobias.yml"))

    assert yml["data"]["1"] == "bam1.bam"
