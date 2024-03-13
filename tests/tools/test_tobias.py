import sctoolbox.atac
import yaml


def test_write_TOBIAS_config():
    """Test write_TOBIAS_config success."""

    sctoolbox.atac.write_TOBIAS_config("tobias.yml", bams=["bam1.bam", "bam2.bam"])
    yml = yaml.full_load(open("tobias.yml"))

    assert yml["data"]["1"] == "bam1.bam"
