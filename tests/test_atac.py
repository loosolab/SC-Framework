import sctoolbox.atac
import yaml


def test_write_TOBIAS_config():

    sctoolbox.atac.write_TOBIAS_config("tobias.yml", bams=["bam1.bam", "bam2.bam"])
    yml = yaml.full_load(open("tobias.yml"))

    assert yml["data"]["1"] == "bam1.bam"

test_write_TOBIAS_config()