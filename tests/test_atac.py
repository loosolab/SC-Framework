import sctoolbox.atac
import yaml


def test_write_TOBIAS_config():

    sctoolbox.atac.write_TOBIAS_config("tobias.yml", bams=["bam1.bam", "bam2.bam"])
    yml = yaml.load(open("tobias.yml"))

    assert yml["data"]["bam1"] == "bam1.bam"
