import pysam

def buffered_reader(path, read_num):
    """
    Open bam file and yield chunks of reads.

    Parameters
    ----------
    path : str
        Path to bam file.
    read_num : int
        Number of reads per chunk.
    """
    # open bam
    bam = pysam.AlignmentFile(path, "rb")

    # produce a chunk of reads
    chunk = list()
    for read in bam:
        chunk.append(read)

        if len(chunk) >= read_num:
            # yield full chunk
            yield chunk
            # remove old chunk
            chunk = list()

    # yield last reads/ partially filled chunk
    yield chunk
