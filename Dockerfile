FROM condaforge/mambaforge

LABEL maintainer="Jan Detleffsen <jan.detleffsen@mpi-bn.mpg.de>"

COPY . /tmp/
COPY scripts /scripts/

# make scripts executeable
RUN chmod +x scripts/bedGraphToBigWig 

# install Fortran compiler 
RUN apt-get update --assume-yes && \
    apt-get install --assume-yes gfortran && \
    # Install missing libraries
    apt-get install bedtools && \
    apt-get install -y libcurl4

# install git to check for file changes
RUN apt-get install -y git-all

# update mamba
RUN mamba update -n base mamba && \
    mamba --version

# install enviroment
RUN mamba env update -n base -f /tmp/sctoolbox_env.yml

# install sctoolbox
RUN pip install "/tmp/[all]" && \
    pip install pytest && \
    pip install pytest-cov && \
    pip install pytest-html 

# clear tmp
RUN rm -r /tmp/*

# Set the time zone
RUN echo 'Europe/Berlin' > apt-get install -y tzdata

# Generate an ssh key
RUN apt-get install -y openssh-client && \
    mkdir .ssh && \
    ssh-keygen -t ed25519 -N "" -f .ssh/id_ed25519

ENV ROOT=TRUE \
    DISABLE_AUTH=TRUE

