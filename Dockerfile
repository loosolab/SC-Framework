FROM condaforge/mambaforge

LABEL maintainer="Jan Detleffsen <jan.detleffsen@mpi-bn.mpg.de>"

COPY . /tmp/
COPY scripts /scripts/

# Set the time zone (before installing any packages)
RUN echo 'Europe/Berlin' > apt-get install -y tzdata

# make scripts executeable
RUN chmod +x scripts/bedGraphToBigWig 

# install dependencies for pages build
RUN pip install sphinx-exec-code && \
    pip install sphinx sphinx-rtd-theme && \
    pip install nbsphinx && \
    pip install nbsphinx_link

# system install of pandoc is needed
RUN apt-get update && \
    apt-get install -qq -y pandoc

# install Fortran compiler 
RUN apt-get update --assume-yes && \
    apt-get install --assume-yes gfortran && \
    # Install missing libraries
    apt-get install bedtools && \
    apt-get install -y libcurl4

# install git to check for file changes
RUN apt-get install -y git

# update mamba
RUN mamba update -n base mamba && \
    mamba --version

# install enviroment
RUN mamba env update -n base -f /tmp/sctoolbox_env.yml

# install sctoolbox
RUN pip install "/tmp/[all]" && \
    pip install pytest && \
    pip install pytest-cov && \
    pip install pytest-html && \
    pip install pytest-mock

# clear tmp
RUN rm -r /tmp/*

# Generate an ssh key
RUN apt-get install -y openssh-client && \
    mkdir .ssh && \
    ssh-keygen -t ed25519 -N "" -f .ssh/id_ed25519

ENV ROOT=TRUE \
    DISABLE_AUTH=TRUE

