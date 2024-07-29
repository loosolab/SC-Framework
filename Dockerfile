FROM condaforge/mambaforge

LABEL maintainer="Jan Detleffsen <jan.detleffsen@mpi-bn.mpg.de>"

COPY . /home/sc_framework/
COPY scripts /scripts/

# Set the time zone (before installing any packages)
RUN echo 'Europe/Berlin' > apt-get install -y tzdata

# make scripts executeable
RUN chmod +x scripts/bedGraphToBigWig 

# Clear the local repository of retrieved package files
RUN apt-get update --assume-yes && \
    apt-get clean

# install Fortran compiler 
RUN apt-get install --assume-yes gfortran

# Install missing libraries
RUN apt-get install bedtools && \
    apt-get install -y libcurl4 && \
    apt-get install -y git

# update mamba
RUN mamba update -n base mamba && \
    mamba --version

# Workaround for https://github.com/pypa/setuptools/issues/4519
RUN echo "setuptools<72.0.0" > /home/contraint.txt && \
    export PIP_CONSTRAINT=/home/contraint.txt

# install enviroment
RUN mamba env update -n base -f /home/sc_framework/sctoolbox_env.yml

# Workaround for https://github.com/pypa/setuptools/issues/4519
RUN pip install setuptools<72.0.0

# install sctoolbox
RUN pip install "/home/sc_framework/[all]" && \
    pip install pytest && \
    pip install pytest-cov && \
    pip install pytest-html && \
    pip install pytest-mock

# Generate an ssh key
RUN apt-get install -y openssh-client && \
    mkdir .ssh && \
    ssh-keygen -t ed25519 -N "" -f .ssh/id_ed25519

