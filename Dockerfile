FROM condaforge/mambaforge

LABEL maintainer="Jan Detleffsen <jan.detleffsen@mpi-bn.mpg.de>"

COPY . /home/sc_framework/
COPY scripts /scripts/

# Set the time zone (before installing any packages)
RUN echo 'Europe/Berlin' > apt-get install -y tzdata

# make scripts executable
RUN chmod +x scripts/bedGraphToBigWig 

# Clear the local repository of retrieved package files
RUN apt-get update --assume-yes && \
    apt-get clean

# install Fortran compiler 
RUN apt-get install --assume-yes gfortran

# Set timezone for tzdata
ARG DEBIAN_FRONTEND=noninteractive
ENV TZ=Etc/UTC

# Install missing libraries
RUN apt-get install bedtools && \
    apt-get install -y libcurl4 && \
    apt-get install -y git && \
    apt-get install -y build-essential && \ 
    pip install --upgrade pip

# update mamba
RUN mamba update -n base mamba && \
    mamba --version 

# install environment
RUN mamba env update -n base -f /home/sc_framework/sctoolbox_env.yml

# Install Jupyter extensions
RUN jupyter contrib nbextension install --sys-prefix && \
    jupyter nbextension enable scratchpad/main --sys-prefix && \
    jupyter nbextension enable runtools/main --sys-prefix && \
    jupyter nbextension enable init_cell/main --sys-prefix && \
    jupyter nbextension enable execute_time/ExecuteTime --sys-prefix && \
    jupyter nbextension enable scroll_down/main --sys-prefix

# install sctoolbox
RUN pip install "/home/sc_framework/[all]" --group "/home/sc_framework/pyproject.toml:test"

# clean cache files
RUN mamba clean --all -y && \
    pip cache purge

# Generate an ssh key
RUN apt-get install -y openssh-client && \
    mkdir .ssh && \
    ssh-keygen -t ed25519 -N "" -f .ssh/id_ed25519
