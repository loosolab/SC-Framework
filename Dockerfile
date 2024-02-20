FROM condaforge/mambaforge

LABEL maintainer="Jan Detleffsen <jan.detleffsen@mpi-bn.mpg.de>"

COPY . /tmp/
COPY scripts /scripts/

# Set the time zone (before installing any packages)
RUN echo 'Europe/Berlin' > apt-get install -y tzdata

# make scripts executeable
RUN chmod +x scripts/bedGraphToBigWig 

# install Fortran compiler 
RUN apt-get update --assume-yes && \
    apt-get install --assume-yes gfortran && \
    apt-get install bedtools && \
    apt-get install -y libcurl4

# install git to check for file changes
RUN apt-get install -y git

# create non-root user
RUN useradd --no-log-init -r -g users user

# setup home directory
WORKDIR /home/user

# change permissions and groups
RUN chown -R user:users /opt && \
    chown -R user:users /tmp && \
    chown -R user:users /scripts && \
    chown -R user:users /home/user && \
    chmod -R 777 /opt && \
    chmod -R 777 /tmp && \
    chmod -R 777 /scripts && \
    chmod -R 777 /home/user

# change the user
USER user

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

# change user to root for cleanup
USER root

# clear tmp
RUN rm -r /tmp/*

# Generate an ssh key
RUN apt-get install -y openssh-client && \
    mkdir .ssh && \
    ssh-keygen -t ed25519 -N "" -f .ssh/id_ed25519

# Switch to non root default user
USER user

