FROM condaforge/mambaforge

LABEL maintainer="Jan Detleffsen <jan.detleffsen@mpi-bn.mpg.de>"

COPY . /tmp/

RUN mamba update -n base mamba && \
    mamba --version

RUN mamba env update -n base -f /tmp/sctoolbox_env.yml

RUN pip install "/tmp/[all]" 

# Change user to root to clear tmp
RUN rm -r /tmp/*

# Set the time zone
RUN apt-get update && \
    echo 'Europe/Berlin' > apt-get install -y tzdata

# Generate an ssh key
RUN apt-get install -y openssh-client && \
    mkdir .ssh && \
    ssh-keygen -t ed25519 -N "" -f .ssh/id_ed25519

ENV ROOT=TRUE \
    DISABLE_AUTH=TRUE

