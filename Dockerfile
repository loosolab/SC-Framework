FROM mambaorg/micromamba:latest

LABEL maintainer="Jan Detleffsen <jan.detleffsen@mpi-bn.mpg.de>"

COPY . /tmp

#RUN apt-get update && \
#    apt-get clean
USER root
# Install Python packages \
RUN apt-get update && \
    apt-get install -y openssh-client && \
    apt-get install -y python3-pip && \
    apt-get clean \
USER mambauser

RUN micromamba install -y -n base -f /tmp/sctoolbox_env.yml && \
    micromamba clean --all --yes && \
    pip install /tmp/.

# Set the time zone
RUN sh -c "echo 'Europe/Berlin' > /etc/timezone" && \
    dpkg-reconfigure -f noninteractive tzdata

# Generate an ssh key
RUN mkdir .ssh && \
    ssh-keygen -t ed25519 -N "" -f .ssh/id_ed25519

ENV ROOT=TRUE \
    DISABLE_AUTH=TRUE

