FROM python:3.10.12

LABEL maintainer="Jan Detleffsen <jan.detleffsen@mpi-bn.mpg.de>"

ARG INSTALL_ALL=TRUE
ARG INSTALL_JUPYTER=TRUE

COPY sc_framework /app

RUN apt-get update && \
    apt-get clean

# Install Python packages
RUN pip install /app[all]

# Install Jupyter if needed
RUN if [ "$INSTALL_JUPYTER" = "TRUE" ]; then \
        pip install jupyter notebook; \
    fi

# Set the time zone
RUN sh -c "echo 'Europe/Berlin' > /etc/timezone" && \
    dpkg-reconfigure -f noninteractive tzdata

# Generate an ssh key
RUN mkdir .ssh && \
    ssh-keygen -t ed25519 -N "" -f .ssh/id_ed25519

ENV ROOT=TRUE \
    DISABLE_AUTH=TRUE

