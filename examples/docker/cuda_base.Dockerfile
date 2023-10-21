# A base image with CUDA, OpenMPI, SSHD

# Matches the host's driver
# NVIDIA-SMI 515.65.01    Driver Version: 515.65.01    CUDA Version: 11.7
FROM nvidia/cuda:11.7.1-devel-ubuntu20.04

RUN apt-get update && \
    apt-get install -y --no-install-recommends git wget python3-pip \
    openssh-client openssh-server && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

RUN mkdir -p /var/run/sshd

# Enable password-less SSH login
# Needed by launching with OpenMPI
RUN sed -i 's/[ #]\(.*StrictHostKeyChecking \).*/ \1no/g' /etc/ssh/ssh_config && \
    echo "    UserKnownHostsFile /dev/null" >> /etc/ssh/ssh_config && \
    sed -i 's/#\(StrictModes \).*/\1no/g' /etc/ssh/sshd_config

# Build OpenMPI
RUN mkdir /tmp/openmpi && \
    cd /tmp/openmpi && \
    wget --quiet https://download.open-mpi.org/release/open-mpi/v4.1/openmpi-4.1.5.tar.gz && \
    tar zxf openmpi-4.1.5.tar.gz && \
    cd openmpi-4.1.5 && \
    ./configure --enable-orterun-prefix-by-default && \
    make -j $(nproc) all && \
    make install && \
    ldconfig && \
    rm -rf /tmp/openmpi

