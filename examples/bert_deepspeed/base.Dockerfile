# A base image with OpenMPI

# Matches the host's driver
# NVIDIA-SMI 515.65.01    Driver Version: 515.65.01    CUDA Version: 11.7
FROM nvidia/cuda:11.7.1-devel-ubuntu20.04

RUN apt-get update && apt-get install -y python3-pip && \
    apt-get install -y git wget

# Build OpenMPI
RUN mkdir /tmp/openmpi && \
    cd /tmp/openmpi && \
    wget https://download.open-mpi.org/release/open-mpi/v4.1/openmpi-4.1.5.tar.gz && \
    tar zxf openmpi-4.1.5.tar.gz && \
    cd openmpi-4.1.5 && \
    ./configure --enable-orterun-prefix-by-default && \
    make -j $(nproc) all && \
    make install && \
    ldconfig && \
    rm -rf /tmp/openmpi

# Install SSH
RUN apt-get install -y --no-install-recommends openssh-client openssh-server && \
    mkdir -p /var/run/sshd

# Enable password-less SSH login
RUN sed -i 's/[ #]\(.*StrictHostKeyChecking \).*/ \1no/g' /etc/ssh/ssh_config && \
    echo "    UserKnownHostsFile /dev/null" >> /etc/ssh/ssh_config && \
    sed -i 's/#\(StrictModes \).*/\1no/g' /etc/ssh/sshd_config

# Change pip source to THU source, after changing pip source, use Python3 to
# reinstall pip，then use Python3 to invoke pip.
RUN pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple && \
    python3 -m pip install --no-cache-dir --upgrade pip

# These are required Python packages for running the bert training code.
COPY requirements.txt requirements.txt
RUN python3 -m pip install -r requirements.txt
