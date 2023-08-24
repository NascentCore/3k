# Official MPI Operator Base image
FROM mpioperator/base

# mpi-operator mounts the .ssh folder from a Secret. For that to work, we need to disable UserKnownHostsFile to avoid write permissions.
# Disable StrictModes avoids directory and files read permission checks and update system packages & install dependencies
RUN apt-get update && apt-get install -y \
    git \
    wget \
    build-essential \
    cmake \
    libopenmpi-dev \
    openssh-server \
    python3 \
    python3-pip \
    && rm -rf /var/lib/apt/lists/* \
    && echo "    UserKnownHostsFile /dev/null" >> /etc/ssh/ssh_config \
    && sed -i 's/#\(StrictModes \).*/\1no/g' /etc/ssh/sshd_config

# Install DeepSpeed library and Torch with cu11.8 wheels
RUN pip3 config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple \
    && pip3 install deepspeed \
    && pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu117 \
    && pip3 install mpi4py matplotlib

# Create folder for deepspeed workspace
RUN mkdir /deepspeed

# Workspace for DeepSpeed examples
WORKDIR "/deepspeed"

# Clone the DeepSpeedExamples from repository
ADD ./DeepSpeedExamples /deepspeed/DeepSpeedExamples

# Set the working directory to DeepSpeedExamples for models
WORKDIR "/deepspeed/DeepSpeedExamples/"

# Set the default command to bash
CMD ["/bin/bash"]
