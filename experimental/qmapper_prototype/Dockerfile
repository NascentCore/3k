FROM nvidia/cuda:12.2.2-cudnn8-devel-ubuntu22.04

ENV SHELL=/bin/bash
SHELL ["/bin/bash", "-c"]

RUN sed -i s@/archive.ubuntu.com/@/mirrors.aliyun.com/@g /etc/apt/sources.list \
    && sed -i s@/security.ubuntu.com/@/mirrors.aliyun.com/@g /etc/apt/sources.list \
    && apt clean  \
    && apt-get update && \  
    apt-get install -y build-essential ca-certificates curl git libnuma-dev python3 python3-pip wget cmake autoconf automake libtool && \
    wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh && \
    bash Miniconda3-latest-Linux-x86_64.sh -b -p /opt/miniconda && \
    rm Miniconda3-latest-Linux-x86_64.sh && \
    ln -s /opt/miniconda/etc/profile.d/conda.sh /etc/profile.d/conda.sh && \
    echo ". /opt/miniconda/etc/profile.d/conda.sh" >> ~/.bashrc && \
    source /opt/miniconda/etc/profile.d/conda.sh && \
    conda activate base && \
    conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main/ && \
    conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/free/ && \
    conda config --set show_channel_urls yes

RUN source /opt/miniconda/etc/profile.d/conda.sh && \
    conda activate base && \
    conda create -n torch-env python=3.11 && \
    conda activate torch-env && \
    pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple && \
    pip install torch torchvision torchaudio && \
    echo "conda activate torch-env" >> ~/.bashrc


# UCX
RUN git clone https://github.com/openucx/ucx.git /tmp/ucx && \
    cd /tmp/ucx && \
    ./autogen.sh && \
    ./configure --prefix=/usr/local/ucx --with-cuda=/usr/local/cuda && \
    make && \
    make install && \
    git clone https://github.com/openucx/ucc.git /tmp/ucc && \
    cd /tmp/ucc && \
    ./autogen.sh && \
    ./configure --prefix=/usr/local/ucc --with-ucx=/usr/local/ucx && \
    make && \
    make install && \
    wget https://download.open-mpi.org/release/open-mpi/v4.1/openmpi-4.1.6.tar.gz -P /tmp && \
    tar -xzf /tmp/openmpi-4.1.6.tar.gz -C /tmp && \
    cd /tmp/openmpi-4.1.6 && \
    ./configure --prefix=/usr/local/ompi --with-ucx=/usr/local/ucx --enable-mca-no-build=btl-uct --with-cuda --enable-orterun-prefix-by-default && \
    make && \
    make install && \
    rm -rf /tmp/*

COPY ./tvm.tar.gz /tmp

RUN cd /tmp && tar -zxvf tvm.tar.gz && \
    source /opt/miniconda/etc/profile.d/conda.sh && \
    conda activate torch-env && \
    conda install -y numpy pytest cython bzip2 scipy pillow && \
    python -m pip install --upgrade pip && \
    pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple && \
    pip install tornado psutil xgboost scikit-image ml_dtypes decorator cloudpickle attrs -i https://pypi.tuna.tsinghua.edu.cn/simple && \
    pip install matplotlib tabulate -i https://pypi.tuna.tsinghua.edu.cn/simple && \
    cd /tmp/tvm && mkdir build && cd build && \
    cp ../cmake/config.cmake ./ && \
    sed -i 's/set(USE_CUDA OFF)/set(USE_CUDA ON)/g' config.cmake && \
    cmake .. && make -j24 && \
    cd ../python && python setup.py install && \
    rm -rf /opt/miniconda/envs/torch-env/lib/libstdc++.so* && \
    cp /usr/lib/x86_64-linux-gnu/libstdc++.so.6.0.30 /opt/miniconda/envs/torch-env/lib/libstdc++.so && \
    cp /usr/lib/x86_64-linux-gnu/libstdc++.so.6.0.30 /opt/miniconda/envs/torch-env/lib/libstdc++.so.6 && \
    cp /usr/lib/x86_64-linux-gnu/libstdc++.so.6.0.30 /opt/miniconda/envs/torch-env/lib/libstdc++.so.6.0.30

RUN echo "export PATH=\$PATH:/usr/local/ompi/bin" >> ~/.bashrc &&\
    echo "export LD_LIBRARY_PATH=\$LD_LIBRARY_PATH:/usr/local/ompi/lib:/usr/local/cuda-12.2/compat" >> ~/.bashrc

CMD ["sh", "-c", "while true; do echo 'Hello, World!'; sleep 1; done"]