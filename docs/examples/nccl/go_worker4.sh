#!/bin/bash -x

WORLD_SIZE=2 \
        RANK=0 \
        MASTER_PORT=29501 \
        MASTER_ADDR=214.2.5.4 \
                TORCH_CPP_LOG_LEVEL=INFO \
        TORCH_DISTRIBUTED_DEBUG=INFO \
        NCCL_SOCKET_IFNAME=bond0 \
        python ./dist_nccl_demo.py
