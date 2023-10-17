#!/bin/bash -x
#
# 该脚本用来启动docker容器
#

IMAGE_NAME="swr.cn-east-3.myhuaweicloud.com/sxwl/for_nccl_test:latest"
DOCKER_NAME="lwn_for_nccl_test"

docker rm -f ${DOCKER_NAME}
docker run -it --rm \
		    --shm-size=1g \
			   --runtime=nvidia \
				  --name ${DOCKER_NAME} \
				  --cap-add=IPC_LOCK \
				  --network host \
				  --device=/dev/infiniband/uverbs0 \
				  --hostname ${DOCKER_NAME} \
				  ${IMAGE_NAME}
