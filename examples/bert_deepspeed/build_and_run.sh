#!/bin/bash -xe

if [[ $# < 2 ]]; then
	echo "$0 <old-image-version> <new-image-version> ..."
	echo "Need old version and new version, old version is used by current YAML"
	echo "The new version is the one used for building the new container version"
	exit 1
fi

old_version="$1"
new_version="$2"

echo "Building new version ${new_version} ..."
docker build . -t swr.cn-east-3.myhuaweicloud.com/sxwl/bert:${new_version}

echo "Pushing new version ${new_version} ..."
docker push swr.cn-east-3.myhuaweicloud.com/sxwl/bert:${new_version}

echo "Replace ${old_version} with ${new_version}"
sed -i "s/${old_version}/${new_version}/" k8s_config/mpi_bert_ds.yaml

echo "Delete mpijob ..."
kubectl delete mpijob --all

echo "Create mpijob ..."
kubectl apply -f k8s_config/mpi_bert_ds.yaml

echo "Examine MPI job pods ..."
kubectl get pods
