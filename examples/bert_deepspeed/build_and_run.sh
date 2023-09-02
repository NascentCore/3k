#!/bin/bash -xe

if [[ $# < 2 ]]; then
	echo "$0 <version> ..."
	echo "Need version"
	exit 1
fi

old_version="$1"
new_version="$2"

docker build . -t swr.cn-east-3.myhuaweicloud.com/sxwl/bert:${new_version}
docker push swr.cn-east-3.myhuaweicloud.com/sxwl/bert:${new_version}
echo "Replace ${old_version} with ${new_version}"
sed "s/${old_version}/${new_version}/" k8s_config/mpi_bert_ds.yaml

echo "Delete mpijob ..."
kubectl delete mpijob --all

echo "Create mpijob ..."
kubectl apply -f k8s_config/mpi_bert_ds.yaml

kubectl get pods
