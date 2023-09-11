#!/bin/bash -xe

if [[ $# < 1 ]]; then
  echo "Need namespace, $0 <namespace> ..."
  exit 1
fi

ns="$1"

kubectl get namespace "${ns}" -o json |\
  tr -d "\n" |\
  sed "s/\"finalizers\": \[[^]]\+\]/\"finalizers\": []/" |\
  kubectl replace --raw /api/v1/namespaces/"${ns}"/finalize -f -
