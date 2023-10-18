#!/bin/bash

echo "Checking dockerfile naming"
found_breakage=false
# shellcheck disable=SC2044
for fpath in $(find . -type f -iname dockerfile); do
  fname=$(basename "${fpath}")
  if [[ "${fname}" != "Dockerfile" ]]; then
    found_breakage=true
    echo "${fpath}"
  fi
done

if [[ "${found_breakage}" == "true" ]]; then
  exit 1
fi
