#!/bin/bash

echo "Checking dockerfile naming"
found_breakage=false
# shellcheck disable=SC2044
for fpath in $(find . -type f -name dockerfile); do
  found_breakage=true
  echo "${fpath}"
done

if [[ "${found_breakage}" == "true" ]]; then
  exit 1
fi
