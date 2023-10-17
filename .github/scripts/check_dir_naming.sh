#!/bin/bash

echo "Checking directory names include only lower case chars and '-'"
found_breakage=false
for dirname in $(find home cli manager tools -type d); do
  fname=$(basename "${dirname}")
  if ! [[ ${fname} =~ ^[a-z-]+$ ]]; then
    found_breakage=true
    echo "${dirname}"
  fi
done

if [[ "${found_breakage}" == "true" ]]; then
  exit 1
fi
