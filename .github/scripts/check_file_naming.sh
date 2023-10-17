#!/bin/bash

echo "Checking file names include only lower case chars and '-'"
found_breakage=false
for filename in $(find home cli manager tools -type f); do
  fname=$(basename "${filename}")
  if ! [[ ${fname} =~ ^[a-zA-Z0-9_\.]+$ ]]; then
    found_breakage=true
    echo "${filename}"
  fi
done

if [[ "${found_breakage}" == "true" ]]; then
  exit 1
fi
