#!/bin/bash

echo "Checking directory names include only lower case chars and '-'"
found_breakage=false
# shellcheck disable=SC2044
for dirname in $(find . -type d); do
  fname=$(basename "${dirname}")
  if [[ ${fname} == "ISSUE_TEMPLATE" ]]; then
    # Skip GitHub ISSUE_TEMPLATE
    continue
  fi
  if ! [[ ${fname} =~ [.0-9a-z-]+ ]]; then
    found_breakage=true
    echo "${dirname}"
  fi
done

if [[ "${found_breakage}" == "true" ]]; then
  exit 1
fi
