#!/bin/bash

echo "Checking README.md exists in all directories"
found_dirs_missing_readme=false
# shellcheck disable=SC2044
for dir in $(find cli manager tools home -type d); do
  dir_name=$(basename "${dir}")
  if [[ "${dir_name}" != "testdata" && ! -f "${dir}/README.md" ]]; then
    found_dirs_missing_readme=true
    echo "${dir}"
  fi
done

if [[ "${found_dirs_missing_readme}" == "true" ]]; then
  exit 1
fi
