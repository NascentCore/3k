#!/bin/bash

echo "Checking markdown files are named with uppercase chars"
found_md_not_upper=false
# shellcheck disable=SC2044
for mdfile in $(find home cli manager tools -name '*.md'); do
  fname=$(basename "${mdfile}")
  if ! [[ ${fname%%.md} =~ ^[A-Z_]+$ ]]; then
    found_md_not_upper=true
    echo "${mdfile}"
  fi
done

if [[ "${found_md_not_upper}" == "true" ]]; then
  exit 1
fi
