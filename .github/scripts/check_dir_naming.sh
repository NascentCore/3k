#!/bin/bash

echo "Checking directory names include only lower case chars and '-'"
find home cli manager tools -type d | while read dirname; do
  fname=$(basename ${dirname})
  if ! [[ ${fname} =~ ^[a-z-]+$ ]]; then
    echo "${dirname}"
    exit 1
  fi
done
