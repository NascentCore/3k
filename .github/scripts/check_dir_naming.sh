#!/bin/bash

echo "Checking directory names include only lower case chars and '-'"
# Declare variable outside of while loop, and modify its value inside
# the loop does not work, as while loop executes in a sub-shell
find home cli manager tools -type d | while read dirname; do
  fname=$(basename ${dirname})
  if ! [[ ${fname} =~ ^[a-z-]+$ ]]; then
    echo "${dirname}"
    exit 1
  fi
done
