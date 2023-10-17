#!/bin/bash

function print_divider() {
  echo "============================"
}

echo
print_divider
echo "Running golangci-lint ..."
print_divider
golangci-lint run --fix --config=.github/linters/.golangci.yml

echo
print_divider
echo "Running check_add ..."
print_divider
.github/scripts/check_all.sh
