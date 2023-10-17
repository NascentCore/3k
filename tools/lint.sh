#!/bin/bash

function print_divider() {
  echo "============================"
}

echo
print_divider
echo "Running golangci-lint ..."
print_divider
# Run golangci-lint and fix issues. Config file is .golangci.yml
golangci-lint run --fix

echo
print_divider
echo "Running check_add ..."
print_divider
.github/scripts/check_all.sh
