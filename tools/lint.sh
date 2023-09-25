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
echo "Running check_readme ..."
print_divider
.github/scripts/check_readme.sh

echo
print_divider
echo "Running check_markdown_naming ..."
print_divider
.github/scripts/check_markdown_filename.sh

echo
print_divider
echo "Running check_dir_naming ..."
print_divider
.github/scripts/check_dir_naming.sh

echo
print_divider
echo "Running check_todo ..."
print_divider
.github/scripts/check_todo.sh

echo
print_divider
echo "Running check_go_tests ..."
print_divider
.github/scripts/check_go_tests.sh

