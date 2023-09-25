#!/bin/bash -xe

# Return 1 if there is missing foo_test.go for foo.go file
# The exceptions are:
# 1. foo.go is < 10 lines
# 2. foo.go has a line of `// NO_TEST_NEEDED`

no_test_needed_marker='// NO_TEST_NEEDED'

for go_file in $(find . -name '*.go'); do
  echo "go_file: ${go_file}"
  line_count=$(wc -l <${go_file})
  line_count=$((line_count + 0))
  if [[ ${line_count} < 10 ]]; then
    echo "${go_file} has less than 10 lines"
    continue
  fi
  if grep "^${no_test_needed_marker}" ${go_file}; then
    echo "${go_file} has '${no_test_needed_marker}'"
    continue
  fi
  if [[ ${go_file} = *_test.go ]]; then
    echo "${go_file} is a test file"
    continue
  fi

  prefix=${go_file%.go}
  go_test_file="${prefix}_test.go"
  if [ ! -f ${go_test_file} ]; then
    echo "${go_file} needs ${go_test_file}!"
    exit 1
  fi
done
