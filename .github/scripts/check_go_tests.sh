#!/bin/bash

# Return 1 if there is missing foo_test.go for foo.go file
# The exceptions are:
# 1. foo.go is < 10 lines
# 2. foo.go has a line of `// NO_TEST_NEEDED`

no_test_needed_marker='// NO_TEST_NEEDED'

echo "Checking all go source file has _test.go"
found_offending_file=0
for go_file in $(find . -name '*.go'); do
  line_count=$(wc -l <${go_file})
  line_count=$((line_count + 0))

  if [[ ${go_file} = *_test.go ]]; then
    continue
  fi

  # If the source file is less than 10 lines, then no test file is needed.
  if [[ ${line_count} < 10 ]]; then
    continue
  fi

  if grep "^${no_test_needed_marker}" ${go_file} -q; then
    continue
  fi

  prefix=${go_file%.go}
  go_test_file="${prefix}_test.go"
  if [ ! -f ${go_test_file} ]; then
    echo "${go_file} needs ${go_test_file}!"
    found_offending_file=1
  fi
done
exit ${found_offending_file}
