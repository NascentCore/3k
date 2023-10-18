#!/bin/bash

if [[ $# -lt 1 ]]; then
  echo "$0 <target-to-lint/file-or-directory>"
  exit 1
fi

target="$1"
tot=$(git rev-parse --show-toplevel)
tmp_lint="/tmp/lint"

docker run --rm --env-file ${tot}/.github/super_linter.env \
    -e USE_FIND_ALGORITHM=true -e RUN_LOCAL=true \
    -v ${tot}/.github/super_linter.env:${tmp_lint}/.github/super_linter.env \
    -v ${tot}/.github/linters:${tmp_lint}/.github/linters \
    -v ${tot}/.git:${tmp_lint}/.git \
    -v ${tot}/${target}:${tmp_lint}/${target} \
    --workdir ${tmp_lint} \
    super-linter
