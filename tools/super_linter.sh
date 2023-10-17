#!/bin/bash

git_code_path="$1"
git_tot=$(git rev-parse --show-toplevel)
tmp_lint="/tmp/lint"

docker run --rm --env-file ${git_tot}/.github/super_linter.env \
    -e USE_FIND_ALGORITHM=true -e RUN_LOCAL=true \
    -v ${git_tot}/.github:${tmp_lint}/.github \
    -v ${git_tot}/.git:${tmp_lint}/.git \
    -v ${git_tot}/${git_code_path}:${tmp_lint}/${git_code_path} \
    --workdir ${tmp_lint} \
    super-linter
