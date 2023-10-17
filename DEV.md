# Development

## Common items

* Clone the minimal repo
  ```
  git clone --depth=1 --branch=main --single-branch \
    git@github.com:NascentCore/3k.git
  ```
* Goproxy setup, open your terminal and execute, this allows downloading Golang
  packages from a China proxy.
  ```
  go env -w GO111MODULE=on
  go env -w GOPROXY=https://goproxy.cn,direct
  ```
* Pull request needs to be checked with `tools/lint.sh` before being submitted
  for review.
  ```
  tools/lint.sh
  ```

## Notes

* Add `// nolint:<linter name>` to disable a check of golangci-lint, for
  example: `// nolint:unused`.
* Run super-linter locally:
  ```
  docker run --rm --env-file .github/super_linter.env \
    -e USE_FIND_ALGORITHM=true -e RUN_LOCAL=true \
    -v "$(git rev-parse --show-toplevel)":/tmp/lint \
    super-linter
  ```
