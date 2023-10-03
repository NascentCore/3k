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

* `.golangci.yml` is a symlink for `tools/lint.sh` to run `golangci-lint`,
  because `golangci-lint` expects config file in the top directory of repo.
* Add `// nolint:<linter name>` to disable a check of golangci-lint, for
  example: `// nolint:unused`.
