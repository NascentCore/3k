# Development

## Common items

* Goproxy setup, open your terminal and execute, this allows downloading Golang packages from a China proxy.
  ```
  go env -w GO111MODULE=on
  go env -w GOPROXY=https://goproxy.cn,direct
  ```
* Pull request needs to be checked with `tools/lint.sh` before being submitted for review.
  ```
  tools/lint.sh
  ```

## Notes

* `.golangci.yml` is a symlink for `tools/lint.sh` to run `golangci-lint`, because `golangci-lint` expects
  config file in current directory.
