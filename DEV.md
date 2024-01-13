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
* Init submodule:
  ```
  git submodule update --init --recursive
  ```

## Notes

* Add `// nolint:<linter name>` to disable a check of golangci-lint, for
  example: `// nolint:unused`.
* Run super-linter locally:
  ```
  # `--workdir /tmp/lint` is needed per
  # https://github.com/super-linter/super-linter/issues/4495
  docker run --rm --env-file .github/super_linter.env \
    -e USE_FIND_ALGORITHM=true -e RUN_LOCAL=true \
    -v $(pwd)/.github:/tmp/lint/.github \
    -v $(pwd)/.git:/tmp/lint/.git \
    -v $(pwd)/<code-path>:/tmp/lint/<code-path> \
    --workdir /tmp/lint \
    super-linter
  ```

## Pip mirro

```
python3 -m pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple
```

## File and dir naming convention

- OK to break conventions
- The most important rule is to keep consistent with the dominant convention in the existing codebase

Use '-' to separate file and dir name components, as in foo-bar/baz-tik-tok
