# DevOps

Scripts and tools and documentation for working with the codebase in this repo.
Including CI/CD, release, deployment etc.

For example, anyone who wants to contribute to or use 3k, can consult content
here to setup the correct environment.

## Helm Charts release
- 在 3k repo 中加入了 deployment/charts 目录
- 其中 `sx3k` 为父 chart ，其中包含 `sxcloud` 和 `cpodoperator` 两个子 chart
- 子 chart 可通过 `values.yaml` 中的 `enabled` 来控制是否安装
- 发布 release 时只需修改父 chart 下的 `values.yaml`，将镜像更新为新的版本，然后添加相应的 tag 即可自动发布
```bash
git tag -a v1.0.0 -m "Release version 1.0.0"
git push origin v1.0.0
```