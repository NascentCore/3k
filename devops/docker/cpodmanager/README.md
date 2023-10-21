# CpodManager

Build cpod manager docker container image
TODO(yzhao): Change to use bazel

```
cd "$(git rev-parse --show-toplevel)"
docker build -f devops/docker/cpodmanager/Dockerfile . -t cpodmanager
docker run cpodmanager /app/cpodmanager
```
