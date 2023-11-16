CGO_ENABLED=0  GOOS=linux  GOARCH=amd64  go build cpodmanager.go
docker build -f ./Dockerfile . -t registry.cn-hangzhou.aliyuncs.com/sxwl-ai/cpodmanager:`git rev-parse --short HEAD`