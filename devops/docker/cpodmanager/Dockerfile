FROM golang:1.21

WORKDIR /app

COPY . ./
RUN go env -w GO111MODULE=on && go env -w GOPROXY=https://goproxy.cn,direct && \
    go mod download

RUN CGO_ENABLED=0 GOOS=linux go build manager/cmd/cpodmanager/cpodmanager.go
