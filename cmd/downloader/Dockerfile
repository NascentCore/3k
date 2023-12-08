FROM golang:1.21-alpine AS builder

ENV GOPROXY https://goproxy.cn,direct
WORKDIR /build
ADD go.mod .
ADD go.sum .
ADD ./cmd/downloader ./cmd/downloader
ADD ./pkg ./pkg

WORKDIR /build/cmd/downloader
RUN go build -o app main.go

FROM sxwl-registry.cn-beijing.cr.aliyuncs.com/sxwl-ai/alpine-git-lfs:v1

ENV TZ Asia/Shanghai

WORKDIR /app
COPY --from=builder /build/cmd/downloader/app /app/app

ENTRYPOINT ["./app"]
CMD ["-o","/data", "-s", ""]