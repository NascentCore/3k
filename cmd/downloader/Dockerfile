# 使用 Alpine Linux 作为基础镜像
FROM --platform=linux/amd64 golang:alpine:v1.21.1 AS builder

# 安装 Git
RUN apk --no-cache add git

WORKDIR /build

ADD ../../go.mod .
ADD ../../go.sum .
ADD . ./cmd/downloader

WORKDIR /build/cmd/downloader
RUN make build

FROM scratch

ENV TZ Asia/Shanghai

WORKDIR /app
COPY --from=builder /build/cmd/downloader/app /app/app

CMD ["./app", "git", "-s", "https://www.modelscope.cn/qwen/Qwen-72B-Chat.git"]
