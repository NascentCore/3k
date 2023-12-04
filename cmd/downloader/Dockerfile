FROM golang:1.21-alpine AS builder

ENV GOPROXY https://goproxy.cn,direct
WORKDIR /build
ADD go.mod .
ADD go.sum .
ADD ./cmd/downloader ./cmd/downloader

WORKDIR /build/cmd/downloader
RUN go build -o app main.go

FROM alpine/git

ENV TZ Asia/Shanghai

WORKDIR /app
COPY --from=builder /build/cmd/downloader/app /app/app

ENTRYPOINT ["./app"]
CMD ["-o","/data", "-s", ""]