FROM golang:1.21-alpine AS builder

ENV GOPROXY https://goproxy.cn,direct
WORKDIR /build
ADD go.mod .
ADD go.sum .
ADD ./cmd/uploader ./cmd/uploader
ADD ./pkg ./pkg

WORKDIR /build/cmd/uploader
RUN go build -o app main.go

FROM alpine:3.19

ENV TZ Asia/Shanghai

WORKDIR /app
COPY --from=builder /build/cmd/uploader/app /app/app

ENTRYPOINT ["./app"]