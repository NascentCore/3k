FROM golang:1.21-alpine AS builder

LABEL stage=gobuilder

ENV CGO_ENABLED 0
ENV GOPROXY https://goproxy.cn,direct
RUN sed -i 's/dl-cdn.alpinelinux.org/mirrors.aliyun.com/g' /etc/apk/repositories

RUN apk update --no-cache && apk add --no-cache tzdata

WORKDIR /build

ADD go.mod .
ADD go.sum .
RUN go mod download
COPY cmd/scheduler cmd/scheduler
COPY internal/scheduler internal/scheduler
COPY pkg pkg
COPY cmd/scheduler/etc /app/etc
COPY cmd/scheduler/ftl /app/ftl
RUN go build -ldflags="-s -w" -o /app/app cmd/scheduler/scheduler.go


FROM alpine:3.19

COPY --from=builder /etc/ssl/certs/ca-certificates.crt /etc/ssl/certs/ca-certificates.crt
COPY --from=builder /usr/share/zoneinfo/Asia/Shanghai /usr/share/zoneinfo/Asia/Shanghai
ENV TZ Asia/Shanghai

WORKDIR /app
COPY --from=builder /app/app /app/app
COPY --from=builder /app/etc /app/etc
COPY --from=builder /app/ftl /app/ftl

CMD ["./app"]
