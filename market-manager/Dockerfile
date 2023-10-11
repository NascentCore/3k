# TODO(yzhao): Test this dockerfile
FROM golang:alpine as builder

WORKDIR /go/src/3k/market-manager

COPY . ./

ENV GO11MODULE=on
ENV GOSUMDB=on
ENV GOPROXY=https://goproxy.cn/

RUN GOOS=linux go build -o market-manager ./cmd/main.go

FROM alpine:latest as final

RUN apk --no-cache add ca-certificates

COPY --from=builder /go/src/3k/market-manager/market-manager /market-manager

CMD ["/market-manager"]