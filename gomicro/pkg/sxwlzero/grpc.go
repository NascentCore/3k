package sxwlzero

import (
	"context"
	"encoding/json"

	"github.com/zeromicro/go-zero/core/logx"
	"google.golang.org/grpc"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/status"
	"gopkg.in/errgo.v2/errors"
)

// LoggerInterceptor rpc log 拦截器
// 自定义Error转换成gRPC标准错误
func LoggerInterceptor(ctx context.Context, req interface{}, info *grpc.UnaryServerInfo, handler grpc.UnaryHandler) (resp interface{}, err error) {
	resp, err = handler(ctx, req)
	if err != nil {
		causeErr := errors.Cause(err)
		e, ok := causeErr.(*Error)
		if ok {
			bytes, _ := json.Marshal(e)
			err = status.Error(codes.Unknown, string(bytes))
		}

		logx.WithContext(ctx).Errorf("[rpc error] %s", e.Error())
	}

	return
}

// FromGrpcError rpc调用返回的error转换成Error
func FromGrpcError(err error) *Error {
	grpcStatus, ok := status.FromError(err)
	if grpcStatus == nil {
		return nil
	}
	if ok && grpcStatus.Code() == codes.Unknown {
		var e Error
		msg := grpcStatus.Message()
		if json.Unmarshal([]byte(msg), &e) == nil {
			return &e
		}
	}

	return NewError(ErrDefaultCode, err.Error())
}
