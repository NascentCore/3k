package handler

import (
	"context"
	"net/http"

	"github.com/zeromicro/go-zero/core/logx"
	"github.com/zeromicro/go-zero/rest/httpx"
)

type response struct {
	Message string `json:"message"`
}

func InitErrorHandler() {
	httpx.SetErrorHandlerCtx(func(ctx context.Context, err error) (int, any) {
		errMsg := err.Error()

		logx.WithContext(ctx).Errorf("[api error] errMsg=%s", errMsg)

		return http.StatusBadRequest, response{
			Message: errMsg,
		}
	})
}
