package sxwlzero

import (
	"context"
	"net/http"
	"time"

	"github.com/zeromicro/go-zero/core/logx"
	"github.com/zeromicro/go-zero/rest/httpx"
)

type response struct {
	Status    uint32 `json:"status"`
	Content   any    `json:"content,omitempty"`
	ErrMsg    string `json:"errMsg,omitempty"`
	Timestamp int64  `json:"timestamp"`
}

func OkJsonCtx(ctx context.Context, w http.ResponseWriter, v any) {
	httpx.OkJsonCtx(ctx, w, response{
		Status:    http.StatusOK,
		Content:   v,
		Timestamp: time.Now().Unix(),
	})
}

func SetErrorHandlerCtx() {
	httpx.SetErrorHandlerCtx(func(ctx context.Context, err error) (int, any) {
		errCode := ErrDefaultCode
		errMsg := err.Error()

		e, ok := err.(*Error)
		if ok {
			errCode = e.ErrCode
			errMsg = e.ErrMsg
		}

		logx.WithContext(ctx).Errorf("[api error] errCode=%s errMsg=%s", errCode, errMsg)

		return http.StatusOK, response{
			Status:    errCode,
			ErrMsg:    errMsg,
			Timestamp: time.Now().Unix(),
		}
	})
}
