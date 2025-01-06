package handler

import (
	"context"
	"fmt"
	"net/http"

	"sxwl/3k/internal/scheduler/logic"
	"sxwl/3k/internal/scheduler/svc"
	"sxwl/3k/internal/scheduler/types"

	"github.com/zeromicro/go-zero/core/logx"
	"github.com/zeromicro/go-zero/rest/httpx"
)

func ChatCompletionsHandler(svcCtx *svc.ServiceContext) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		var req types.CompletionReq
		if err := httpx.Parse(r, &req); err != nil {
			httpx.ErrorCtx(r.Context(), w, err)
			return
		}

		// 如果是流式请求，使用特殊处理
		if req.Completion.Stream {
			w.Header().Set("Content-Type", "text/event-stream")
			w.Header().Set("Cache-Control", "no-cache")
			w.Header().Set("Connection", "keep-alive")
			w.Header().Set("Transfer-Encoding", "chunked")

			// 创建一个新的上下文，包含响应写入器
			ctx := context.WithValue(r.Context(), logic.CtxKeyResponseWriter, w)
			l := logic.NewChatCompletionsLogic(ctx, svcCtx)
			if _, err := l.ChatCompletions(&req); err != nil {
				// 对于流式响应，我们需要以 SSE 格式发送错误
				if _, writeErr := w.Write([]byte(fmt.Sprintf("event: error\ndata: %s\n\n", err.Error()))); writeErr != nil {
					logx.Errorf("failed to write error response is: %v", writeErr)
				}
			}
			return
		}

		// 非流式请求使用普通处理
		l := logic.NewChatCompletionsLogic(r.Context(), svcCtx)
		resp, err := l.ChatCompletions(&req)
		if err != nil {
			httpx.ErrorCtx(r.Context(), w, err)
		} else {
			httpx.OkJsonCtx(r.Context(), w, resp)
		}
	}
}
