package handler

import (
	"fmt"
	"net/http"

	"sxwl/3k/internal/scheduler/svc"
	"sxwl/3k/internal/scheduler/types"

	"github.com/zeromicro/go-zero/rest/httpx"
)

func DingtalkCallbackHandler(svcCtx *svc.ServiceContext) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		var req types.DingCallbackReq
		if err := httpx.Parse(r, &req); err != nil {
			httpx.ErrorCtx(r.Context(), w, err)
			return
		}

		if req.Code == "" {
			http.Error(w, "Invalid request, missing code", http.StatusBadRequest)
			return
		}

		// 重定向回前端页面，并通过 URL 参数传递用户信息
		redirectURL := fmt.Sprintf("/user/dingtalk-callback?code=%s", req.Code)
		http.Redirect(w, r, redirectURL, http.StatusFound)
	}
}
