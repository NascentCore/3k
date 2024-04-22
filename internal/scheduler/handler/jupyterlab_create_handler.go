package handler

import (
	"net/http"

	"github.com/zeromicro/go-zero/rest/httpx"
	"sxwl/3k/internal/scheduler/logic"
	"sxwl/3k/internal/scheduler/svc"
	"sxwl/3k/internal/scheduler/types"
)

func jupyterlabCreateHandler(svcCtx *svc.ServiceContext) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		var req types.JupyterlabCreateReq
		if err := httpx.Parse(r, &req); err != nil {
			httpx.ErrorCtx(r.Context(), w, err)
			return
		}

		l := logic.NewJupyterlabCreateLogic(r.Context(), svcCtx)
		resp, err := l.JupyterlabCreate(&req)
		if err != nil {
			httpx.ErrorCtx(r.Context(), w, err)
		} else {
			httpx.OkJsonCtx(r.Context(), w, resp)
		}
	}
}
