package handler

import (
	"net/http"

	"github.com/zeromicro/go-zero/rest/httpx"
	"sxwl/3k/internal/scheduler/logic"
	"sxwl/3k/internal/scheduler/svc"
	"sxwl/3k/internal/scheduler/types"
)

func ClusterCpodNamePutHandler(svcCtx *svc.ServiceContext) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		var req types.ClusterCpodNamePutReq
		if err := httpx.Parse(r, &req); err != nil {
			httpx.ErrorCtx(r.Context(), w, err)
			return
		}

		l := logic.NewClusterCpodNamePutLogic(r.Context(), svcCtx)
		resp, err := l.ClusterCpodNamePut(&req)
		if err != nil {
			httpx.ErrorCtx(r.Context(), w, err)
		} else {
			httpx.OkJsonCtx(r.Context(), w, resp)
		}
	}
}
