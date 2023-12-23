package handler

import (
	"net/http"

	"github.com/zeromicro/go-zero/rest/httpx"
	"sxwl/3k/gomicro/scheduler/internal/logic"
	"sxwl/3k/gomicro/scheduler/internal/svc"
	"sxwl/3k/gomicro/scheduler/internal/types"
)

func CpodJobHandler(svcCtx *svc.ServiceContext) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		var req types.CPODJobReq
		if err := httpx.Parse(r, &req); err != nil {
			httpx.ErrorCtx(r.Context(), w, err)
			return
		}

		l := logic.NewCpodJobLogic(r.Context(), svcCtx)
		resp, err := l.CpodJob(&req)
		if err != nil {
			httpx.ErrorCtx(r.Context(), w, err)
		} else {
			httpx.OkJsonCtx(r.Context(), w, resp)
		}
	}
}
