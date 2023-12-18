package handler

import (
	"net/http"

	"sxwl/3k/gomicro/scheduler/internal/logic"
	"sxwl/3k/gomicro/scheduler/internal/svc"
	"sxwl/3k/gomicro/scheduler/internal/types"

	"github.com/zeromicro/go-zero/rest/httpx"
)

func CpodStatusHandler(svcCtx *svc.ServiceContext) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		var req types.CPODStatusReq
		if err := httpx.Parse(r, &req); err != nil {
			httpx.ErrorCtx(r.Context(), w, err)
			return
		}

		l := logic.NewCpodStatusLogic(r.Context(), svcCtx)
		resp, err := l.CpodStatus(&req)
		if err != nil {
			httpx.ErrorCtx(r.Context(), w, err)
		} else {
			httpx.OkJsonCtx(r.Context(), w, resp)
		}
	}
}
