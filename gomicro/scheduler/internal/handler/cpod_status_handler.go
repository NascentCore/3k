package handler

import (
	"net/http"

	"github.com/zeromicro/go-zero/rest/httpx"
	"sxwl/3k/gomicro/pkg/sxwlzero"
	"sxwl/3k/gomicro/scheduler/internal/logic"
	"sxwl/3k/gomicro/scheduler/internal/svc"
	"sxwl/3k/gomicro/scheduler/internal/types"
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
			sxwlzero.OkJsonCtx(r.Context(), w, resp)
		}
	}
}