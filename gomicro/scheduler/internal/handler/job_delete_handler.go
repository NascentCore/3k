package handler

import (
	"net/http"

	"sxwl/3k/gomicro/scheduler/internal/logic"
	"sxwl/3k/gomicro/scheduler/internal/svc"

	"github.com/zeromicro/go-zero/rest/httpx"
)

func JobDeleteHandler(svcCtx *svc.ServiceContext) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		var req []int64
		if err := httpx.ParseJsonBody(r, &req); err != nil {
			httpx.ErrorCtx(r.Context(), w, err)
			return
		}

		l := logic.NewJobDeleteLogic(r.Context(), svcCtx)
		err := l.JobDelete(req)
		if err != nil {
			httpx.ErrorCtx(r.Context(), w, err)
		} else {
			httpx.OkJsonCtx(r.Context(), w, nil)
		}
	}
}
