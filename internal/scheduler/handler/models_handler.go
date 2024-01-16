package handler

import (
	"net/http"

	"sxwl/3k/internal/scheduler/logic"
	"sxwl/3k/internal/scheduler/svc"

	"github.com/zeromicro/go-zero/rest/httpx"
)

func ModelsHandler(svcCtx *svc.ServiceContext) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		l := logic.NewModelsLogic(r.Context(), svcCtx)
		resp, err := l.Models()
		if err != nil {
			httpx.ErrorCtx(r.Context(), w, err)
		} else {
			httpx.OkJsonCtx(r.Context(), w, resp)
		}
	}
}
