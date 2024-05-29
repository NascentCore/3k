package handler

import (
	"net/http"

	"github.com/zeromicro/go-zero/rest/httpx"
	"sxwl/3k/internal/scheduler/logic"
	"sxwl/3k/internal/scheduler/svc"
)

func CreateNewUserIDHandler(svcCtx *svc.ServiceContext) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		l := logic.NewCreateNewUserIDLogic(r.Context(), svcCtx)
		resp, err := l.CreateNewUserID()
		if err != nil {
			httpx.ErrorCtx(r.Context(), w, err)
		} else {
			httpx.OkJsonCtx(r.Context(), w, resp)
		}
	}
}
