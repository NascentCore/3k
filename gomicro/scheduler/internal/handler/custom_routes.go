package handler

import (
	"net/http"

	"sxwl/3k/gomicro/scheduler/internal/svc"

	"github.com/zeromicro/go-zero/rest"
)

func RegisterCustomHandlers(server *rest.Server, serverCtx *svc.ServiceContext) {
	server.AddRoutes(
		[]rest.Route{
			{
				Method:  http.MethodGet,
				Path:    "/cpod/job",
				Handler: CpodJobHandler(serverCtx),
			},
			{
				Method:  http.MethodDelete,
				Path:    "/job/delete",
				Handler: JobDeleteHandler(serverCtx),
			},
		},
	)
}
