package handler

import (
	"net/http"

	"sxwl/3k/internal/scheduler/svc"

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
				Path:    "/job/job",
				Handler: JobDeleteHandler(serverCtx),
			},
		},
	)
}
