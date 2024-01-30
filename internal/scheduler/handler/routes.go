// Code generated by goctl. DO NOT EDIT.
package handler

import (
	"net/http"

	"sxwl/3k/internal/scheduler/svc"

	"github.com/zeromicro/go-zero/rest"
)

func RegisterHandlers(server *rest.Server, serverCtx *svc.ServiceContext) {
	server.AddRoutes(
		[]rest.Route{
			{
				Method:  http.MethodGet,
				Path:    "/cpod/gpu_type",
				Handler: GpuTypeHandler(serverCtx),
			},
			{
				Method:  http.MethodGet,
				Path:    "/cpod/job",
				Handler: CpodJobHandler(serverCtx),
			},
			{
				Method:  http.MethodPost,
				Path:    "/cpod/status",
				Handler: CpodStatusHandler(serverCtx),
			},
			{
				Method:  http.MethodPost,
				Path:    "/inference/deploy",
				Handler: InferenceDeployHandler(serverCtx),
			},
			{
				Method:  http.MethodGet,
				Path:    "/inference/info",
				Handler: InferenceInfoHandler(serverCtx),
			},
			{
				Method:  http.MethodPost,
				Path:    "/info/upload_status",
				Handler: UploadStatusHandler(serverCtx),
			},
			{
				Method:  http.MethodPost,
				Path:    "/job/finetune",
				Handler: FinetuneHandler(serverCtx),
			},
			{
				Method:  http.MethodPost,
				Path:    "/job/job",
				Handler: JobCreateHandler(serverCtx),
			},
			{
				Method:  http.MethodGet,
				Path:    "/job/job",
				Handler: JobGetHandler(serverCtx),
			},
			{
				Method:  http.MethodPost,
				Path:    "/job/status",
				Handler: JobStatusHandler(serverCtx),
			},
			{
				Method:  http.MethodPost,
				Path:    "/job/stop",
				Handler: JobStopHandler(serverCtx),
			},
			{
				Method:  http.MethodGet,
				Path:    "/resource/datasets",
				Handler: DatasetsHandler(serverCtx),
			},
			{
				Method:  http.MethodGet,
				Path:    "/resource/models",
				Handler: ModelsHandler(serverCtx),
			},
		},
	)
}
