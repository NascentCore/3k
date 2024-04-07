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
				Path:    "/cache/datasets",
				Handler: DatasetsHandler(serverCtx),
			},
			{
				Method:  http.MethodGet,
				Path:    "/cache/models",
				Handler: ModelsHandler(serverCtx),
			},
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
				Path:    "/inference",
				Handler: InferenceDeployHandler(serverCtx),
			},
			{
				Method:  http.MethodGet,
				Path:    "/inference",
				Handler: InferenceInfoHandler(serverCtx),
			},
			{
				Method:  http.MethodDelete,
				Path:    "/inference",
				Handler: InferenceDeleteHandler(serverCtx),
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
				Method:  http.MethodPost,
				Path:    "/node",
				Handler: NodeAddHandler(serverCtx),
			},
			{
				Method:  http.MethodGet,
				Path:    "/node",
				Handler: NodeListHandler(serverCtx),
			},
			{
				Method:  http.MethodPost,
				Path:    "/quota",
				Handler: QuotaAddHandler(serverCtx),
			},
			{
				Method:  http.MethodGet,
				Path:    "/quota",
				Handler: QuotaListHandler(serverCtx),
			},
			{
				Method:  http.MethodPut,
				Path:    "/quota",
				Handler: QuotaUpdateHandler(serverCtx),
			},
			{
				Method:  http.MethodDelete,
				Path:    "/quota",
				Handler: QuotaDeleteHandler(serverCtx),
			},
			{
				Method:  http.MethodGet,
				Path:    "/resource/datasets",
				Handler: ResourceDatasetsHandler(serverCtx),
			},
			{
				Method:  http.MethodGet,
				Path:    "/resource/models",
				Handler: ResourceModelsHandler(serverCtx),
			},
			{
				Method:  http.MethodGet,
				Path:    "/resource/uploader_access",
				Handler: UploaderAccessHandler(serverCtx),
			},
			{
				Method:  http.MethodGet,
				Path:    "/user",
				Handler: UserListHandler(serverCtx),
			},
		},
	)
}
