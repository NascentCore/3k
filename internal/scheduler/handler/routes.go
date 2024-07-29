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
				Path:    "/api/resource/adapters",
				Handler: ResourceAdaptersHandler(serverCtx),
			},
			{
				Method:  http.MethodGet,
				Path:    "/api/resource/datasets",
				Handler: ResourceDatasetsHandler(serverCtx),
			},
			{
				Method:  http.MethodGet,
				Path:    "/api/resource/models",
				Handler: ResourceModelsHandler(serverCtx),
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

	server.AddRoutes(
		[]rest.Route{
			{
				Method:  http.MethodGet,
				Path:    "/info",
				Handler: userInfoHandler(serverCtx),
			},
			{
				Method:  http.MethodPost,
				Path:    "/login",
				Handler: userLoginHandler(serverCtx),
			},
		},
		rest.WithPrefix("/auth"),
	)

	server.AddRoutes(
		[]rest.Route{
			{
				Method:  http.MethodGet,
				Path:    "/cluster/cpods",
				Handler: ClusterCpodsHandler(serverCtx),
			},
			{
				Method:  http.MethodPost,
				Path:    "/code/sendEmail",
				Handler: sendEmailHandler(serverCtx),
			},
			{
				Method:  http.MethodDelete,
				Path:    "/job/jobs",
				Handler: jobsDelHandler(serverCtx),
			},
			{
				Method:  http.MethodPost,
				Path:    "/job/jupyter/image",
				Handler: jupyterlabImageCreateHandler(serverCtx),
			},
			{
				Method:  http.MethodGet,
				Path:    "/job/jupyter/image",
				Handler: jupyterlabImageListHandler(serverCtx),
			},
			{
				Method:  http.MethodDelete,
				Path:    "/job/jupyter/image",
				Handler: jupyterlabImageDelHandler(serverCtx),
			},
			{
				Method:  http.MethodGet,
				Path:    "/job/jupyter/imageversion",
				Handler: jupyterlabImageVersionListHandler(serverCtx),
			},
			{
				Method:  http.MethodPost,
				Path:    "/job/jupyterlab",
				Handler: jupyterlabCreateHandler(serverCtx),
			},
			{
				Method:  http.MethodGet,
				Path:    "/job/jupyterlab",
				Handler: jupyterlabListHandler(serverCtx),
			},
			{
				Method:  http.MethodDelete,
				Path:    "/job/jupyterlab",
				Handler: jupyterlabDelHandler(serverCtx),
			},
			{
				Method:  http.MethodPut,
				Path:    "/job/jupyterlab",
				Handler: jupyterlabUpdateHandler(serverCtx),
			},
			{
				Method:  http.MethodPost,
				Path:    "/job/jupyterlab/pause",
				Handler: jupyterlabPauseHandler(serverCtx),
			},
			{
				Method:  http.MethodPost,
				Path:    "/job/jupyterlab/resume",
				Handler: jupyterlabResumeHandler(serverCtx),
			},
			{
				Method:  http.MethodPost,
				Path:    "/pay/balance",
				Handler: BalanceAddHandler(serverCtx),
			},
			{
				Method:  http.MethodGet,
				Path:    "/pay/balance",
				Handler: BalanceGetHandler(serverCtx),
			},
			{
				Method:  http.MethodGet,
				Path:    "/pay/billing",
				Handler: BillingListHandler(serverCtx),
			},
			{
				Method:  http.MethodGet,
				Path:    "/pay/billing/tasks",
				Handler: BillingTasksHandler(serverCtx),
			},
			{
				Method:  http.MethodGet,
				Path:    "/pay/recharge",
				Handler: RechargeListHandler(serverCtx),
			},
			{
				Method:  http.MethodGet,
				Path:    "/resource/baseimages",
				Handler: baseImageListHandler(serverCtx),
			},
			{
				Method:  http.MethodPost,
				Path:    "/resource/sync",
				Handler: OssSyncHandler(serverCtx),
			},
			{
				Method:  http.MethodPost,
				Path:    "/user/newuserid",
				Handler: CreateNewUserIDHandler(serverCtx),
			},
			{
				Method:  http.MethodPost,
				Path:    "/user/register/:code",
				Handler: registerHandler(serverCtx),
			},
		},
		rest.WithPrefix("/api"),
	)
}
