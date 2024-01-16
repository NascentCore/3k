package logic

import (
	"context"
	"sxwl/3k/internal/scheduler/model"
	"sxwl/3k/internal/scheduler/svc"
	"sxwl/3k/internal/scheduler/types"
	"sxwl/3k/pkg/consts"

	"github.com/Masterminds/squirrel"
	"github.com/zeromicro/go-zero/core/logx"
)

type JobStatusLogic struct {
	logx.Logger
	ctx    context.Context
	svcCtx *svc.ServiceContext
}

func NewJobStatusLogic(ctx context.Context, svcCtx *svc.ServiceContext) *JobStatusLogic {
	return &JobStatusLogic{
		Logger: logx.WithContext(ctx),
		ctx:    ctx,
		svcCtx: svcCtx,
	}
}

func (l *JobStatusLogic) JobStatus(req *types.JobStatusReq) (resp *types.JobStatusResp, err error) {
	UserJobModel := l.svcCtx.UserJobModel
	FileURLModel := l.svcCtx.FileURLModel

	job, err := UserJobModel.FindOneByQuery(l.ctx, UserJobModel.AllFieldsBuilder().Where(squirrel.Eq{
		"job_name": req.JobId,
	}))
	if err != nil {
		l.Logger.Errorf("find job_id: %s err: %s", req.JobId, err)
		return nil, err
	}

	resp = &types.JobStatusResp{}

	switch job.WorkStatus {
	case model.JobStatusWorkerUrlSuccess:
		fileURL, err := FileURLModel.FindOneByQuery(l.ctx,
			FileURLModel.AllFieldsBuilder().Where(
				squirrel.Eq{
					"job_name": req.JobId,
				},
			),
		)
		if err != nil {
			l.Logger.Errorf("file_url findOne job_name=%s err=%s", req.JobId, err)
			return nil, err
		}
		resp.URL = fileURL.FileUrl.String
		resp.Status = consts.JobSuccess
	case model.JobStatusWorkerFail:
		resp.Status = consts.JobFail
	default:
		resp.Status = consts.JobWorking
	}

	return resp, nil
}
