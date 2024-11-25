package logic

import (
	"context"

	"sxwl/3k/internal/scheduler/model"
	"sxwl/3k/internal/scheduler/svc"
	"sxwl/3k/internal/scheduler/types"

	"github.com/Masterminds/squirrel"
	"github.com/zeromicro/go-zero/core/logx"
)

type GpuJobStatusLogic struct {
	logx.Logger
	ctx    context.Context
	svcCtx *svc.ServiceContext
}

func NewGpuJobStatusLogic(ctx context.Context, svcCtx *svc.ServiceContext) *GpuJobStatusLogic {
	return &GpuJobStatusLogic{
		Logger: logx.WithContext(ctx),
		ctx:    ctx,
		svcCtx: svcCtx,
	}
}

func (l *GpuJobStatusLogic) GpuJobStatus(req *types.GPUJobStatusReq) (resp *types.GPUJobStatusResp, err error) {
	UserJobModel := l.svcCtx.UserJobModel

	job, err := UserJobModel.FindOneByQuery(l.ctx, UserJobModel.AllFieldsBuilder().Where(squirrel.Eq{
		"job_name": req.JobId,
	}))
	if err != nil {
		l.Logger.Errorf("find job: %s err: %s", req.JobId, err)
		return nil, ErrDBFind
	}

	// 检查是否是该用户的任务
	if job.NewUserId != req.UserID {
		return nil, ErrPermissionDenied
	}

	// 查询推理任务状态并返回
	resp = &types.GPUJobStatusResp{}
	resp.JobId = job.JobName.String
	resp.Status = model.StatusToStr[job.WorkStatus]

	return
}
