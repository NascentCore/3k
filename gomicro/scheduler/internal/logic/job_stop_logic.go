package logic

import (
	"context"

	"sxwl/3k/gomicro/scheduler/internal/svc"
	"sxwl/3k/gomicro/scheduler/internal/types"

	"github.com/zeromicro/go-zero/core/logx"
)

type JobStopLogic struct {
	logx.Logger
	ctx    context.Context
	svcCtx *svc.ServiceContext
}

func NewJobStopLogic(ctx context.Context, svcCtx *svc.ServiceContext) *JobStopLogic {
	return &JobStopLogic{
		Logger: logx.WithContext(ctx),
		ctx:    ctx,
		svcCtx: svcCtx,
	}
}

func (l *JobStopLogic) JobStop(req *types.JobStopReq) (resp *types.JobStopResp, err error) {
	UserJobModel := l.svcCtx.UserJobModel
	err = UserJobModel.DeleteSoftByName(l.ctx, req.JobId)
	if err != nil {
		l.Errorf("soft delete job_name: %s err: %s", req.JobId, err)
		return nil, err
	}

	resp = &types.JobStopResp{Message: "delete job success"}
	return resp, nil
}
