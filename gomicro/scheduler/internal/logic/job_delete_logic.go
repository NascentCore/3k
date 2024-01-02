package logic

import (
	"context"

	"sxwl/3k/gomicro/scheduler/internal/svc"
	"sxwl/3k/gomicro/scheduler/internal/types"

	"github.com/zeromicro/go-zero/core/logx"
)

type JobDeleteLogic struct {
	logx.Logger
	ctx    context.Context
	svcCtx *svc.ServiceContext
}

func NewJobDeleteLogic(ctx context.Context, svcCtx *svc.ServiceContext) *JobDeleteLogic {
	return &JobDeleteLogic{
		Logger: logx.WithContext(ctx),
		ctx:    ctx,
		svcCtx: svcCtx,
	}
}

func (l *JobDeleteLogic) JobDelete(req *types.JobDeleteReq) (resp *types.JobDeleteResp, err error) {
	UserJobModel := l.svcCtx.UserJobModel
	err = UserJobModel.DeleteSoftByName(l.ctx, req.JobId)
	if err != nil {
		l.Errorf("soft delete job_name: %s err: %s", req.JobId, err)
		return nil, err
	}

	resp = &types.JobDeleteResp{Message: "delete job success"}
	return resp, nil
}
