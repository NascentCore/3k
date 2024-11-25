package logic

import (
	"context"
	"fmt"

	"sxwl/3k/internal/scheduler/model"
	"sxwl/3k/internal/scheduler/svc"
	"sxwl/3k/internal/scheduler/types"

	"github.com/Masterminds/squirrel"
	"github.com/zeromicro/go-zero/core/logx"
)

type GpuJobStopLogic struct {
	logx.Logger
	ctx    context.Context
	svcCtx *svc.ServiceContext
}

func NewGpuJobStopLogic(ctx context.Context, svcCtx *svc.ServiceContext) *GpuJobStopLogic {
	return &GpuJobStopLogic{
		Logger: logx.WithContext(ctx),
		ctx:    ctx,
		svcCtx: svcCtx,
	}
}

func (l *GpuJobStopLogic) GpuJobStop(req *types.GPUJobStopReq) (resp *types.BaseResp, err error) {
	UserJobModel := l.svcCtx.UserJobModel

	job, err := UserJobModel.FindOneByQuery(l.ctx, UserJobModel.AllFieldsBuilder().Where(
		squirrel.And{
			squirrel.Eq{"job_name": req.JobId},
			squirrel.Eq{"new_user_id": req.UserID},
		},
	))
	if err != nil {
		l.Errorf("FindOneByQuery job_name=%s user_id=%s err=%s", req.JobId, req.UserID, err)
		return nil, err
	}

	if job.WorkStatus == model.StatusStopped {
		resp = &types.BaseResp{
			Message: fmt.Sprintf("job_name: %s is already stopped", req.JobId),
		}
		return
	}

	result, err := UserJobModel.UpdateColsByCond(l.ctx, UserJobModel.UpdateBuilder().Where(
		squirrel.And{
			squirrel.Eq{"job_name": req.JobId},
			squirrel.Eq{"new_user_id": req.UserID},
		}).SetMap(map[string]interface{}{
		"work_status":   model.StatusStopped,
		"obtain_status": model.StatusObtainNotNeedSend,
	}))
	if err != nil {
		l.Errorf("update job job_name=%s user_id=%s err=%s", req.JobId, req.UserID, err)
		return nil, err
	}

	rows, err := result.RowsAffected()
	if err != nil {
		l.Errorf("RowsAffected job_name=%s user_id=%s err=%s", req.JobId, req.UserID, err)
		return nil, err
	}

	if rows != 1 {
		l.Errorf("RowsAffected rows=%d job_name=%s user_id=%s err=%s", rows, req.JobId, req.UserID, err)
	}

	resp = &types.BaseResp{Message: fmt.Sprintf("job_name: %s stopped", req.JobId)}

	return
}
