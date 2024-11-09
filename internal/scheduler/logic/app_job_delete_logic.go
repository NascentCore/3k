package logic

import (
	"context"
	"fmt"
	"sxwl/3k/internal/scheduler/model"
	"sxwl/3k/pkg/orm"
	"time"

	"sxwl/3k/internal/scheduler/svc"
	"sxwl/3k/internal/scheduler/types"

	"github.com/Masterminds/squirrel"
	"github.com/zeromicro/go-zero/core/logx"
)

type AppJobDeleteLogic struct {
	logx.Logger
	ctx    context.Context
	svcCtx *svc.ServiceContext
}

func NewAppJobDeleteLogic(ctx context.Context, svcCtx *svc.ServiceContext) *AppJobDeleteLogic {
	return &AppJobDeleteLogic{
		Logger: logx.WithContext(ctx),
		ctx:    ctx,
		svcCtx: svcCtx,
	}
}

func (l *AppJobDeleteLogic) AppJobDelete(req *types.AppJobDelReq) (resp *types.BaseResp, err error) {
	AppJobModel := l.svcCtx.AppJobModel

	instance, err := AppJobModel.FindOneByQuery(l.ctx, AppJobModel.AllFieldsBuilder().Where(
		squirrel.And{
			squirrel.Eq{"job_name": req.JobName},
			squirrel.Eq{"user_id": req.UserID},
		},
	))
	if err != nil {
		l.Errorf("FindOneByQuery job_name=%s user_id=%d err=%s", req.JobName, req.UserID, err)
		return nil, err
	}

	if instance.Status == model.StatusStopped {
		resp = &types.BaseResp{
			Message: fmt.Sprintf("app job: %s is already stopped", instance.JobName),
		}
		return
	}

	result, err := AppJobModel.UpdateColsByCond(l.ctx, AppJobModel.UpdateBuilder().Where(
		squirrel.And{
			squirrel.Eq{"job_name": req.JobName},
			squirrel.Eq{"user_id": req.UserID},
		}).SetMap(map[string]interface{}{
		"status":   model.StatusStopped,
		"end_time": orm.NullTime(time.Now()),
	}))
	if err != nil {
		l.Errorf("update app job job_name=%s user_id=%s err=%s", req.JobName, req.UserID, err)
		return nil, err
	}

	rows, err := result.RowsAffected()
	if err != nil {
		l.Errorf("RowsAffected job_name=%s user_id=%s err=%s", req.JobName, req.UserID, err)
		return nil, err
	}

	if rows != 1 {
		l.Errorf("RowsAffected rows=%d job_name=%s user_id=%s err=%s", rows, req.JobName, req.UserID, err)
	}

	resp = &types.BaseResp{Message: fmt.Sprintf("job_name: %s stopped", req.JobName)}

	return
}
