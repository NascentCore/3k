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

type JupyterlabDelLogic struct {
	logx.Logger
	ctx    context.Context
	svcCtx *svc.ServiceContext
}

func NewJupyterlabDelLogic(ctx context.Context, svcCtx *svc.ServiceContext) *JupyterlabDelLogic {
	return &JupyterlabDelLogic{
		Logger: logx.WithContext(ctx),
		ctx:    ctx,
		svcCtx: svcCtx,
	}
}

func (l *JupyterlabDelLogic) JupyterlabDel(req *types.JupyterlabDeleteReq) (resp *types.JupyterlabDeleteResp, err error) {
	JupyterlabModel := l.svcCtx.JupyterlabModel

	instance, err := JupyterlabModel.FindOneByQuery(l.ctx, JupyterlabModel.AllFieldsBuilder().Where(
		squirrel.And{
			squirrel.Eq{"job_name": req.JobName},
			squirrel.Eq{"user_id": req.UserID},
		},
	))
	if err != nil {
		l.Errorf("FindOneByQuery job_name=%d user_id=%d err=%s", req.JobName, req.UserID, err)
		return nil, err
	}

	if instance.Status == model.JupyterStatusStopped {
		resp = &types.JupyterlabDeleteResp{
			Message: fmt.Sprintf("jupyterlab: %s is already stopped", instance.JobName),
		}
		return
	}

	result, err := JupyterlabModel.UpdateColsByCond(l.ctx, JupyterlabModel.UpdateBuilder().Where(
		squirrel.And{
			squirrel.Eq{"job_name": req.JobName},
			squirrel.Eq{"user_id": req.UserID},
		}).SetMap(map[string]interface{}{
		"status":   model.JupyterStatusStopped,
		"end_time": orm.NullTime(time.Now()),
	}))
	if err != nil {
		l.Errorf("update jupyterlab job_name=%s user_id=%d err=%s", req.JobName, req.UserID, err)
		return nil, err
	}

	rows, err := result.RowsAffected()
	if err != nil {
		l.Errorf("RowsAffected job_name=%s user_id=%d err=%s", req.JobName, req.UserID, err)
		return nil, err
	}

	if rows != 1 {
		l.Errorf("RowsAffected rows=%d job_name=%s user_id=%d err=%s", rows, req.JobName, req.UserID, err)
	}

	resp = &types.JupyterlabDeleteResp{Message: fmt.Sprintf("job_name: %s stopped", req.JobName)}

	return
}
