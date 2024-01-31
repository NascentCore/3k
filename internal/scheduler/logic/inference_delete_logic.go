package logic

import (
	"context"
	"fmt"
	"sxwl/3k/internal/scheduler/model"
	"sxwl/3k/internal/scheduler/svc"
	"sxwl/3k/internal/scheduler/types"
	"sxwl/3k/pkg/orm"
	"time"

	"github.com/Masterminds/squirrel"

	"github.com/zeromicro/go-zero/core/logx"
)

type InferenceDeleteLogic struct {
	logx.Logger
	ctx    context.Context
	svcCtx *svc.ServiceContext
}

func NewInferenceDeleteLogic(ctx context.Context, svcCtx *svc.ServiceContext) *InferenceDeleteLogic {
	return &InferenceDeleteLogic{
		Logger: logx.WithContext(ctx),
		ctx:    ctx,
		svcCtx: svcCtx,
	}
}

func (l *InferenceDeleteLogic) InferenceDelete(req *types.InferenceDeleteReq) (resp *types.InferenceDeleteResp, err error) {
	InferenceModel := l.svcCtx.InferenceModel

	service, err := InferenceModel.FindOneByQuery(l.ctx, InferenceModel.AllFieldsBuilder().Where(
		squirrel.And{
			squirrel.Eq{"service_name": req.ServiceName},
			squirrel.Eq{"user_id": req.UserID},
		},
	))
	if err != nil {
		l.Errorf("FindOneByQuery service_name=%s user_id=%d err=%s", req.ServiceName, req.UserID, err)
		return nil, err
	}

	if service.Status == model.InferStatusStopped {
		resp = &types.InferenceDeleteResp{
			Message: fmt.Sprintf("service_name: %s is already stopped", req.ServiceName),
		}
		return
	}

	result, err := InferenceModel.UpdateColsByCond(l.ctx, InferenceModel.UpdateBuilder().Where(
		squirrel.And{
			squirrel.Eq{"service_name": req.ServiceName},
			squirrel.Eq{"user_id": req.UserID},
		}).SetMap(map[string]interface{}{
		"status":   model.InferStatusStopped,
		"end_time": orm.NullTime(time.Now()),
	}))
	if err != nil {
		l.Errorf("update inference service_name=%s user_id=%d err=%s", req.ServiceName, req.UserID, err)
		return nil, err
	}

	rows, err := result.RowsAffected()
	if err != nil {
		l.Errorf("RowsAffected service_name=%s user_id=%d err=%s", req.ServiceName, req.UserID, err)
		return nil, err
	}

	if rows != 1 {
		l.Errorf("RowsAffected rows=%d service_name=%s user_id=%d err=%s", rows, req.ServiceName, req.UserID, err)
	}

	resp = &types.InferenceDeleteResp{Message: fmt.Sprintf("service_name: %s stopped", req.ServiceName)}

	return
}
