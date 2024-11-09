package logic

import (
	"context"
	"fmt"
	"time"

	"sxwl/3k/internal/scheduler/model"
	"sxwl/3k/internal/scheduler/svc"
	"sxwl/3k/internal/scheduler/types"
	"sxwl/3k/pkg/orm"

	"github.com/Masterminds/squirrel"
	"github.com/zeromicro/go-zero/core/logx"
)

type InferenceStopLogic struct {
	logx.Logger
	ctx    context.Context
	svcCtx *svc.ServiceContext
}

func NewInferenceStopLogic(ctx context.Context, svcCtx *svc.ServiceContext) *InferenceStopLogic {
	return &InferenceStopLogic{
		Logger: logx.WithContext(ctx),
		ctx:    ctx,
		svcCtx: svcCtx,
	}
}

func (l *InferenceStopLogic) InferenceStop(req *types.InferenceStopReq) (resp *types.BaseResp, err error) {
	InferenceModel := l.svcCtx.InferenceModel

	service, err := InferenceModel.FindOneByQuery(l.ctx, InferenceModel.AllFieldsBuilder().Where(
		squirrel.And{
			squirrel.Eq{"service_name": req.ServiceName},
			squirrel.Eq{"new_user_id": req.UserID},
		},
	))
	if err != nil {
		l.Errorf("FindOneByQuery service_name=%s user_id=%s err=%s", req.ServiceName, req.UserID, err)
		return nil, err
	}

	if service.Status == model.StatusStopped {
		resp = &types.BaseResp{
			Message: fmt.Sprintf("service_name: %s is already stopped", req.ServiceName),
		}
		return
	}

	result, err := InferenceModel.UpdateColsByCond(l.ctx, InferenceModel.UpdateBuilder().Where(
		squirrel.And{
			squirrel.Eq{"service_name": req.ServiceName},
			squirrel.Eq{"new_user_id": req.UserID},
		}).SetMap(map[string]interface{}{
		"status":        model.StatusStopped,
		"obtain_status": model.StatusObtainNotNeedSend,
		"end_time":      orm.NullTime(time.Now()),
	}))
	if err != nil {
		l.Errorf("update inference service_name=%s user_id=%s err=%s", req.ServiceName, req.UserID, err)
		return nil, err
	}

	rows, err := result.RowsAffected()
	if err != nil {
		l.Errorf("RowsAffected service_name=%s user_id=%s err=%s", req.ServiceName, req.UserID, err)
		return nil, err
	}

	if rows != 1 {
		l.Errorf("RowsAffected rows=%d service_name=%s user_id=%s err=%s", rows, req.ServiceName, req.UserID, err)
	}

	resp = &types.BaseResp{Message: fmt.Sprintf("service_name: %s stopped", req.ServiceName)}

	return
}
