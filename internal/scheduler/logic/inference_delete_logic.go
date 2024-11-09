package logic

import (
	"context"
	"sxwl/3k/internal/scheduler/svc"
	"sxwl/3k/internal/scheduler/types"

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
			squirrel.Eq{"new_user_id": req.UserID},
		},
	))
	if err != nil {
		l.Errorf("FindOneByQuery service_name=%s user_id=%s err=%s", req.ServiceName, req.UserID, err)
		return nil, err
	}

	// 删除service记录
	err = InferenceModel.Delete(l.ctx, service.Id)
	if err != nil {
		l.Errorf("Delete inference service_name=%s user_id=%s err=%s", req.ServiceName, req.UserID, err)
		return nil, err
	}

	return &types.InferenceDeleteResp{}, nil
}
