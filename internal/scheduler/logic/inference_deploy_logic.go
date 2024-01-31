package logic

import (
	"context"
	"sxwl/3k/internal/scheduler/model"
	"sxwl/3k/pkg/orm"

	"github.com/google/uuid"

	"sxwl/3k/internal/scheduler/svc"
	"sxwl/3k/internal/scheduler/types"

	"github.com/zeromicro/go-zero/core/logx"
)

type InferenceDeployLogic struct {
	logx.Logger
	ctx    context.Context
	svcCtx *svc.ServiceContext
}

func NewInferenceDeployLogic(ctx context.Context, svcCtx *svc.ServiceContext) *InferenceDeployLogic {
	return &InferenceDeployLogic{
		Logger: logx.WithContext(ctx),
		ctx:    ctx,
		svcCtx: svcCtx,
	}
}

func (l *InferenceDeployLogic) InferenceDeploy(req *types.InferenceDeployReq) (resp *types.InferenceDeployResp, err error) {
	InferenceModel := l.svcCtx.InferenceModel

	newUUID, err := uuid.NewRandom()
	if err != nil {
		l.Errorf("new uuid userId: %d err: %s", req.UserID, err)
		return nil, err
	}
	serviceName := "infer-" + newUUID.String()

	infer := &model.SysInference{
		ServiceName: serviceName,
		UserId:      req.UserID,
		ModelId:     orm.NullString(req.ModelId),
	}

	_, err = InferenceModel.Insert(l.ctx, infer)
	if err != nil {
		l.Errorf("insert userId: %d err: %s", req.UserID, err)
		return nil, err
	}

	resp = &types.InferenceDeployResp{ServiceName: serviceName}

	return
}
