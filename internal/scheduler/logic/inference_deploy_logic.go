package logic

import (
	"context"

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
	// todo: add your logic here and delete this line

	return
}
