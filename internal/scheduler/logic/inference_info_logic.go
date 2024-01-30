package logic

import (
	"context"

	"sxwl/3k/internal/scheduler/svc"
	"sxwl/3k/internal/scheduler/types"

	"github.com/zeromicro/go-zero/core/logx"
)

type InferenceInfoLogic struct {
	logx.Logger
	ctx    context.Context
	svcCtx *svc.ServiceContext
}

func NewInferenceInfoLogic(ctx context.Context, svcCtx *svc.ServiceContext) *InferenceInfoLogic {
	return &InferenceInfoLogic{
		Logger: logx.WithContext(ctx),
		ctx:    ctx,
		svcCtx: svcCtx,
	}
}

func (l *InferenceInfoLogic) InferenceInfo(req *types.InferenceInfoReq) (resp *types.InferenceInfoResp, err error) {
	// todo: add your logic here and delete this line

	return
}
