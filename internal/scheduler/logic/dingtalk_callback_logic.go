package logic

import (
	"context"

	"sxwl/3k/internal/scheduler/svc"
	"sxwl/3k/internal/scheduler/types"

	"github.com/zeromicro/go-zero/core/logx"
)

type DingtalkCallbackLogic struct {
	logx.Logger
	ctx    context.Context
	svcCtx *svc.ServiceContext
}

func NewDingtalkCallbackLogic(ctx context.Context, svcCtx *svc.ServiceContext) *DingtalkCallbackLogic {
	return &DingtalkCallbackLogic{
		Logger: logx.WithContext(ctx),
		ctx:    ctx,
		svcCtx: svcCtx,
	}
}

func (l *DingtalkCallbackLogic) DingtalkCallback(req *types.DingCallbackReq) error {
	// todo: add your logic here and delete this line

	return nil
}
