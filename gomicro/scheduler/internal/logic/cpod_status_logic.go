package logic

import (
	"context"

	"sxwl/3k/gomicro/scheduler/internal/svc"
	"sxwl/3k/gomicro/scheduler/internal/types"

	"github.com/zeromicro/go-zero/core/logx"
)

type CpodStatusLogic struct {
	logx.Logger
	ctx    context.Context
	svcCtx *svc.ServiceContext
}

func NewCpodStatusLogic(ctx context.Context, svcCtx *svc.ServiceContext) *CpodStatusLogic {
	return &CpodStatusLogic{
		Logger: logx.WithContext(ctx),
		ctx:    ctx,
		svcCtx: svcCtx,
	}
}

func (l *CpodStatusLogic) CpodStatus(req *types.CPODStatusReq) (resp *types.CPODStatusResp, err error) {
	// todo: add your logic here and delete this line

	resp = &types.CPODStatusResp{}

	return
}
