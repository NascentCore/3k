package logic

import (
	"context"

	"sxwl/3k/internal/scheduler/svc"
	"sxwl/3k/internal/scheduler/types"

	"github.com/zeromicro/go-zero/core/logx"
)

type AdapterByNameLogic struct {
	logx.Logger
	ctx    context.Context
	svcCtx *svc.ServiceContext
}

func NewAdapterByNameLogic(ctx context.Context, svcCtx *svc.ServiceContext) *AdapterByNameLogic {
	return &AdapterByNameLogic{
		Logger: logx.WithContext(ctx),
		ctx:    ctx,
		svcCtx: svcCtx,
	}
}

func (l *AdapterByNameLogic) AdapterByName(req *types.AdapterByNameReq) (resp *types.Adapter, err error) {
	// todo: add your logic here and delete this line

	return
}
