package logic

import (
	"context"

	"sxwl/3k/internal/scheduler/svc"
	"sxwl/3k/internal/scheduler/types"

	"github.com/zeromicro/go-zero/core/logx"
)

type ResourceMetaDelLogic struct {
	logx.Logger
	ctx    context.Context
	svcCtx *svc.ServiceContext
}

func NewResourceMetaDelLogic(ctx context.Context, svcCtx *svc.ServiceContext) *ResourceMetaDelLogic {
	return &ResourceMetaDelLogic{
		Logger: logx.WithContext(ctx),
		ctx:    ctx,
		svcCtx: svcCtx,
	}
}

func (l *ResourceMetaDelLogic) ResourceMetaDel(req *types.ResourceMetaDelReq) (resp *types.BaseResp, err error) {
	OssResourceModel := l.svcCtx.OssResourceModel

	err = OssResourceModel.DeleteByResourceID(l.ctx, req.ResourceID)
	if err != nil {
		l.Errorf("delete err: %v", err)
		return nil, ErrDB
	}

	return &types.BaseResp{Message: MsgResourceDeleteOK}, nil
}
