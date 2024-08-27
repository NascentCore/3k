package logic

import (
	"context"
	"sxwl/3k/internal/scheduler/model"
	"sxwl/3k/internal/scheduler/svc"
	"sxwl/3k/internal/scheduler/types"

	"github.com/jinzhu/copier"

	"github.com/zeromicro/go-zero/core/logx"
)

type ResourceMetaAddLogic struct {
	logx.Logger
	ctx    context.Context
	svcCtx *svc.ServiceContext
}

func NewResourceMetaAddLogic(ctx context.Context, svcCtx *svc.ServiceContext) *ResourceMetaAddLogic {
	return &ResourceMetaAddLogic{
		Logger: logx.WithContext(ctx),
		ctx:    ctx,
		svcCtx: svcCtx,
	}
}

func (l *ResourceMetaAddLogic) ResourceMetaAdd(req *types.ResourceMetaAddReq) (resp *types.BaseResp, err error) {
	OssResourceModel := l.svcCtx.OssResourceModel

	public := model.CachePrivate
	if req.IsPublic {
		public = model.CachePublic
	}

	resource := model.SysOssResource{}
	_ = copier.Copy(&resource, req)
	resource.Public = int64(public)

	_, err = OssResourceModel.Insert(l.ctx, &resource)
	if err != nil {
		l.Errorf("insert err=%v", err)
		return nil, ErrDB
	}

	return
}
