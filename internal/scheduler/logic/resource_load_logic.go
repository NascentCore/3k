package logic

import (
	"context"

	"sxwl/3k/internal/scheduler/model"
	"sxwl/3k/internal/scheduler/svc"
	"sxwl/3k/internal/scheduler/types"

	"github.com/zeromicro/go-zero/core/logx"
)

type ResourceLoadLogic struct {
	logx.Logger
	ctx    context.Context
	svcCtx *svc.ServiceContext
}

func NewResourceLoadLogic(ctx context.Context, svcCtx *svc.ServiceContext) *ResourceLoadLogic {
	return &ResourceLoadLogic{
		Logger: logx.WithContext(ctx),
		ctx:    ctx,
		svcCtx: svcCtx,
	}
}

func (l *ResourceLoadLogic) ResourceLoad(req *types.ResourceLoadReq) (resp *types.BaseResp, err error) {
	UserModel := l.svcCtx.UserModel
	TaskModel := l.svcCtx.ResourceSyncTaskModel

	// 1. 检查管理员权限
	isAdmin, err := UserModel.IsAdmin(l.ctx, req.UserID)
	if err != nil {
		l.Errorf("UserModel.IsAdmin userID=%s err=%s", req.UserID, err)
		return nil, ErrNotAdmin
	}
	if !isAdmin {
		l.Infof("UserModel.IsAdmin userID=%s is not admin", req.UserID)
		return nil, ErrNotAdmin
	}

	// 2. 检查是否存在相同资源的同步任务
	builder := TaskModel.AllFieldsBuilder().Where("resource_id = ?", req.ResourceID)
	existTask, err := TaskModel.FindOneByQuery(l.ctx, builder)
	if err != nil && err != model.ErrNotFound {
		l.Errorf("ResourceSyncTaskModel.FindOneByResourceID err=%s", err)
		return nil, ErrDB
	}
	if existTask != nil {
		l.Infof("resource sync task already exists, resource_id=%s", req.ResourceID)
		return nil, ErrResourceSyncTaskExists
	}

	// 3. 创建同步任务记录
	_, err = l.svcCtx.ResourceSyncTaskModel.Insert(l.ctx, &model.ResourceSyncTask{
		ResourceId:   req.ResourceID,
		ResourceType: req.ResourceType,
		Source:       req.Source,
		Status:       model.ResourceSyncTaskStatusPending,
		CreatorId:    req.UserID,
	})
	if err != nil {
		l.Errorf("ResourceSyncTaskModel.Insert err=%s", err)
		return nil, ErrDB
	}

	return &types.BaseResp{
		Message: MsgResourceAddOK,
	}, nil
}
