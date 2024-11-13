package logic

import (
	"context"

	"sxwl/3k/internal/scheduler/model"
	"sxwl/3k/internal/scheduler/svc"
	"sxwl/3k/internal/scheduler/types"

	"github.com/zeromicro/go-zero/core/logx"
)

type ResourceTaskUpdateLogic struct {
	logx.Logger
	ctx    context.Context
	svcCtx *svc.ServiceContext
}

func NewResourceTaskUpdateLogic(ctx context.Context, svcCtx *svc.ServiceContext) *ResourceTaskUpdateLogic {
	return &ResourceTaskUpdateLogic{
		Logger: logx.WithContext(ctx),
		ctx:    ctx,
		svcCtx: svcCtx,
	}
}

func (l *ResourceTaskUpdateLogic) ResourceTaskUpdate(req *types.ResourceTaskUpdateReq) (resp *types.BaseResp, err error) {
	UserModel := l.svcCtx.UserModel

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

	// 2. 更新资源同步任务状态
	TaskModel := l.svcCtx.ResourceSyncTaskModel
	for _, task := range req.Data {
		selectBuilder := TaskModel.AllFieldsBuilder().Where("resource_id = ?", task.ResourceID)
		existTask, err := TaskModel.FindOneByQuery(l.ctx, selectBuilder)
		if err != nil {
			l.Errorf("ResourceSyncTaskModel.FindOneByResourceID err=%s", err)
			return nil, ErrDB
		}
		if existTask == nil {
			l.Errorf("resource sync task not found, resource_id=%s", task.ResourceID)
			continue
		}

		// 检查一下excutor_id与req.UserID是否一致
		if existTask.ExecutorId != req.UserID {
			l.Errorf("resource sync task executor_id not match, resource_id=%s, executor_id=%s, user_id=%s", task.ResourceID, existTask.ExecutorId, req.UserID)
			continue
		}

		// 检查resource_type是否一致
		if existTask.ResourceType != task.ResourceType {
			l.Errorf("resource sync task resource_type not match, resource_id=%s, resource_type=%s, task_resource_type=%s", task.ResourceID, existTask.ResourceType, task.ResourceType)
			continue
		}

		updateBuilder := TaskModel.UpdateBuilder().Where("id = ?", existTask.Id)
		if task.OK {
			updateBuilder = updateBuilder.Set("status", model.ResourceSyncTaskStatusUploaded)
			updateBuilder = updateBuilder.Set("size", task.Size)
		} else {
			updateBuilder = updateBuilder.Set("status", model.ResourceSyncTaskStatusFailed)
			updateBuilder = updateBuilder.Set("err_info", task.Err)
		}
		_, err = TaskModel.UpdateColsByCond(l.ctx, updateBuilder)
		if err != nil {
			l.Errorf("ResourceSyncTaskModel.UpdateByQuery err=%s", err)
			return nil, ErrDB
		}
	}

	return
}
