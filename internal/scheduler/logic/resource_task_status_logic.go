package logic

import (
	"context"

	"sxwl/3k/internal/scheduler/model"
	"sxwl/3k/internal/scheduler/svc"
	"sxwl/3k/internal/scheduler/types"

	"github.com/Masterminds/squirrel"
	"github.com/zeromicro/go-zero/core/logx"
)

type ResourceTaskStatusLogic struct {
	logx.Logger
	ctx    context.Context
	svcCtx *svc.ServiceContext
}

func NewResourceTaskStatusLogic(ctx context.Context, svcCtx *svc.ServiceContext) *ResourceTaskStatusLogic {
	return &ResourceTaskStatusLogic{
		Logger: logx.WithContext(ctx),
		ctx:    ctx,
		svcCtx: svcCtx,
	}
}

func (l *ResourceTaskStatusLogic) ResourceTaskStatus(req *types.ResourceTaskStatusReq) (resp *types.ResourceTaskStatusResp, err error) {
	taskModel := l.svcCtx.ResourceSyncTaskModel
	builder := taskModel.AllFieldsBuilder().Where(squirrel.Eq{"creator_id": req.UserID})
	if req.ResourceID != "" {
		builder = builder.Where(squirrel.Eq{"resource_id": req.ResourceID})
	}
	if req.ResourceType != "" {
		builder = builder.Where(squirrel.Eq{"resource_type": req.ResourceType})
	}

	tasks, err := taskModel.Find(l.ctx, builder)
	if err != nil && err != model.ErrNotFound {
		l.Errorf("find resource sync task error: %v", err)
		return nil, err
	}

	resp = &types.ResourceTaskStatusResp{
		Data:  make([]types.ResourceSyncTask, 0),
		Total: int64(len(tasks)),
	}

	for _, task := range tasks {
		resp.Data = append(resp.Data, types.ResourceSyncTask{
			ResourceID:   task.ResourceId,
			ResourceType: task.ResourceType,
			Source:       task.Source,
			Status:       model.ResourceSyncTaskStatusMap[task.Status],
		})
	}

	return
}
