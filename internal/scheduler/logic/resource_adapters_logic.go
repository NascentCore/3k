package logic

import (
	"context"
	"encoding/json"
	"sxwl/3k/internal/scheduler/model"
	"sxwl/3k/internal/scheduler/svc"
	"sxwl/3k/internal/scheduler/types"
	"sxwl/3k/pkg/consts"

	"github.com/Masterminds/squirrel"
	"github.com/zeromicro/go-zero/core/logx"
)

type ResourceAdaptersLogic struct {
	logx.Logger
	ctx    context.Context
	svcCtx *svc.ServiceContext
}

func NewResourceAdaptersLogic(ctx context.Context, svcCtx *svc.ServiceContext) *ResourceAdaptersLogic {
	return &ResourceAdaptersLogic{
		Logger: logx.WithContext(ctx),
		ctx:    ctx,
		svcCtx: svcCtx,
	}
}

func (l *ResourceAdaptersLogic) ResourceAdapters(req *types.ResourceAdaptersReq) (resp *types.ResourceListResp, err error) {
	OssResourceModel := l.svcCtx.OssResourceModel

	resp = &types.ResourceListResp{
		PublicList: make([]types.Resource, 0),
		UserList:   make([]types.Resource, 0),
	}

	adapters, err := OssResourceModel.Find(l.ctx, OssResourceModel.AllFieldsBuilder().Where(
		squirrel.Eq{"resource_type": consts.Adapter},
		squirrel.Or{
			squirrel.Eq{"public": model.CachePublic},
			squirrel.Eq{"user_id": req.UserID},
		},
	))
	if err != nil {
		l.Errorf("OssResourceModel find err: %v", err)
		return nil, ErrDBFind
	}

	for _, adapter := range adapters {
		meta := model.OssResourceAdapterMeta{}
		err = json.Unmarshal([]byte(adapter.Meta), &meta)
		if err != nil {
			l.Errorf("OssResourceModel find err: %v", err)
			return nil, ErrDBFind
		}

		isPublic := false
		if adapter.Public == model.CachePublic {
			isPublic = true
		}

		r := types.Resource{
			ID:                adapter.ResourceId,
			Name:              adapter.ResourceName,
			Object:            adapter.ResourceType,
			Owner:             adapter.UserId,
			IsPublic:          isPublic,
			Size:              adapter.ResourceSize,
			BaseModel:         meta.BaseModel,
			FinetuneGPUCount:  1,
			InferenceGPUCount: 1,
		}

		if isPublic {
			resp.PublicList = append(resp.PublicList, r)
		} else {
			resp.UserList = append(resp.UserList, r)
		}
	}

	return
}
