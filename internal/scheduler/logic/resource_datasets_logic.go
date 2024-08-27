package logic

import (
	"context"
	"sxwl/3k/internal/scheduler/model"
	"sxwl/3k/internal/scheduler/svc"
	"sxwl/3k/internal/scheduler/types"
	"sxwl/3k/pkg/consts"

	"github.com/Masterminds/squirrel"
	"github.com/zeromicro/go-zero/core/logx"
)

type ResourceDatasetsLogic struct {
	logx.Logger
	ctx    context.Context
	svcCtx *svc.ServiceContext
}

func NewResourceDatasetsLogic(ctx context.Context, svcCtx *svc.ServiceContext) *ResourceDatasetsLogic {
	return &ResourceDatasetsLogic{
		Logger: logx.WithContext(ctx),
		ctx:    ctx,
		svcCtx: svcCtx,
	}
}

func (l *ResourceDatasetsLogic) ResourceDatasets(req *types.ResourceDatasetsReq) (resp *types.ResourceListResp, err error) {
	OssResourceModel := l.svcCtx.OssResourceModel

	resp = &types.ResourceListResp{
		PublicList: make([]types.Resource, 0),
		UserList:   make([]types.Resource, 0),
	}

	datasets, err := OssResourceModel.Find(l.ctx, OssResourceModel.AllFieldsBuilder().Where(
		squirrel.Eq{"resource_type": consts.Dataset},
		squirrel.Or{
			squirrel.Eq{"public": model.CachePublic},
			squirrel.Eq{"user_id": req.UserID},
		},
	))
	if err != nil {
		l.Errorf("OssResourceModel find err: %v", err)
		return nil, ErrDBFind
	}

	for _, dataset := range datasets {
		isPublic := false
		if dataset.Public == model.CachePublic {
			isPublic = true
		}

		r := types.Resource{
			ID:       dataset.ResourceId,
			Name:     dataset.ResourceName,
			Object:   dataset.ResourceType,
			Owner:    dataset.UserId,
			IsPublic: isPublic,
			Size:     dataset.ResourceSize,
		}

		if isPublic {
			resp.PublicList = append(resp.PublicList, r)
		} else {
			resp.UserList = append(resp.UserList, r)
		}
	}

	return
}
