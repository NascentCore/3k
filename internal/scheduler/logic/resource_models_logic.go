package logic

import (
	"context"
	"encoding/json"
	"sort"
	"sxwl/3k/internal/scheduler/model"
	"sxwl/3k/pkg/consts"

	"github.com/Masterminds/squirrel"

	"sxwl/3k/internal/scheduler/svc"
	"sxwl/3k/internal/scheduler/types"

	"github.com/zeromicro/go-zero/core/logx"
)

type ResourceModelsLogic struct {
	logx.Logger
	ctx    context.Context
	svcCtx *svc.ServiceContext
}

func NewResourceModelsLogic(ctx context.Context, svcCtx *svc.ServiceContext) *ResourceModelsLogic {
	return &ResourceModelsLogic{
		Logger: logx.WithContext(ctx),
		ctx:    ctx,
		svcCtx: svcCtx,
	}
}

func (l *ResourceModelsLogic) ResourceModels(req *types.ResourceModelsReq) (resp *types.ResourceListResp, err error) {
	OssResourceModel := l.svcCtx.OssResourceModel

	resp = &types.ResourceListResp{
		PublicList: make([]types.Resource, 0),
		UserList:   make([]types.Resource, 0),
	}

	var validPublicModels, otherPublicModels []types.Resource

	models, err := OssResourceModel.Find(l.ctx, OssResourceModel.AllFieldsBuilder().Where(
		squirrel.Eq{"resource_type": consts.Model},
		squirrel.Or{
			squirrel.Eq{"public": model.CachePublic},
			squirrel.Eq{"user_id": req.UserID},
		},
	))
	if err != nil {
		l.Errorf("OssResourceModel find err: %v", err)
		return nil, ErrDBFind
	}

	for _, ossModel := range models {
		meta := model.OssResourceModelMeta{}
		err = json.Unmarshal([]byte(ossModel.Meta), &meta)
		if err != nil {
			l.Errorf("OssResourceModel find err: %v", err)
			return nil, ErrDBFind
		}

		var tag []string
		if meta.CanFinetune {
			tag = append(tag, "finetune")
		}
		if meta.CanInference {
			tag = append(tag, "inference")
		}

		isPublic := false
		if ossModel.Public == model.CachePublic {
			isPublic = true
		}

		m := types.Resource{
			ID:                ossModel.ResourceId,
			Name:              ossModel.ResourceName,
			Object:            ossModel.ResourceType,
			Owner:             ossModel.UserId,
			IsPublic:          isPublic,
			UserID:            ossModel.UserId,
			Size:              ossModel.ResourceSize,
			Tag:               tag,
			Template:          meta.Template,
			Category:          meta.Category,
			FinetuneGPUCount:  1,
			InferenceGPUCount: 1,
		}

		if isPublic {
			_, ok := l.svcCtx.Config.FinetuneModel[ossModel.ResourceName]
			if ok {
				validPublicModels = append(validPublicModels, m)
			} else {
				otherPublicModels = append(otherPublicModels, m)
			}
		} else {
			resp.UserList = append(resp.UserList, m)
		}
	}

	sort.Slice(validPublicModels, func(i, j int) bool {
		return validPublicModels[i].ID < validPublicModels[j].ID
	})

	sort.Slice(otherPublicModels, func(i, j int) bool {
		return otherPublicModels[i].ID < otherPublicModels[j].ID
	})

	resp.PublicList = append(resp.PublicList, validPublicModels...)
	resp.PublicList = append(resp.PublicList, otherPublicModels...)

	return
}
