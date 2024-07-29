package logic

import (
	"context"
	"encoding/json"
	"fmt"
	"path"
	"sort"
	"strings"
	"sxwl/3k/internal/scheduler/model"
	"sxwl/3k/pkg/consts"
	"sxwl/3k/pkg/storage"
	"time"

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

	var validModels, otherModels []types.Resource

	// public models
	beginTime := time.Now().Add(time.Duration(-(l.svcCtx.Config.OSS.SyncInterval * 2)) * time.Minute)
	publicModels, err := OssResourceModel.Find(l.ctx, OssResourceModel.AllFieldsBuilder().Where(
		squirrel.GtOrEq{"updated_at": beginTime},
		squirrel.Eq{"resource_type": "model"},
		squirrel.Eq{"public": model.CachePublic},
	))
	if err != nil {
		l.Errorf("OssResourceModel find err: %v", err)
		return nil, ErrDBFind
	}

	for _, publicModel := range publicModels {
		meta := model.OssResourceMeta{}
		err = json.Unmarshal([]byte(publicModel.Meta), &meta)
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

		m := types.Resource{
			ID:                publicModel.ResourceId,
			Name:              publicModel.ResourceName,
			Object:            publicModel.ResourceType,
			Owner:             publicModel.UserId,
			IsPublic:          true,
			Size:              publicModel.ResourceSize,
			Tag:               tag,
			Template:          meta.Template,
			FinetuneGPUCount:  1,
			InferenceGPUCount: 1,
		}
		_, ok := l.svcCtx.Config.FinetuneModel[publicModel.ResourceName]
		if ok {
			validModels = append(validModels, m)
		} else {
			otherModels = append(otherModels, m)
		}
	}

	sort.Slice(validModels, func(i, j int) bool {
		return validModels[i].ID < validModels[j].ID
	})

	sort.Slice(otherModels, func(i, j int) bool {
		return otherModels[i].ID < otherModels[j].ID
	})

	resp.PublicList = append(resp.PublicList, validModels...)
	resp.PublicList = append(resp.PublicList, otherModels...)

	// user models
	dirs, err := storage.ListDir(l.svcCtx.Config.OSS.Bucket,
		fmt.Sprintf(l.svcCtx.Config.OSS.UserModelDir, req.UserID), 1)
	if err != nil {
		return nil, err
	}

	for dir, size := range dirs {
		canFinetune, _, err := storage.ExistFile(l.svcCtx.Config.OSS.Bucket,
			path.Join(dir, l.svcCtx.Config.OSS.FinetuneTagFile))
		if err != nil {
			return nil, err
		}
		canInference, _, err := storage.ExistFile(l.svcCtx.Config.OSS.Bucket,
			path.Join(dir, l.svcCtx.Config.OSS.InferenceTagFile))
		if err != nil {
			return nil, err
		}

		var tag []string
		if canFinetune {
			tag = append(tag, "finetune")
		}
		if canInference {
			tag = append(tag, "inference")
		}

		modelName := strings.TrimPrefix(strings.TrimSuffix(dir, "/"), l.svcCtx.Config.OSS.UserModelPrefix)
		resp.UserList = append(resp.UserList, types.Resource{
			ID:                storage.ModelCRDName(storage.ResourceToOSSPath(consts.Model, modelName)),
			Name:              modelName,
			Object:            "model",
			Owner:             req.UserID,
			Size:              size,
			IsPublic:          false,
			UserID:            req.UserID,
			Tag:               tag,
			Template:          storage.ModelTemplate(l.svcCtx.Config.OSS.Bucket, modelName),
			FinetuneGPUCount:  1,
			InferenceGPUCount: 1,
		})
	}

	return
}
