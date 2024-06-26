package logic

import (
	"context"
	"fmt"
	"path"
	"sort"
	"strings"
	"sxwl/3k/internal/scheduler/model"
	"sxwl/3k/pkg/consts"
	"sxwl/3k/pkg/storage"

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
	CpodCacheModel := l.svcCtx.CpodCacheModel
	resp = &types.ResourceListResp{
		PublicList: make([]types.Resource, 0),
		UserList:   make([]types.Resource, 0),
	}

	// public models
	// ListDir 只能查询1000个匹配前缀的文件，小批量数据ok，更完善还是需要有db来存储模型元数据
	dirs, err := storage.ListDir(l.svcCtx.Config.OSS.Bucket, l.svcCtx.Config.OSS.PublicModelDir, 2)
	if err != nil {
		return nil, err
	}

	var validModels, otherModels []types.Resource

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

		modelName := strings.TrimPrefix(strings.TrimSuffix(dir, "/"), l.svcCtx.Config.OSS.PublicModelDir)
		m := types.Resource{
			ID:                storage.ModelCRDName(storage.ResourceToOSSPath(consts.Model, modelName)),
			Name:              modelName,
			Object:            "model",
			Owner:             "public",
			IsPublic:          true,
			Size:              size,
			Tag:               tag,
			Template:          storage.ModelTemplate(l.svcCtx.Config.OSS.Bucket, modelName),
			FinetuneGPUCount:  1,
			InferenceGPUCount: 1,
		}
		_, ok := l.svcCtx.Config.FinetuneModel[modelName]
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
	if l.svcCtx.Config.OSS.LocalMode {
		models, err := CpodCacheModel.FindActive(l.ctx, model.CacheModel, req.UserID, 30)
		if err != nil {
			return nil, err
		}

		for _, m := range models {
			tag := []string{"finetune", "inference"} // TODO 根据其他元信息来判断是否能微调和推理
			modelName := strings.TrimPrefix(strings.TrimSuffix(m.DataName, "/"), l.svcCtx.Config.OSS.UserModelPrefix)
			resp.UserList = append(resp.UserList, types.Resource{
				ID:                m.DataId,
				Name:              modelName,
				Object:            "model",
				Owner:             m.NewUserId.String,
				Size:              m.DataSize,
				IsPublic:          false,
				UserID:            m.NewUserId.String,
				Tag:               tag,
				Template:          m.Template,
				FinetuneGPUCount:  int(m.FinetuneGpuCount),
				InferenceGPUCount: int(m.InferenceGpuCount),
			})
		}
	} else {
		dirs, err = storage.ListDir(l.svcCtx.Config.OSS.Bucket,
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
	}

	return
}
