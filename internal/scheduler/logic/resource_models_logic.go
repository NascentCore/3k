package logic

import (
	"context"
	"fmt"
	"path"
	"strconv"
	"strings"
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

func (l *ResourceModelsLogic) ResourceModels(req *types.ResourceModelsReq) (resp []types.Resource, err error) {
	resp = []types.Resource{}

	// ListDir 只能查询1000个匹配前缀的文件，小批量数据ok，更完善还是需要有db来存储模型元数据
	dirs, err := storage.ListDir(l.svcCtx.Config.OSS.Bucket, l.svcCtx.Config.OSS.PublicModelDir, 2)
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

		tag := []string{}
		if canFinetune {
			tag = append(tag, "finetune")
		}
		if canInference {
			tag = append(tag, "inference")
		}

		resp = append(resp, types.Resource{
			ID:     strings.TrimPrefix(strings.TrimSuffix(dir, "/"), l.svcCtx.Config.OSS.PublicModelDir),
			Object: "model",
			Owner:  "public",
			Size:   size,
			Tag:    tag,
		})
	}

	dirs, err = storage.ListDir(l.svcCtx.Config.OSS.Bucket,
		fmt.Sprintf(l.svcCtx.Config.OSS.UserModelDir, req.UserID), 1)
	if err != nil {
		return nil, err
	}

	for dir, size := range dirs {
		resp = append(resp, types.Resource{
			ID:     strings.TrimPrefix(strings.TrimSuffix(dir, "/"), l.svcCtx.Config.OSS.UserModelPrefix),
			Object: "model",
			Owner:  strconv.FormatInt(req.UserID, 10),
			Size:   size,
			Tag:    []string{},
		})
	}

	return
}
