package logic

import (
	"context"
	"fmt"
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

	dirs, err := storage.ListDir(l.svcCtx.Config.OSS.Bucket, l.svcCtx.Config.OSS.PublicModelDir, 2)
	if err != nil {
		return nil, err
	}

	for dir, size := range dirs {
		resp = append(resp, types.Resource{
			ID:     strings.TrimSuffix(dir, "/"),
			Object: "model",
			Owner:  "public",
			Size:   size,
		})
	}

	dirs, err = storage.ListDir(l.svcCtx.Config.OSS.Bucket,
		fmt.Sprintf(l.svcCtx.Config.OSS.UserModelDir, req.UserID), 1)
	if err != nil {
		return nil, err
	}

	for dir, size := range dirs {
		resp = append(resp, types.Resource{
			ID:     fmt.Sprintf("user-%d/%s", req.UserID, strings.TrimSuffix(dir, "/")),
			Object: "model",
			Owner:  strconv.FormatInt(req.UserID, 10),
			Size:   size,
		})
	}

	return
}