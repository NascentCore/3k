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

func (l *ResourceDatasetsLogic) ResourceDatasets(req *types.ResourceDatasetsReq) (resp []types.Resource, err error) {
	resp = []types.Resource{}

	dirs, err := storage.ListDir(l.svcCtx.Config.OSS.Bucket, l.svcCtx.Config.OSS.PublicDatasetDir, 2)
	if err != nil {
		return nil, err
	}

	for dir, size := range dirs {
		resp = append(resp, types.Resource{
			ID:     strings.TrimPrefix(strings.TrimSuffix(dir, "/"), l.svcCtx.Config.OSS.PublicDatasetDir),
			Object: "dataset",
			Owner:  "public",
			Size:   size,
		})
	}

	dirs, err = storage.ListDir(l.svcCtx.Config.OSS.Bucket,
		fmt.Sprintf(l.svcCtx.Config.OSS.UserDatasetDir, req.UserID), 1)
	if err != nil {
		return nil, err
	}

	for dir, size := range dirs {
		resp = append(resp, types.Resource{
			ID:     strings.TrimPrefix(strings.TrimSuffix(dir, "/"), l.svcCtx.Config.OSS.UserDatasetPrefix),
			Object: "dataset",
			Owner:  strconv.FormatInt(req.UserID, 10),
			Size:   size,
		})
	}

	return
}
