package logic

import (
	"context"
	"fmt"
	"sort"
	"strconv"
	"strings"
	"sxwl/3k/internal/scheduler/model"
	"sxwl/3k/pkg/consts"
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
	CpodCacheModel := l.svcCtx.CpodCacheModel
	resp = []types.Resource{}

	dirs, err := storage.ListDir(l.svcCtx.Config.OSS.Bucket, l.svcCtx.Config.OSS.PublicDatasetDir, 2)
	if err != nil {
		return nil, err
	}

	for dir, size := range dirs {
		datasetName := strings.TrimPrefix(strings.TrimSuffix(dir, "/"), l.svcCtx.Config.OSS.PublicDatasetDir)
		resp = append(resp, types.Resource{
			ID:       storage.DatasetCRDName(storage.ResourceToOSSPath(consts.Dataset, datasetName)),
			Name:     datasetName,
			Object:   "dataset",
			Owner:    "public",
			IsPublic: true,
			Tag:      []string{},
			Size:     size,
		})
	}

	if l.svcCtx.Config.OSS.LocalMode {
		datasets, err := CpodCacheModel.FindActive(l.ctx, model.CacheDataset, req.UserID, 30)
		if err != nil {
			return nil, err
		}

		for _, dataset := range datasets {
			datasetName := strings.TrimPrefix(strings.TrimSuffix(dataset.DataName, "/"), l.svcCtx.Config.OSS.UserDatasetPrefix)
			resp = append(resp, types.Resource{
				ID:       dataset.DataId,
				Name:     datasetName,
				Object:   "dataset",
				Owner:    "user",
				IsPublic: false,
				UserId:   dataset.UserId.Int64,
				Tag:      []string{},
				Size:     dataset.DataSize,
			})
		}
	} else {
		dirs, err = storage.ListDir(l.svcCtx.Config.OSS.Bucket,
			fmt.Sprintf(l.svcCtx.Config.OSS.UserDatasetDir, req.UserID), 1)
		if err != nil {
			return nil, err
		}

		for dir, size := range dirs {
			datasetName := strings.TrimPrefix(strings.TrimSuffix(dir, "/"), l.svcCtx.Config.OSS.UserDatasetPrefix)
			resp = append(resp, types.Resource{
				ID:       storage.DatasetCRDName(storage.ResourceToOSSPath(consts.Dataset, datasetName)),
				Name:     datasetName,
				Object:   "dataset",
				Owner:    strconv.FormatInt(req.UserID, 10),
				IsPublic: false,
				UserId:   req.UserID,
				Tag:      []string{},
				Size:     size,
			})
		}

		sort.Slice(resp, func(i, j int) bool {
			if resp[i].Owner != resp[j].Owner {
				return resp[i].Owner < resp[j].Owner
			}

			return resp[i].ID < resp[j].ID
		})
	}

	return
}
