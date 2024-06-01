package logic

import (
	"context"
	"fmt"
	"sort"
	"strings"
	"sxwl/3k/internal/scheduler/model"
	"sxwl/3k/pkg/consts"
	"sxwl/3k/pkg/storage"

	"sxwl/3k/internal/scheduler/svc"
	"sxwl/3k/internal/scheduler/types"

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
	CpodCacheModel := l.svcCtx.CpodCacheModel
	resp = &types.ResourceListResp{
		PublicList: make([]types.Resource, 0),
		UserList:   make([]types.Resource, 0),
	}

	dirs, err := storage.ListDir(l.svcCtx.Config.OSS.Bucket, l.svcCtx.Config.OSS.PublicAdapterDir, 2)
	if err != nil {
		return nil, err
	}

	for dir, size := range dirs {
		adapterName := strings.TrimPrefix(strings.TrimSuffix(dir, "/"), l.svcCtx.Config.OSS.PublicAdapterDir)
		resp.PublicList = append(resp.PublicList, types.Resource{
			ID:       storage.AdapterCRDName(storage.ResourceToOSSPath(consts.Adapter, adapterName)),
			Name:     adapterName,
			Object:   consts.Adapter,
			Owner:    "public",
			IsPublic: true,
			Tag:      []string{},
			Size:     size,
		})
	}

	if l.svcCtx.Config.OSS.LocalMode {
		adapters, err := CpodCacheModel.FindActive(l.ctx, model.CacheAdapter, req.UserID, 30)
		if err != nil {
			return nil, err
		}

		for _, adapter := range adapters {
			adapterName := strings.TrimPrefix(strings.TrimSuffix(adapter.DataName, "/"), l.svcCtx.Config.OSS.UserAdapterPrefix)
			resp.PublicList = append(resp.PublicList, types.Resource{
				ID:       adapter.DataId,
				Name:     adapterName,
				Object:   consts.Adapter,
				Owner:    adapter.NewUserId.String,
				IsPublic: false,
				UserID:   adapter.NewUserId.String,
				Tag:      []string{},
				Size:     adapter.DataSize,
			})
		}
	} else {
		dirs, err = storage.ListDir(l.svcCtx.Config.OSS.Bucket,
			fmt.Sprintf(l.svcCtx.Config.OSS.UserAdapterDir, req.UserID), 1)
		if err != nil {
			return nil, err
		}

		for dir, size := range dirs {
			adapterName := strings.TrimPrefix(strings.TrimSuffix(dir, "/"), l.svcCtx.Config.OSS.UserAdapterPrefix)
			resp.UserList = append(resp.UserList, types.Resource{
				ID:       storage.AdapterCRDName(storage.ResourceToOSSPath(consts.Adapter, adapterName)),
				Name:     adapterName,
				Object:   consts.Adapter,
				Owner:    req.UserID,
				IsPublic: false,
				UserID:   req.UserID,
				Tag:      []string{},
				Size:     size,
			})
		}

		sort.Slice(resp.PublicList, func(i, j int) bool {
			if resp.PublicList[i].Owner != resp.PublicList[j].Owner {
				return resp.PublicList[i].Owner < resp.PublicList[j].Owner
			}

			return resp.PublicList[i].ID < resp.PublicList[j].ID
		})

		sort.Slice(resp.UserList, func(i, j int) bool {
			if resp.UserList[i].Owner != resp.UserList[j].Owner {
				return resp.UserList[i].Owner < resp.UserList[j].Owner
			}

			return resp.UserList[i].ID < resp.UserList[j].ID
		})
	}

	return
}
