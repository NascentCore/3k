package logic

import (
	"context"
	"sxwl/3k/internal/scheduler/model"

	"github.com/jinzhu/copier"

	"sxwl/3k/internal/scheduler/svc"
	"sxwl/3k/internal/scheduler/types"

	"github.com/zeromicro/go-zero/core/logx"
)

type ModelsLogic struct {
	logx.Logger
	ctx    context.Context
	svcCtx *svc.ServiceContext
}

func NewModelsLogic(ctx context.Context, svcCtx *svc.ServiceContext) *ModelsLogic {
	return &ModelsLogic{
		Logger: logx.WithContext(ctx),
		ctx:    ctx,
		svcCtx: svcCtx,
	}
}

func (l *ModelsLogic) Models() (resp []types.Cache, err error) {
	CpodCacheModel := l.svcCtx.CpodCacheModel
	caches, err := CpodCacheModel.FindActive(l.ctx, model.CacheModel, 30)
	if err != nil {
		l.Logger.Errorf("cache find active error: %s", err)
		return nil, err
	}

	resp = make([]types.Cache, 0)
	for _, cache := range caches {
		dataType, ok := dbToCacheTypeMap[int(cache.DataType)]
		if !ok {
			l.Logger.Errorf("data_type not match data_type=%d", cache.DataType)
			continue
		}
		respCache := types.Cache{}
		_ = copier.Copy(&respCache, cache)
		respCache.DataType = dataType
		resp = append(resp, respCache)
	}

	return
}
