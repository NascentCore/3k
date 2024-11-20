package logic

import (
	"context"

	"sxwl/3k/internal/scheduler/model"
	"sxwl/3k/internal/scheduler/svc"
	"sxwl/3k/internal/scheduler/types"

	"github.com/Masterminds/squirrel"
	"github.com/zeromicro/go-zero/core/logx"
)

type DatasetByNameLogic struct {
	logx.Logger
	ctx    context.Context
	svcCtx *svc.ServiceContext
}

func NewDatasetByNameLogic(ctx context.Context, svcCtx *svc.ServiceContext) *DatasetByNameLogic {
	return &DatasetByNameLogic{
		Logger: logx.WithContext(ctx),
		ctx:    ctx,
		svcCtx: svcCtx,
	}
}

func (l *DatasetByNameLogic) DatasetByName(req *types.DatasetByNameReq) (resp *types.Dataset, err error) {
	ResourceModel := l.svcCtx.OssResourceModel
	builder := ResourceModel.AllFieldsBuilder().Where(squirrel.Eq{
		"resource_name": req.DatasetName,
		"resource_type": "dataset",
	})

	d, err := ResourceModel.FindOneByQuery(l.ctx, builder)
	if err != nil && err == model.ErrNotFound {
		l.Errorf("find dataset by name %s error: %v", req.DatasetName, err)
		return nil, ErrDatasetNotFound
	}
	if err != nil {
		l.Errorf("find dataset by name %s error: %v", req.DatasetName, err)
		return nil, ErrDBFind
	}

	return &types.Dataset{
		DatasetId:       d.ResourceId,
		DatasetName:     d.ResourceName,
		DatasetSize:     d.ResourceSize,
		DatasetIsPublic: d.Public == 1,
	}, nil
}
