package logic

import (
	"context"
	"encoding/json"

	"sxwl/3k/internal/scheduler/model"
	"sxwl/3k/internal/scheduler/svc"
	"sxwl/3k/internal/scheduler/types"

	"github.com/Masterminds/squirrel"
	"github.com/zeromicro/go-zero/core/logx"
)

type ModelByNameLogic struct {
	logx.Logger
	ctx    context.Context
	svcCtx *svc.ServiceContext
}

func NewModelByNameLogic(ctx context.Context, svcCtx *svc.ServiceContext) *ModelByNameLogic {
	return &ModelByNameLogic{
		Logger: logx.WithContext(ctx),
		ctx:    ctx,
		svcCtx: svcCtx,
	}
}

func (l *ModelByNameLogic) ModelByName(req *types.ModelByNameReq) (resp *types.Model, err error) {
	ResourceModel := l.svcCtx.OssResourceModel
	builder := ResourceModel.AllFieldsBuilder().Where(squirrel.Eq{
		"resource_name": req.ModelName,
		"resource_type": "model",
	})
	m, err := ResourceModel.FindOneByQuery(l.ctx, builder)
	if err != nil && err == model.ErrNotFound {
		l.Errorf("find model by name %s error: %v", req.ModelName, err)
		return nil, ErrModelNotFound
	}
	if err != nil {
		l.Errorf("find model by name %s error: %v", req.ModelName, err)
		return nil, ErrDBFind
	}

	meta := &model.OssResourceModelMeta{}
	if err := json.Unmarshal([]byte(m.Meta), meta); err != nil {
		l.Errorf("unmarshal model %s meta error: %v", req.ModelName, err)
		return nil, ErrSystem
	}

	return &types.Model{
		ModelId:       m.ResourceId,
		ModelName:     m.ResourceName,
		ModelSize:     m.ResourceSize,
		ModelTemplate: meta.Template,
		ModelMeta:     m.Meta,
		ModelCategory: meta.Category,
	}, nil
}
