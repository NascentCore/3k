package logic

import (
	"context"
	"errors"
	"fmt"
	"sxwl/3k/internal/scheduler/model"

	"sxwl/3k/internal/scheduler/svc"
	"sxwl/3k/internal/scheduler/types"

	"github.com/Masterminds/squirrel"
	"github.com/zeromicro/go-zero/core/logx"
)

type QuotaAddLogic struct {
	logx.Logger
	ctx    context.Context
	svcCtx *svc.ServiceContext
}

func NewQuotaAddLogic(ctx context.Context, svcCtx *svc.ServiceContext) *QuotaAddLogic {
	return &QuotaAddLogic{
		Logger: logx.WithContext(ctx),
		ctx:    ctx,
		svcCtx: svcCtx,
	}
}

func (l *QuotaAddLogic) QuotaAdd(req *types.QuotaAddReq) (resp *types.QuotaAddResp, err error) {
	QuotaModel := l.svcCtx.QuotaModel
	// TODO 增加配额的检查
	// 是否超过配额上限
	// 是否是管理员
	_, err = QuotaModel.FindOneByQuery(l.ctx, QuotaModel.AllFieldsBuilder().Where(squirrel.Eq{
		"user_id":  req.UserId,
		"resource": req.Resource,
	}))
	if !errors.Is(err, model.ErrNotFound) {
		err = fmt.Errorf("QuotaAdd quota already exists user_id=%d resource=%s", req.UserID, req.Resource)
		l.Error(err)
		return nil, err
	}

	_, err = QuotaModel.Insert(l.ctx, &model.SysQuota{
		UserId:   req.UserId,
		Resource: req.Resource,
		Quota:    req.Quota.Quota,
	})
	if err != nil {
		l.Errorf("QuotaAdd user_id=%d resource=%s quota=%d", req.UserID, req.Resource, req.Quota.Quota)
		return nil, err
	}

	resp = &types.QuotaAddResp{Message: "ok"}
	return
}
