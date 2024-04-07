package logic

import (
	"context"

	"sxwl/3k/internal/scheduler/svc"
	"sxwl/3k/internal/scheduler/types"

	"github.com/Masterminds/squirrel"
	"github.com/zeromicro/go-zero/core/logx"
)

type QuotaUpdateLogic struct {
	logx.Logger
	ctx    context.Context
	svcCtx *svc.ServiceContext
}

func NewQuotaUpdateLogic(ctx context.Context, svcCtx *svc.ServiceContext) *QuotaUpdateLogic {
	return &QuotaUpdateLogic{
		Logger: logx.WithContext(ctx),
		ctx:    ctx,
		svcCtx: svcCtx,
	}
}

func (l *QuotaUpdateLogic) QuotaUpdate(req *types.QuotaUpdateReq) (resp *types.QuotaUpdateResp, err error) {
	QuotaModel := l.svcCtx.QuotaModel
	_, err = QuotaModel.UpdateColsByCond(l.ctx, QuotaModel.UpdateBuilder().Where(squirrel.Eq{
		"id": req.Id,
	}).Set("quota", req.Quota))
	if err != nil {
		l.Errorf("QuotaUpdate update quota id=%d quota=%d err=%s", req.Id, req.Quota, err)
		return nil, err
	}

	return &types.QuotaUpdateResp{Message: "ok"}, nil
}
