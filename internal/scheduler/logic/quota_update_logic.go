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
	UserModel := l.svcCtx.UserModel
	QuotaModel := l.svcCtx.QuotaModel

	isAdmin, err := UserModel.IsAdmin(l.ctx, req.UserID)
	if err != nil {
		l.Errorf("UserModel.IsAdmin userID=%s err=%s", req.UserID, err)
		return nil, ErrNotAdmin
	}
	if !isAdmin {
		l.Infof("UserModel.IsAdmin userID=%s is not admin", req.UserID)
		return nil, ErrNotAdmin
	}

	_, err = QuotaModel.UpdateColsByCond(l.ctx, QuotaModel.UpdateBuilder().Where(squirrel.Eq{
		"id": req.Id,
	}).Set("quota", req.Quota))
	if err != nil {
		l.Errorf("QuotaUpdate update quota id=%d quota=%d err=%s", req.Id, req.Quota, err)
		return nil, err
	}

	return &types.QuotaUpdateResp{Message: "ok"}, nil
}
