package logic

import (
	"context"

	"sxwl/3k/internal/scheduler/svc"
	"sxwl/3k/internal/scheduler/types"

	"github.com/Masterminds/squirrel"
	"github.com/zeromicro/go-zero/core/logx"
)

type BalanceGetLogic struct {
	logx.Logger
	ctx    context.Context
	svcCtx *svc.ServiceContext
}

func NewBalanceGetLogic(ctx context.Context, svcCtx *svc.ServiceContext) *BalanceGetLogic {
	return &BalanceGetLogic{
		Logger: logx.WithContext(ctx),
		ctx:    ctx,
		svcCtx: svcCtx,
	}
}

func (l *BalanceGetLogic) BalanceGet(req *types.BalanceGetReq) (resp *types.BalanceGetResp, err error) {
	UserModel := l.svcCtx.UserModel
	BalanceModel := l.svcCtx.UserBalanceModel

	isAdmin, err := UserModel.IsAdmin(l.ctx, req.UserID)
	if err != nil {
		l.Errorf("UserModel.IsAdmin userID=%s err=%s", req.UserID, err)
		return nil, ErrNotAdmin
	}
	if !isAdmin && req.UserID != req.ToUser {
		l.Infof("UserModel.IsAdmin userID=%s is not admin", req.UserID)
		return nil, ErrNotAdmin
	}

	balance, err := BalanceModel.FindOneByQuery(l.ctx, BalanceModel.AllFieldsBuilder().Where(squirrel.Eq{
		"new_user_id": req.ToUser,
	}))
	if err != nil {
		l.Errorf("BalanceModel.FindOneByQuery userID=%s err=%s", req.ToUser, err)
		return nil, ErrBalanceFindFail
	}

	return &types.BalanceGetResp{Balance: balance.Balance, UserID: req.ToUser}, nil
}
