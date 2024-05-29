package logic

import (
	"context"

	"sxwl/3k/internal/scheduler/svc"
	"sxwl/3k/internal/scheduler/types"

	"github.com/Masterminds/squirrel"
	"github.com/zeromicro/go-zero/core/logx"
)

type BalanceAddLogic struct {
	logx.Logger
	ctx    context.Context
	svcCtx *svc.ServiceContext
}

func NewBalanceAddLogic(ctx context.Context, svcCtx *svc.ServiceContext) *BalanceAddLogic {
	return &BalanceAddLogic{
		Logger: logx.WithContext(ctx),
		ctx:    ctx,
		svcCtx: svcCtx,
	}
}

func (l *BalanceAddLogic) BalanceAdd(req *types.BalanceAddReq) (resp *types.BalanceAddResp, err error) {
	UserModel := l.svcCtx.UserModel
	BalanceModel := l.svcCtx.UserBalanceModel

	isAdmin, err := UserModel.IsAdmin(l.ctx, req.UserID)
	if err != nil {
		l.Errorf("UserModel.IsAdmin userID=%s err=%s", req.UserID, err)
		return nil, ErrNotAdmin
	}
	if !isAdmin {
		l.Infof("UserModel.IsAdmin userID=%s is not admin", req.UserID)
		return nil, ErrNotAdmin
	}

	result, err := BalanceModel.UpdateColsByCond(l.ctx, BalanceModel.UpdateBuilder().Where(squirrel.Eq{
		"new_user_id": req.ToUser,
	}).Set("balance", squirrel.Expr("balance + ?", req.Amount)))
	if err != nil {
		l.Errorf("update balance userID=%s amount=%.2f err=%s", req.UserID, req.Amount, err)
		return nil, ErrBalanceAddFail
	}

	rows, err := result.RowsAffected()
	if err != nil {
		l.Errorf("update balance RowsAffected user_id=%s err=%s", req.UserID, err)
		return nil, ErrBalanceAddFail
	}

	if rows != 1 {
		l.Errorf("update balance RowsAffected rows=%d user_id=%s err=%s", rows, req.UserID, err)
		return nil, ErrBalanceAddFail
	}

	return &types.BalanceAddResp{Message: MsgBalanceAddSuccess}, nil
}
