package logic

import (
	"context"
	"sxwl/3k/internal/scheduler/model"
	"sxwl/3k/internal/scheduler/svc"
	"sxwl/3k/internal/scheduler/types"
	"sxwl/3k/pkg/uuid"

	"github.com/zeromicro/go-zero/core/logx"
	"github.com/zeromicro/go-zero/core/stores/sqlx"
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

func (l *BalanceAddLogic) BalanceAdd(req *types.BalanceAddReq) (*types.BalanceAddResp, error) {
	UserModel := l.svcCtx.UserModel
	BalanceModel := l.svcCtx.UserBalanceModel
	RechargeModel := l.svcCtx.RechargeModel

	// Check if the user is an admin
	isAdmin, err := UserModel.IsAdmin(l.ctx, req.UserID)
	if err != nil {
		l.Errorf("UserModel.IsAdmin userID=%s err=%s", req.UserID, err)
		return nil, ErrNotAdmin
	}
	if !isAdmin {
		l.Infof("UserModel.IsAdmin userID=%s is not admin", req.UserID)
		return nil, ErrNotAdmin
	}

	// Generate recharge ID
	rechargeID, err := uuid.WithPrefix("recharge")
	if err != nil {
		l.Errorf("new uuid userID=%s err=%s", req.UserID, err)
		return nil, ErrSystem
	}

	// Transaction handling
	err = l.svcCtx.DB.TransactCtx(l.ctx, func(ctx context.Context, session sqlx.Session) error {
		// Get current balance
		beforeBalance, err := BalanceModel.GetBalance(ctx, session, req.ToUser)
		if err != nil {
			l.Errorf("BalanceModel.GetBalance userID=%s err=%s", req.UserID, err)
			return err
		}

		// Update balance
		afterBalance := beforeBalance + req.Amount
		err = BalanceModel.SetBalance(ctx, session, req.ToUser, afterBalance)
		if err != nil {
			l.Errorf("BalanceModel.SetBalance userID=%s err=%s", req.UserID, err)
			return err
		}

		// Create recharge record
		recharge := &model.UserRecharge{
			RechargeId:    rechargeID,
			UserId:        req.ToUser,
			Amount:        req.Amount,
			BeforeBalance: beforeBalance,
			AfterBalance:  afterBalance,
			Description:   DescRechargeAdmin,
		}

		_, err = RechargeModel.TransInsert(ctx, session, recharge)
		if err != nil {
			l.Errorf("RechargeModel.TransInsert userID=%s err=%s", req.UserID, err)
			return err
		}

		return nil
	})

	if err != nil {
		l.Errorf("DbTrans err=%s", err)
		return nil, ErrDB
	}

	return &types.BalanceAddResp{
		Message: MsgBalanceAddSuccess,
	}, nil
}
