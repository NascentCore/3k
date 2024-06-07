package logic

import (
	"context"
	"time"

	"sxwl/3k/internal/scheduler/svc"
	"sxwl/3k/internal/scheduler/types"

	"github.com/Masterminds/squirrel"
	"github.com/jinzhu/copier"
	"github.com/zeromicro/go-zero/core/logx"
)

type RechargeListLogic struct {
	logx.Logger
	ctx    context.Context
	svcCtx *svc.ServiceContext
}

func NewRechargeListLogic(ctx context.Context, svcCtx *svc.ServiceContext) *RechargeListLogic {
	return &RechargeListLogic{
		Logger: logx.WithContext(ctx),
		ctx:    ctx,
		svcCtx: svcCtx,
	}
}

func (l *RechargeListLogic) RechargeList(req *types.RechargeListReq) (resp *types.RechargeListResp, err error) {
	UserModel := l.svcCtx.UserModel
	RechargeModel := l.svcCtx.RechargeModel

	// auth check
	isAdmin, err := UserModel.IsAdmin(l.ctx, req.UserID)
	if err != nil {
		l.Errorf("UserModel.IsAdmin userID=%s err=%s", req.UserID, err)
		return nil, ErrNotAdmin
	}
	if !isAdmin && req.UserID != req.ToUser {
		l.Infof("UserModel.IsAdmin userID=%s is not admin", req.UserID)
		return nil, ErrNotAdmin
	}

	var userID string
	if req.ToUser != "" {
		userID = req.ToUser
	} else {
		userID = req.UserID
	}

	// Paginate results
	records, total, err := RechargeModel.FindPageListByPage(l.ctx, squirrel.Eq{
		"user_id": userID,
	}, req.Page, req.PageSize, "")
	if err != nil {
		l.Errorf("RechargeModel.FindPageListByPage userID=%s err=%s", userID, err)
		return nil, err
	}

	// Prepare response
	resp = &types.RechargeListResp{
		Total: total,
		Data:  make([]types.UserRecharge, len(records)),
	}

	for i, record := range records {
		userRechargeResp := types.UserRecharge{}
		_ = copier.Copy(&userRechargeResp, record)
		userRechargeResp.CreatedAt = record.CreatedAt.Format(time.DateTime)
		userRechargeResp.UpdatedAt = record.UpdatedAt.Format(time.DateTime)
		resp.Data[i] = userRechargeResp
	}

	return resp, nil
}
