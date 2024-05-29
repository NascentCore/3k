package logic

import (
	"context"
	"errors"
	"sxwl/3k/internal/scheduler/model"
	"time"

	"sxwl/3k/internal/scheduler/svc"
	"sxwl/3k/internal/scheduler/types"

	"github.com/Masterminds/squirrel"
	"github.com/jinzhu/copier"
	"github.com/zeromicro/go-zero/core/logx"
)

type BillingListLogic struct {
	logx.Logger
	ctx    context.Context
	svcCtx *svc.ServiceContext
}

func NewBillingListLogic(ctx context.Context, svcCtx *svc.ServiceContext) *BillingListLogic {
	return &BillingListLogic{
		Logger: logx.WithContext(ctx),
		ctx:    ctx,
		svcCtx: svcCtx,
	}
}

func (l *BillingListLogic) BillingList(req *types.BillingListReq) (resp *types.BillingListResp, err error) {
	UserModel := l.svcCtx.UserModel
	BillingModel := l.svcCtx.UserBillingModel

	isAdmin, err := UserModel.IsAdmin(l.ctx, req.UserID)
	if err != nil {
		l.Errorf("UserModel.IsAdmin userID=%s err=%s", req.UserID, err)
		return nil, ErrNotAdmin
	}
	if !isAdmin && req.UserID != req.ToUser {
		l.Infof("UserModel.IsAdmin userID=%s is not admin", req.UserID)
		return nil, ErrNotAdmin
	}

	builder := BillingModel.AllFieldsBuilder()
	if req.ToUser != "" {
		builder = builder.Where(squirrel.Eq{"new_user_id": req.ToUser})
	}
	if req.JobID != "" {
		builder = builder.Where(squirrel.Eq{"job_id": req.JobID})
	}
	if req.StartTime != "" {
		builder = builder.Where(squirrel.GtOrEq{"billing_time": req.StartTime})
	}
	if req.EndTime != "" {
		builder = builder.Where(squirrel.LtOrEq{"billing_time": req.EndTime})
	}

	billings, err := BillingModel.Find(l.ctx, builder)
	if err != nil && !errors.Is(err, model.ErrNotFound) {
		l.Errorf("BillingModel.Find userID=%s jobID=%s startTime=%s endTime=%s err=%s",
			req.ToUser, req.JobID, req.StartTime, req.EndTime, err)
		return nil, ErrDBFind
	}

	resp = &types.BillingListResp{Data: make([]types.UserBilling, 0)}
	_ = copier.Copy(&resp.Data, billings)

	for i, billing := range billings {
		resp.Data[i].BillingTime = billing.BillingTime.Format(time.DateTime)
		if billing.PaymentTime.Valid {
			resp.Data[i].PaymentTime = billing.PaymentTime.Time.Format(time.DateTime)
		}
	}

	return resp, nil
}
