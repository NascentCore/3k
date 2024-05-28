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

	// TODO 增加配额的检查
	// 是否超过配额上限

	_, err = QuotaModel.FindOneByQuery(l.ctx, QuotaModel.AllFieldsBuilder().Where(squirrel.Eq{
		"new_user_id": req.UserId,
		"resource":    req.Resource,
	}))
	if !errors.Is(err, model.ErrNotFound) {
		err = fmt.Errorf("QuotaAdd quota already exists user_id=%s resource=%s", req.UserID, req.Resource)
		l.Error(err)
		return nil, err
	}

	_, err = QuotaModel.Insert(l.ctx, &model.SysQuota{
		NewUserId: req.UserId,
		Resource:  req.Resource,
		Quota:     req.Quota.Quota,
	})
	if err != nil {
		l.Errorf("QuotaAdd user_id=%s resource=%s quota=%d", req.UserID, req.Resource, req.Quota.Quota)
		return nil, err
	}

	resp = &types.QuotaAddResp{Message: "ok"}
	return
}
