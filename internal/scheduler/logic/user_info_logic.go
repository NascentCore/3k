package logic

import (
	"context"
	"fmt"
	"sxwl/3k/internal/scheduler/svc"
	"sxwl/3k/internal/scheduler/types"
	"time"

	"github.com/jinzhu/copier"
	"github.com/zeromicro/go-zero/core/logx"
)

type UserInfoLogic struct {
	logx.Logger
	ctx    context.Context
	svcCtx *svc.ServiceContext
}

func NewUserInfoLogic(ctx context.Context, svcCtx *svc.ServiceContext) *UserInfoLogic {
	return &UserInfoLogic{
		Logger: logx.WithContext(ctx),
		ctx:    ctx,
		svcCtx: svcCtx,
	}
}

func (l *UserInfoLogic) UserInfo(req *types.AuthReq) (resp *types.UserInfoResp, err error) {
	UserModel := l.svcCtx.UserModel
	user, err := UserModel.FindOne(l.ctx, req.UserID)
	if err != nil {
		return nil, fmt.Errorf("UserInfo err=%s", err)
	}

	resp = &types.UserInfoResp{User: types.UserInfo{}}
	_ = copier.Copy(&resp.User, user)

	// createTime
	resp.User.CreateTime = user.CreateTime.Time.Format(time.DateTime)
	// updateTime
	resp.User.UpdateTime = user.UpdateTime.Time.Format(time.DateTime)
	// enabled
	if user.Enabled.Valid && user.Enabled.Int64 > 0 {
		resp.User.Enabled = true
	}
	// id
	resp.User.ID = int(req.UserID)
	// isAdmin
	if user.Admin > 0 {
		resp.User.IsAdmin = true
	}
	// TODO remove password

	return
}
