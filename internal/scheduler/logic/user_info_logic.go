package logic

import (
	"context"
	"fmt"
	"sxwl/3k/internal/scheduler/svc"
	"sxwl/3k/internal/scheduler/types"
	user2 "sxwl/3k/internal/scheduler/user"
	"time"

	"github.com/Masterminds/squirrel"
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

func (l *UserInfoLogic) UserInfo(req *types.UserInfoReq) (resp *types.UserInfoResp, err error) {
	UserModel := l.svcCtx.UserModel
	user, err := UserModel.FindOneByUserID(l.ctx, req.UserID)
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
	resp.User.ID = user.UserId
	// user_id
	resp.User.UserID = user.NewUserId
	// isAdmin
	if user.Admin > 0 {
		resp.User.IsAdmin = true
	}

	// 如果用户没有user_id，生成一个
	if user.NewUserId == "" {
		userID, err := user2.NewUserID()
		if err != nil {
			l.Errorf("new user_id err=%s", err)
		} else {
			_, err = UserModel.UpdateColsByCond(l.ctx, UserModel.UpdateBuilder().Where(squirrel.Eq{
				"user_id": user.UserId,
			}).Set("new_user_id", userID))
			if err != nil {
				l.Errorf("update user_id id=%d new_user_id=%s err=%s", user.UserId, userID)
			}
		}
	}

	return
}
