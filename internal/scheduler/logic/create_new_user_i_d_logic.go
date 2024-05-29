package logic

import (
	"context"
	user2 "sxwl/3k/internal/scheduler/user"

	"sxwl/3k/internal/scheduler/svc"
	"sxwl/3k/internal/scheduler/types"

	"github.com/Masterminds/squirrel"
	"github.com/zeromicro/go-zero/core/logx"
)

type CreateNewUserIDLogic struct {
	logx.Logger
	ctx    context.Context
	svcCtx *svc.ServiceContext
}

func NewCreateNewUserIDLogic(ctx context.Context, svcCtx *svc.ServiceContext) *CreateNewUserIDLogic {
	return &CreateNewUserIDLogic{
		Logger: logx.WithContext(ctx),
		ctx:    ctx,
		svcCtx: svcCtx,
	}
}

func (l *CreateNewUserIDLogic) CreateNewUserID() (resp *types.CreateNewUserIDResp, err error) {
	UserModel := l.svcCtx.UserModel

	users, err := UserModel.FindAll(l.ctx, "")
	if err != nil {
		return nil, err
	}

	for _, user := range users {
		// 如果用户没有user_id，生成一个
		if user.NewUserId == "" {
			userID, err := user2.NewUserID()
			if err != nil {
				l.Errorf("new user_id err=%s", err)
				return nil, err
			} else {
				_, err = UserModel.UpdateColsByCond(l.ctx, UserModel.UpdateBuilder().Where(squirrel.Eq{
					"user_id": user.UserId,
				}).Set("new_user_id", userID))
				if err != nil {
					l.Errorf("update user_id id=%d new_user_id=%s err=%s", user.UserId, userID)
					return nil, err
				}
			}
		}
	}

	return &types.CreateNewUserIDResp{Message: "ok"}, nil
}
