package logic

import (
	"context"

	"sxwl/3k/internal/scheduler/svc"
	"sxwl/3k/internal/scheduler/types"

	"github.com/zeromicro/go-zero/core/logx"
)

type UserListLogic struct {
	logx.Logger
	ctx    context.Context
	svcCtx *svc.ServiceContext
}

func NewUserListLogic(ctx context.Context, svcCtx *svc.ServiceContext) *UserListLogic {
	return &UserListLogic{
		Logger: logx.WithContext(ctx),
		ctx:    ctx,
		svcCtx: svcCtx,
	}
}

func (l *UserListLogic) UserList() (resp *types.UserListResp, err error) {
	UserModel := l.svcCtx.UserModel

	userList, err := UserModel.FindAll(l.ctx, "")
	if err != nil {
		l.Errorf("UserList err=%s", err)
		return nil, err
	}

	resp = &types.UserListResp{Data: make([]types.User, 0)}
	for _, user := range userList {
		resp.Data = append(resp.Data, types.User{
			UserId:   user.UserId,
			UserName: user.Username.String,
		})
	}

	return
}
