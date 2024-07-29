package logic

import (
	"context"
	"sxwl/3k/internal/scheduler/resource"

	"sxwl/3k/internal/scheduler/svc"
	"sxwl/3k/internal/scheduler/types"

	"github.com/zeromicro/go-zero/core/logx"
)

type OssSyncLogic struct {
	logx.Logger
	ctx    context.Context
	svcCtx *svc.ServiceContext
}

func NewOssSyncLogic(ctx context.Context, svcCtx *svc.ServiceContext) *OssSyncLogic {
	return &OssSyncLogic{
		Logger: logx.WithContext(ctx),
		ctx:    ctx,
		svcCtx: svcCtx,
	}
}

func (l *OssSyncLogic) OssSync(req *types.BaseReq) (resp *types.BaseResp, err error) {
	UserModel := l.svcCtx.UserModel

	isAdmin, err := UserModel.IsAdmin(l.ctx, req.UserID)
	if err != nil {
		l.Errorf("UserModel.IsAdmin userID=%s err=%s", req.UserID, err)
		return nil, ErrNotAdmin
	}
	if !isAdmin {
		l.Infof("UserModel.IsAdmin userID=%s is not admin", req.UserID)
		return nil, ErrNotAdmin
	}

	go func() {
		resource.NewManager(l.svcCtx).SyncOSS()
	}()
	return &types.BaseResp{Message: MsgOssSyncBegin}, nil
}
