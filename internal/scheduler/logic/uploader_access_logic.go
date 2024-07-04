package logic

import (
	"context"

	"sxwl/3k/internal/scheduler/svc"
	"sxwl/3k/internal/scheduler/types"

	"github.com/zeromicro/go-zero/core/logx"
)

type UploaderAccessLogic struct {
	logx.Logger
	ctx    context.Context
	svcCtx *svc.ServiceContext
}

func NewUploaderAccessLogic(ctx context.Context, svcCtx *svc.ServiceContext) *UploaderAccessLogic {
	return &UploaderAccessLogic{
		Logger: logx.WithContext(ctx),
		ctx:    ctx,
		svcCtx: svcCtx,
	}
}

func (l *UploaderAccessLogic) UploaderAccess(req *types.UploaderAccessReq) (resp *types.UploaderAccessResp, err error) {
	UserModel := l.svcCtx.UserModel

	isAdmin, err := UserModel.IsAdmin(l.ctx, req.UserID)
	if err != nil {
		l.Errorf("UserModel.IsAdmin userID=%s err=%s", req.UserID, err)
		return nil, ErrNotAdmin
	}

	resp = &types.UploaderAccessResp{
		AccessID:  l.svcCtx.Config.OSSAccess.UploadAccessID,
		AccessKey: l.svcCtx.Config.OSSAccess.UploadAccessKey,
		UserID:    req.UserID,
		IsAdmin:   isAdmin,
	}

	return
}
