package logic

import (
	"context"
	"sxwl/3k/internal/scheduler/model"
	"sxwl/3k/pkg/uuid"

	"sxwl/3k/internal/scheduler/svc"
	"sxwl/3k/internal/scheduler/types"

	"github.com/zeromicro/go-zero/core/logx"
)

type AppRegisterLogic struct {
	logx.Logger
	ctx    context.Context
	svcCtx *svc.ServiceContext
}

func NewAppRegisterLogic(ctx context.Context, svcCtx *svc.ServiceContext) *AppRegisterLogic {
	return &AppRegisterLogic{
		Logger: logx.WithContext(ctx),
		ctx:    ctx,
		svcCtx: svcCtx,
	}
}

func (l *AppRegisterLogic) AppRegister(req *types.AppRegisterReq) (resp *types.BaseResp, err error) {
	UserModel := l.svcCtx.UserModel
	AppModel := l.svcCtx.AppModel

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

	// app id
	appID, err := uuid.WithPrefix("app")
	if err != nil {
		l.Errorf("create app id err=%s", err)
		return nil, ErrSystem
	}

	_, err = AppModel.Insert(l.ctx, &model.SysApp{
		AppId:   appID,
		AppName: req.Name,
		UserId:  req.UserID,
		Desc:    req.Desc,
		Crd:     req.CRD,
		Status:  model.AppStatusApproved,
	})
	if err != nil {
		l.Errorf("AppModel insert userID=%s name=%s err=%s", req.UserID, req.Name, err)
		return nil, ErrDB
	}

	return &types.BaseResp{Message: "应用注册成功"}, nil
}
