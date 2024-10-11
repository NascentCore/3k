package logic

import (
	"context"
	"errors"
	"sxwl/3k/internal/scheduler/model"

	"sxwl/3k/internal/scheduler/svc"
	"sxwl/3k/internal/scheduler/types"

	"github.com/Masterminds/squirrel"
	"github.com/zeromicro/go-zero/core/logx"
)

type AppUnregisterLogic struct {
	logx.Logger
	ctx    context.Context
	svcCtx *svc.ServiceContext
}

func NewAppUnregisterLogic(ctx context.Context, svcCtx *svc.ServiceContext) *AppUnregisterLogic {
	return &AppUnregisterLogic{
		Logger: logx.WithContext(ctx),
		ctx:    ctx,
		svcCtx: svcCtx,
	}
}

func (l *AppUnregisterLogic) AppUnregister(req *types.AppUnregisterReq) (resp *types.BaseResp, err error) {
	UserModel := l.svcCtx.UserModel
	AppModel := l.svcCtx.AppModel
	AppJobModel := l.svcCtx.AppJobModel

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

	// Get app
	app, err := AppModel.FindOneByQuery(l.ctx, AppModel.AllFieldsBuilder().Where(squirrel.Eq{"app_id": req.AppID}))
	if err != nil {
		l.Errorf("AppModel.FindOneByQuery err=%s", err)
		return nil, ErrAppNotExists
	}

	// Cannot unregister the job because it has jobs associated with it.
	job, errDuplicate := AppJobModel.FindOneByQuery(l.ctx, AppJobModel.AllFieldsBuilder().Where(squirrel.And{
		squirrel.Eq{
			"app_id": req.AppID,
		},
		squirrel.NotEq{
			"status": model.StatusDeleted,
		},
	}))
	if !errors.Is(errDuplicate, model.ErrNotFound) {
		l.Errorf("Unregister job %s %s but has jobs", job.AppId, job.AppName)
		return nil, ErrAppHasJobs
	}

	err = AppModel.Delete(l.ctx, app.Id)
	if err != nil {
		l.Errorf("Unregister app %s %s err: %s", app.AppId, app.AppName, err)
		return nil, ErrAppHasJobs
	}

	return &types.BaseResp{Message: MsgAppUnregisterOK}, nil
}
