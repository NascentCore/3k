package logic

import (
	"context"
	"sxwl/3k/internal/scheduler/model"
	"sxwl/3k/pkg/uuid"

	"sxwl/3k/internal/scheduler/svc"
	"sxwl/3k/internal/scheduler/types"

	"github.com/Masterminds/squirrel"
	"github.com/zeromicro/go-zero/core/logx"
)

type AppJobAddLogic struct {
	logx.Logger
	ctx    context.Context
	svcCtx *svc.ServiceContext
}

func NewAppJobAddLogic(ctx context.Context, svcCtx *svc.ServiceContext) *AppJobAddLogic {
	return &AppJobAddLogic{
		Logger: logx.WithContext(ctx),
		ctx:    ctx,
		svcCtx: svcCtx,
	}
}

func (l *AppJobAddLogic) AppJobAdd(req *types.AppJobAddReq) (resp *types.BaseResp, err error) {
	AppModel := l.svcCtx.AppModel
	AppJobModel := l.svcCtx.AppJobModel

	// check app id
	app, err := AppModel.FindOneByQuery(l.ctx, AppModel.AllFieldsBuilder().Where(squirrel.Eq{
		"app_id": req.AppId,
	}))
	if err != nil {
		l.Errorf("Find app user_id: %s err: %v", req.UserID, err)
		return nil, ErrDBFind
	}

	if app.AppName != req.AppName {
		l.Errorf("Name not match %s %s", app.AppName, req.AppName)
		return nil, ErrDBFind
	}

	// job name
	jobName, err := uuid.WithPrefix("appjob")
	if err != nil {
		l.Errorf("create job name err: %s", err)
		return nil, ErrSystem
	}

	_, err = AppJobModel.Insert(l.ctx, &model.SysAppJob{
		JobName:       jobName,
		UserId:        req.UserID,
		AppId:         req.AppId,
		AppName:       req.AppName,
		InstanceName:  req.InstanceName,
		Status:        model.StatusNotAssigned,
		BillingStatus: model.BillingStatusComplete,
	})
	if err != nil {
		l.Errorf("create job insert user_id: %s err: %s", err)
		return nil, ErrDB
	}

	return
}
