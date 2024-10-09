package logic

import (
	"context"
	"time"

	"sxwl/3k/internal/scheduler/svc"
	"sxwl/3k/internal/scheduler/types"

	"github.com/zeromicro/go-zero/core/logx"
)

type AppListLogic struct {
	logx.Logger
	ctx    context.Context
	svcCtx *svc.ServiceContext
}

func NewAppListLogic(ctx context.Context, svcCtx *svc.ServiceContext) *AppListLogic {
	return &AppListLogic{
		Logger: logx.WithContext(ctx),
		ctx:    ctx,
		svcCtx: svcCtx,
	}
}

func (l *AppListLogic) AppList(req *types.BaseReq) (resp *types.AppListResp, err error) {
	AppModel := l.svcCtx.AppModel

	apps, err := AppModel.FindAll(l.ctx, "")
	if err != nil {
		l.Errorf("AppModel.FindAll userID=%s err=%s", req.UserID, err)
		return nil, ErrDBFind
	}

	resp = &types.AppListResp{
		Data: make([]types.App, 0),
	}
	for _, app := range apps {
		resp.Data = append(resp.Data, types.App{
			ID:        app.Id,
			AppID:     app.AppId,
			AppName:   app.AppName,
			UserID:    app.UserId,
			Desc:      app.Desc,
			CRD:       app.Crd,
			Status:    app.Status,
			CreatedAt: app.CreatedAt.Format(time.DateTime),
			UpdatedAt: app.UpdatedAt.Format(time.DateTime),
		})
	}
	resp.Total = int64(len(apps))

	return
}
