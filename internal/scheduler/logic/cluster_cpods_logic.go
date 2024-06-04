package logic

import (
	"context"
	"time"

	"sxwl/3k/internal/scheduler/svc"
	"sxwl/3k/internal/scheduler/types"

	"github.com/Masterminds/squirrel"
	"github.com/jinzhu/copier"
	"github.com/zeromicro/go-zero/core/logx"
)

type ClusterCpodsLogic struct {
	logx.Logger
	ctx    context.Context
	svcCtx *svc.ServiceContext
}

func NewClusterCpodsLogic(ctx context.Context, svcCtx *svc.ServiceContext) *ClusterCpodsLogic {
	return &ClusterCpodsLogic{
		Logger: logx.WithContext(ctx),
		ctx:    ctx,
		svcCtx: svcCtx,
	}
}

func (l *ClusterCpodsLogic) ClusterCpods(req *types.BaseReq) (resp *types.ClusterCpodsResp, err error) {
	CpodModel := l.svcCtx.CpodMainModel
	cpods, err := CpodModel.Find(l.ctx, CpodModel.AllFieldsBuilder().Where(
		squirrel.Expr("update_time > NOW() - INTERVAL 30 MINUTE"),
	))
	if err != nil {
		l.Errorf("find active cpods err: %s", err)
		return nil, ErrDBFind
	}

	resp = &types.ClusterCpodsResp{Data: make([]types.CpodInfo, 0)}
	for _, cpod := range cpods {
		cpodInfo := types.CpodInfo{}
		_ = copier.Copy(&cpodInfo, cpod)
		cpodInfo.CreateTime = cpod.CreateTime.Time.Format(time.DateTime)
		cpodInfo.UpdateTime = cpod.UpdateTime.Time.Format(time.DateTime)
		resp.Data = append(resp.Data, cpodInfo)
	}

	return
}
