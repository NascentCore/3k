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
	CpodNodeModel := l.svcCtx.CpodNodeModel
	nodes, err := CpodNodeModel.Find(l.ctx, CpodNodeModel.AllFieldsBuilder().Where(
		squirrel.Expr("updated_at > NOW() - INTERVAL 30 MINUTE"),
	))
	if err != nil {
		l.Errorf("find active nodes err: %s", err)
		return nil, ErrDBFind
	}

	resp = &types.ClusterCpodsResp{Data: make([]types.CpodInfo, 0)}
	for _, node := range nodes {
		cpodInfo := types.CpodInfo{}
		_ = copier.Copy(&cpodInfo, node)
		cpodInfo.CreateTime = node.CreatedAt.Format(time.DateTime)
		cpodInfo.UpdateTime = node.UpdatedAt.Format(time.DateTime)
		resp.Data = append(resp.Data, cpodInfo)
	}

	return
}
