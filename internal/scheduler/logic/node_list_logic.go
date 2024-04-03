package logic

import (
	"context"
	"fmt"
	"net/http"

	"sxwl/3k/internal/scheduler/svc"
	"sxwl/3k/internal/scheduler/types"

	"github.com/zeromicro/go-zero/core/logx"
	"github.com/zeromicro/go-zero/rest/httpc"
)

type NodeListLogic struct {
	logx.Logger
	ctx    context.Context
	svcCtx *svc.ServiceContext
}

func NewNodeListLogic(ctx context.Context, svcCtx *svc.ServiceContext) *NodeListLogic {
	return &NodeListLogic{
		Logger: logx.WithContext(ctx),
		ctx:    ctx,
		svcCtx: svcCtx,
	}
}

func (l *NodeListLogic) NodeList(req *types.NodeListReq) (resp *types.NodeListResp, err error) {
	url := fmt.Sprintf("%s/nodes", l.svcCtx.Config.K8S.BaseApi)
	nodesResp, err := httpc.Do(context.Background(), http.MethodGet, url, nil)
	if err != nil {
		l.Errorf("NodeList http request url=%s err=%s", url, err)
		return
	}

	var nodes []types.ClusterNodeInfo
	err = httpc.ParseJsonBody(nodesResp, &nodes)
	if err != nil {
		l.Errorf("NodeList parse err=%s", err)
		return nil, err
	}

	resp = &types.NodeListResp{Data: nodes}
	return
}
