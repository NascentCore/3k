package logic

import (
	"context"
	"fmt"
	"io"
	"net/http"

	"sxwl/3k/internal/scheduler/svc"
	"sxwl/3k/internal/scheduler/types"

	"github.com/zeromicro/go-zero/core/logx"
	"github.com/zeromicro/go-zero/rest/httpc"
)

type NodeAddLogic struct {
	logx.Logger
	ctx    context.Context
	svcCtx *svc.ServiceContext
}

func NewNodeAddLogic(ctx context.Context, svcCtx *svc.ServiceContext) *NodeAddLogic {
	return &NodeAddLogic{
		Logger: logx.WithContext(ctx),
		ctx:    ctx,
		svcCtx: svcCtx,
	}
}

func (l *NodeAddLogic) NodeAdd(req *types.NodeAddReq) (resp *types.NodeAddResp, err error) {
	// 检查req信息，是否节点重名
	nodeListLogic := NewNodeListLogic(l.ctx, l.svcCtx)
	listResp, err := nodeListLogic.NodeList(nil)
	if err != nil {
		l.Errorf("NodeAdd list err=%s", err)
		return nil, err
	}

	for _, node := range listResp.Data {
		if node.Name == req.NodeName {
			l.Errorf("NodeAdd name duplicate")
			return nil, fmt.Errorf("node name duplicate")
		}
	}

	url := fmt.Sprintf("%s/add-node", l.svcCtx.Config.K8S.BaseApi)
	addNodeResp, err := httpc.Do(context.Background(), http.MethodPost, url, req.ClusterNode)
	if err != nil {
		l.Errorf("NodeAdd http request url=%s err=%s", url, err)
		return
	}

	if addNodeResp.StatusCode != http.StatusOK {
		// Read the response body
		body, err := io.ReadAll(addNodeResp.Body)
		if err != nil {
			l.Errorf("NodeAdd reading body err=%s", err)
			addNodeResp.Body.Close()
			return nil, err
		}

		addNodeResp.Body.Close()
		return nil, fmt.Errorf("%s", body)
	}

	resp = &types.NodeAddResp{}
	err = httpc.ParseJsonBody(addNodeResp, resp)
	if err != nil {
		l.Errorf("NodeAdd http parse url=%s err=%s", url, err)
		return
	}

	return
}
