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

type JupyterlabImageVersionListLogic struct {
	logx.Logger
	ctx    context.Context
	svcCtx *svc.ServiceContext
}

func NewJupyterlabImageVersionListLogic(ctx context.Context, svcCtx *svc.ServiceContext) *JupyterlabImageVersionListLogic {
	return &JupyterlabImageVersionListLogic{
		Logger: logx.WithContext(ctx),
		ctx:    ctx,
		svcCtx: svcCtx,
	}
}

func (l *JupyterlabImageVersionListLogic) JupyterlabImageVersionList(req *types.JupyterlabImageVersionListReq) (resp *types.JupyterlabImageVersionListResp, err error) {
	url := fmt.Sprintf("%s/repos?user_id=%s&instance_name=%s", l.svcCtx.Config.K8S.BaseApi, req.UserID, req.InstanceName)
	imageListResp, err := httpc.Do(context.Background(), http.MethodGet, url, nil)
	if err != nil {
		l.Errorf("repos http request url=%s err=%s", url, err)
		return
	}

	var imageList []types.JupyterlabImageVersion
	err = httpc.ParseJsonBody(imageListResp, &imageList)
	if err != nil {
		l.Errorf("repos parse err=%s", err)
		return nil, err
	}

	resp = &types.JupyterlabImageVersionListResp{Data: imageList}
	return
}
