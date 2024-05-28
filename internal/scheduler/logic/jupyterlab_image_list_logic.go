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

type JupyterlabImageListLogic struct {
	logx.Logger
	ctx    context.Context
	svcCtx *svc.ServiceContext
}

func NewJupyterlabImageListLogic(ctx context.Context, svcCtx *svc.ServiceContext) *JupyterlabImageListLogic {
	return &JupyterlabImageListLogic{
		Logger: logx.WithContext(ctx),
		ctx:    ctx,
		svcCtx: svcCtx,
	}
}

func (l *JupyterlabImageListLogic) JupyterlabImageList(req *types.JupyterlabImageListReq) (resp *types.JupyterlabImageListResp, err error) {
	url := fmt.Sprintf("%s/repos?user_id=%s", l.svcCtx.Config.K8S.BaseApi, req.UserID)
	imageListResp, err := httpc.Do(context.Background(), http.MethodGet, url, nil)
	if err != nil {
		l.Errorf("repos http request url=%s err=%s", url, err)
		return
	}

	var imageList []types.JupyterlabImage
	err = httpc.ParseJsonBody(imageListResp, &imageList)
	if err != nil {
		l.Errorf("repos parse err=%s", err)
		return nil, err
	}

	resp = &types.JupyterlabImageListResp{Data: imageList}
	return
}
