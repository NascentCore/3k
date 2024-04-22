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

type JupyterlabImageDelLogic struct {
	logx.Logger
	ctx    context.Context
	svcCtx *svc.ServiceContext
}

func NewJupyterlabImageDelLogic(ctx context.Context, svcCtx *svc.ServiceContext) *JupyterlabImageDelLogic {
	return &JupyterlabImageDelLogic{
		Logger: logx.WithContext(ctx),
		ctx:    ctx,
		svcCtx: svcCtx,
	}
}

func (l *JupyterlabImageDelLogic) JupyterlabImageDel(req *types.JupyterlabImageDelReq) (resp *types.JupyterlabImageDelResp, err error) {
	url := fmt.Sprintf("%s/delete_repo", l.svcCtx.Config.K8S.BaseApi)
	imageDelResp, err := httpc.Do(context.Background(), http.MethodDelete, url, types.ImageDelReq{
		UserID:    int(req.UserID),
		ImageName: req.ImageName,
		Tag:       req.TagName,
	})

	if err != nil {
		l.Errorf("delete_repo http request url=%s err=%s", url, err)
		return
	}

	if imageDelResp.StatusCode != http.StatusOK {
		// Read the response body
		body, err := io.ReadAll(imageDelResp.Body)
		if err != nil {
			l.Errorf("delete_repo reading body err=%s", err)
			imageDelResp.Body.Close()
			return nil, err
		}

		imageDelResp.Body.Close()
		return nil, fmt.Errorf("%s", body)
	}

	status := struct {
		Status string `json:"status"`
	}{}
	err = httpc.ParseJsonBody(imageDelResp, &status)
	if err != nil {
		l.Errorf("JupyterlabImageDel http parse url=%s err=%s", url, err)
		return
	}

	resp = &types.JupyterlabImageDelResp{Message: status.Status}

	return
}
