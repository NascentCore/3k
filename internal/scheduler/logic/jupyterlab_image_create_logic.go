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

type JupyterlabImageCreateLogic struct {
	logx.Logger
	ctx    context.Context
	svcCtx *svc.ServiceContext
}

func NewJupyterlabImageCreateLogic(ctx context.Context, svcCtx *svc.ServiceContext) *JupyterlabImageCreateLogic {
	return &JupyterlabImageCreateLogic{
		Logger: logx.WithContext(ctx),
		ctx:    ctx,
		svcCtx: svcCtx,
	}
}

func (l *JupyterlabImageCreateLogic) JupyterlabImageCreate(req *types.JupyterlabImageCreateReq) (resp *types.JupyterlabImageCreateResp, err error) {
	url := fmt.Sprintf("%s/build_image", l.svcCtx.Config.K8S.BaseApi)
	// TODO types.BuildImageReq 可以用 types.JupyterlabImageCreateReq 代替掉
	buildImageResp, err := httpc.Do(context.Background(), http.MethodPost, url, types.BuildImageReq{
		BaseImage:    req.BaseImage,
		UserID:       req.UserID,
		InstanceName: req.InstanceName,
		JobName:      req.JobName,
	})

	if err != nil {
		l.Errorf("build_image http request url=%s err=%s", url, err)
		return
	}

	if buildImageResp.StatusCode != http.StatusOK {
		// Read the response body
		body, err := io.ReadAll(buildImageResp.Body)
		if err != nil {
			l.Errorf("build_image reading body err=%s", err)
			buildImageResp.Body.Close()
			return nil, err
		}

		buildImageResp.Body.Close()
		return nil, fmt.Errorf("%s", body)
	}

	resp = &types.JupyterlabImageCreateResp{}
	err = httpc.ParseJsonBody(buildImageResp, resp)
	if err != nil {
		l.Errorf("JupyterlabImageCreate http parse url=%s err=%s", url, err)
		return
	}
	return
}
