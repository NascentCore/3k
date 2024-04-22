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

type BaseImageListLogic struct {
	logx.Logger
	ctx    context.Context
	svcCtx *svc.ServiceContext
}

func NewBaseImageListLogic(ctx context.Context, svcCtx *svc.ServiceContext) *BaseImageListLogic {
	return &BaseImageListLogic{
		Logger: logx.WithContext(ctx),
		ctx:    ctx,
		svcCtx: svcCtx,
	}
}

func (l *BaseImageListLogic) BaseImageList(req *types.BaseImageListReq) (resp *types.BaseImageListResp, err error) {
	url := fmt.Sprintf("%s/base_images", l.svcCtx.Config.K8S.BaseApi)
	baseImagesResp, err := httpc.Do(context.Background(), http.MethodGet, url, nil)
	if err != nil {
		l.Errorf("BaseImageList http request url=%s err=%s", url, err)
		return
	}

	var baseImages []string
	err = httpc.ParseJsonBody(baseImagesResp, &baseImages)
	if err != nil {
		l.Errorf("BaseImageList parse err=%s", err)
		return nil, err
	}

	resp = &types.BaseImageListResp{Data: baseImages}
	return
}
