package logic

import (
	"context"
	"fmt"
	"sxwl/3k/internal/scheduler/svc"
	"sxwl/3k/internal/scheduler/types"

	"github.com/Masterminds/squirrel"
	"github.com/jinzhu/copier"
	"github.com/zeromicro/go-zero/core/logx"
)

type JupyterlabListLogic struct {
	logx.Logger
	ctx    context.Context
	svcCtx *svc.ServiceContext
}

func NewJupyterlabListLogic(ctx context.Context, svcCtx *svc.ServiceContext) *JupyterlabListLogic {
	return &JupyterlabListLogic{
		Logger: logx.WithContext(ctx),
		ctx:    ctx,
		svcCtx: svcCtx,
	}
}

func (l *JupyterlabListLogic) JupyterlabList(req *types.JupyterlabListReq) (resp *types.JupyterlabListResp, err error) {
	JupyterlabModel := l.svcCtx.JupyterlabModel

	selectBuilder := JupyterlabModel.AllFieldsBuilder()
	selectBuilder = selectBuilder.Where(squirrel.Eq{"user_id": req.UserID})
	jupyterlabs, err := JupyterlabModel.Find(l.ctx, selectBuilder)
	if err != nil {
		l.Errorf("JupyterlabModel.Find user_id: %d err: %s", req.UserID, err)
		return nil, err
	}

	resp = &types.JupyterlabListResp{}
	resp.Data = make([]types.Jupyterlab, 0)
	for _, jupyterlab := range jupyterlabs {
		jupyterlabResp := types.Jupyterlab{}
		_ = copier.Copy(&jupyterlabResp, &jupyterlab)
		jupyterlabResp.GPUProduct = jupyterlab.GpuProd
		jupyterlabResp.Memory = jupyterlab.MemCount
		if jupyterlab.Url != "" {
			jupyterlabResp.URL = fmt.Sprintf("%s%s?token=%s", l.svcCtx.Config.K8S.BaseUrl, jupyterlab.Url, jupyterlab.JobName)
		}

		resp.Data = append(resp.Data, jupyterlabResp)
	}

	return
}