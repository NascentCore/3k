package logic

import (
	"context"
	"encoding/json"
	"fmt"
	"sxwl/3k/internal/scheduler/model"
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
	selectBuilder = selectBuilder.Where(squirrel.And{
		squirrel.Eq{
			"new_user_id": req.UserID,
		},
		squirrel.NotEq{
			"status": model.StatusStopped,
		},
	})
	jupyterlabs, err := JupyterlabModel.Find(l.ctx, selectBuilder)
	if err != nil {
		l.Errorf("JupyterlabModel.Find user_id: %s err: %s", req.UserID, err)
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
		statusDesc, ok := model.StatusToStr[jupyterlab.Status]
		if ok {
			jupyterlabResp.Status = statusDesc
		}
		jupyterlabResp.UserId = jupyterlab.NewUserId
		err = json.Unmarshal([]byte(jupyterlab.Resource), &jupyterlabResp.Resource)
		if err != nil {
			l.Errorf("json unmarshal jupyterlab: %d err: %s", jupyterlab.Id, err)
		}
		resp.Data = append(resp.Data, jupyterlabResp)
	}

	return
}
