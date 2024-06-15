package logic

import (
	"context"
	"sxwl/3k/internal/scheduler/model"

	"sxwl/3k/internal/scheduler/svc"
	"sxwl/3k/internal/scheduler/types"

	"github.com/Masterminds/squirrel"
	"github.com/zeromicro/go-zero/core/logx"
)

type JupyterlabResumeLogic struct {
	logx.Logger
	ctx    context.Context
	svcCtx *svc.ServiceContext
}

func NewJupyterlabResumeLogic(ctx context.Context, svcCtx *svc.ServiceContext) *JupyterlabResumeLogic {
	return &JupyterlabResumeLogic{
		Logger: logx.WithContext(ctx),
		ctx:    ctx,
		svcCtx: svcCtx,
	}
}

func (l *JupyterlabResumeLogic) JupyterlabResume(req *types.JupyterlabResumeReq) (resp *types.BaseResp, err error) {
	JupyterModel := l.svcCtx.JupyterlabModel

	jupyter, err := JupyterModel.FindOneByJobName(l.ctx, req.JobName)
	if err != nil {
		l.Errorf("find jupyter job_name: %s err: %s", req.JobName, err)
		return nil, ErrJupyterNotFound
	}

	if jupyter.NewUserId != req.UserID {
		l.Errorf("jupyter job_name: %s is not belong user: %s", req.JobName, req.UserID)
		return nil, ErrJupyterNotOwner
	}

	_, err = JupyterModel.UpdateColsByCond(l.ctx, JupyterModel.UpdateBuilder().Where(squirrel.Eq{
		"job_name": req.JobName,
	}).Set("replicas", model.ReplicasRunning))
	if err != nil {
		l.Errorf("jupyter resume job_name: %s replicas err: %s", req.JobName, err)
		return nil, ErrDB
	}

	return &types.BaseResp{Message: MsgJupyterStop}, nil
}
