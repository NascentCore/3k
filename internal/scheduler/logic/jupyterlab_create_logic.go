package logic

import (
	"context"
	"fmt"
	"sxwl/3k/internal/scheduler/job"
	"sxwl/3k/internal/scheduler/model"

	"sxwl/3k/internal/scheduler/svc"
	"sxwl/3k/internal/scheduler/types"

	"github.com/Masterminds/squirrel"
	"github.com/google/uuid"
	"github.com/zeromicro/go-zero/core/logx"
)

type JupyterlabCreateLogic struct {
	logx.Logger
	ctx    context.Context
	svcCtx *svc.ServiceContext
}

func NewJupyterlabCreateLogic(ctx context.Context, svcCtx *svc.ServiceContext) *JupyterlabCreateLogic {
	return &JupyterlabCreateLogic{
		Logger: logx.WithContext(ctx),
		ctx:    ctx,
		svcCtx: svcCtx,
	}
}

func (l *JupyterlabCreateLogic) JupyterlabCreate(req *types.JupyterlabCreateReq) (resp *types.JupyterlabCreateResp, err error) {
	JupyterlabModel := l.svcCtx.JupyterlabModel

	// check quota
	ok, left, err := job.CheckQuota(l.ctx, l.svcCtx, req.UserID, req.GPUProduct, req.GPUCount)
	if err != nil {
		l.Errorf("JupyterlabCreate CheckQuota userId: %d GpuType: %s err: %s", req.UserID, req.GPUProduct, err)
		return nil, err
	}
	if !ok {
		err = fmt.Errorf("JupyterlabCreate CheckQuota userId: %d gpu: %s left: %d need: %d", req.UserID, req.GPUProduct, left, req.GPUCount)
		l.Error(err)
		return nil, err
	}

	// check name unique
	jupyterList, err := JupyterlabModel.Find(l.ctx, JupyterlabModel.AllFieldsBuilder().Where(squirrel.Eq{
		"user_id": req.UserId,
	}))
	if err != nil {
		l.Errorf("JupyterlabCreate find userId: %d err: %s", req.UserId, err)
		return nil, err
	}

	for _, jupyterlab := range jupyterList {
		if jupyterlab.InstanceName == req.InstanceName && jupyterlab.Status != model.JupyterStatusStopped {
			l.Errorf("JupyterlabCreate name duplicate userId: %d name: %s", req.UserID, req.InstanceName)
			return nil, fmt.Errorf("实例名字重复")
		}
	}

	newUUID, err := uuid.NewRandom()
	if err != nil {
		l.Errorf("new uuid userId: %d err: %s", req.UserID, err)
		return nil, err
	}
	jobName := "jupyter-" + newUUID.String()

	jupyterInstance := model.SysJupyterlab{
		JobName:        jobName,
		UserId:         req.UserId,
		Status:         model.JupyterStatusWaitDeploy,
		InstanceName:   req.InstanceName,
		GpuCount:       req.GPUCount,
		GpuProd:        req.GPUProduct,
		CpuCount:       req.CPUCount,
		MemCount:       req.Memory,
		DataVolumeSize: req.DataVolumeSize,
		ModelId:        req.ModelId,
		ModelName:      req.ModelName,
		ModelPath:      req.ModelPath,
	}
	_, err = JupyterlabModel.Insert(l.ctx, &jupyterInstance)
	if err != nil {
		l.Errorf("insert userId: %d err: %s", req.UserID, err)
		return nil, err
	}

	resp = &types.JupyterlabCreateResp{Message: fmt.Sprintf("jupyterlab:%s 创建成功", jobName)}
	return
}
