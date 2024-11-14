package logic

import (
	"context"
	"encoding/json"
	"fmt"
	"sxwl/3k/internal/scheduler/job"
	"sxwl/3k/internal/scheduler/model"
	uuid2 "sxwl/3k/pkg/uuid"

	"sxwl/3k/internal/scheduler/svc"
	"sxwl/3k/internal/scheduler/types"

	"github.com/Masterminds/squirrel"
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
	BalanceModel := l.svcCtx.UserBalanceModel
	JupyterlabModel := l.svcCtx.JupyterlabModel

	// check balance
	balance, err := BalanceModel.FindOneByQuery(l.ctx, BalanceModel.AllFieldsBuilder().Where(squirrel.Eq{
		"new_user_id": req.UserID,
	}))
	if err != nil {
		return nil, err
	}
	if balance.Balance < 0.0 {
		return nil, fmt.Errorf("余额不足")
	}

	// check quota
	ok, left, err := job.CheckQuota(l.ctx, l.svcCtx, req.UserID, req.GPUProduct, req.GPUCount)
	if err != nil {
		l.Errorf("JupyterlabCreate CheckQuota userId: %s GpuType: %s err: %s", req.UserID, req.GPUProduct, err)
		return nil, err
	}
	if !ok {
		err = fmt.Errorf("JupyterlabCreate CheckQuota userId: %s gpu: %s left: %d need: %d", req.UserID, req.GPUProduct, left, req.GPUCount)
		l.Error(err)
		return nil, err
	}

	// check name unique
	jupyterList, err := JupyterlabModel.Find(l.ctx, JupyterlabModel.AllFieldsBuilder().Where(squirrel.And{
		squirrel.Eq{
			"new_user_id": req.UserID,
		},
		squirrel.NotEq{
			"status": model.StatusStopped,
		},
	}))
	if err != nil {
		l.Errorf("JupyterlabCreate find userId: %s err: %s", req.UserID, err)
		return nil, err
	}

	for _, jupyterlab := range jupyterList {
		if jupyterlab.InstanceName == req.InstanceName && jupyterlab.Status != model.StatusStopped {
			l.Errorf("JupyterlabCreate name duplicate userId: %s name: %s", req.UserID, req.InstanceName)
			return nil, fmt.Errorf("实例名字重复")
		}
	}

	// create job id
	jobName, err := uuid2.WithPrefix("jupyter")
	if err != nil {
		l.Errorf("create jupyter job name err=%s", err)
		return nil, ErrSystem
	}

	var billingStatus int64 = model.BillingStatusComplete
	if req.GPUProduct != "" {
		billingStatus = model.BillingStatusContinue
	}

	// resource
	jsonResource, err := json.Marshal(req.Resource)
	if err != nil {
		l.Errorf("json marshal err: %s", err)
		return nil, ErrSystem
	}

	jupyterInstance := model.SysJupyterlab{
		JobName:        jobName,
		NewUserId:      req.UserID,
		Status:         model.StatusNotAssigned,
		BillingStatus:  billingStatus,
		InstanceName:   req.InstanceName,
		GpuCount:       req.GPUCount,
		GpuProd:        req.GPUProduct,
		CpuCount:       req.CPUCount,
		MemCount:       req.Memory,
		DataVolumeSize: req.DataVolumeSize,
		Resource:       string(jsonResource),
		Replicas:       model.ReplicasRunning,
	}
	if req.CpodID != "" {
		jupyterInstance.CpodId = req.CpodID
		jupyterInstance.Status = model.StatusAssigned
	}
	_, err = JupyterlabModel.Insert(l.ctx, &jupyterInstance)
	if err != nil {
		l.Errorf("insert userId: %s err: %s", req.UserID, err)
		return nil, err
	}

	resp = &types.JupyterlabCreateResp{Message: fmt.Sprintf("jupyterlab:%s 创建成功", jobName)}
	return
}
