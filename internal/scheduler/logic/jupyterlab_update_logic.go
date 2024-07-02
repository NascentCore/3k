package logic

import (
	"context"
	"sxwl/3k/internal/scheduler/model"

	"sxwl/3k/internal/scheduler/svc"
	"sxwl/3k/internal/scheduler/types"

	"github.com/zeromicro/go-zero/core/logx"
)

type JupyterlabUpdateLogic struct {
	logx.Logger
	ctx    context.Context
	svcCtx *svc.ServiceContext
}

func NewJupyterlabUpdateLogic(ctx context.Context, svcCtx *svc.ServiceContext) *JupyterlabUpdateLogic {
	return &JupyterlabUpdateLogic{
		Logger: logx.WithContext(ctx),
		ctx:    ctx,
		svcCtx: svcCtx,
	}
}

func (l *JupyterlabUpdateLogic) JupyterlabUpdate(req *types.JupyterlabUpdateReq) (resp *types.BaseResp, err error) {
	JupyterModel := l.svcCtx.JupyterlabModel

	jupyterlab, err := JupyterModel.FindOneByJobName(l.ctx, req.JobName)
	if err != nil {
		l.Errorf("JupyterModel FindOneByJobName jobname: %s err: %s", req.JobName, err)
		return nil, ErrDBFind
	}

	if jupyterlab.NewUserId != req.UserID {
		l.Errorf("jupyter job_name: %s is not belong user: %s", req.JobName, req.UserID)
		return nil, ErrJupyterNotOwner
	}

	// jupyterlab should be paused status
	if jupyterlab.Status != model.StatusPaused {
		return nil, ErrJupyterNotPaused
	}

	setMap := make(map[string]interface{})
	if jupyterlab.CpuCount != req.CPUCount {
		setMap["cpu_count"] = req.CPUCount
	}
	if jupyterlab.MemCount != req.Memory {
		setMap["mem_count"] = req.Memory
	}
	if jupyterlab.GpuCount != req.GPUCount {
		setMap["gpu_count"] = req.GPUCount
	}
	if jupyterlab.GpuProd != req.GPUProduct {
		setMap["gpu_product"] = req.GPUProduct
	}
	if jupyterlab.DataVolumeSize != req.DataVolumeSize {
		setMap["data_volume_size"] = req.DataVolumeSize
	}
	if len(setMap) == 0 {
		return nil, ErrJupyterNoUpdate
	}

	_, err = JupyterModel.UpdateColsByCond(l.ctx, JupyterModel.UpdateBuilder().Where("job_name = ?", req.JobName).SetMap(setMap))
	if err != nil {
		l.Errorf("JupyterModel UpdateColsByCond jobname: %s err: %s", req.JobName, err)
		return nil, ErrDB
	}

	// 如果修改了gpu类型或数量需要立即触发账单生成，否则漏单部分会计算错误
	// 这里触发的意义并不大，因为每分钟支付系统都会自动生成账单。
	// if 系统异常 then 这里理论上根本也走不到
	// if 系统正常 then 每分钟的账单生成就能正确的处理

	return &types.BaseResp{Message: MsgJupyterUpdate}, nil
}
