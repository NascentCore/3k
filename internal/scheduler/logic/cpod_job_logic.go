package logic

import (
	"context"
	"database/sql"
	"encoding/json"
	"fmt"
	"strconv"
	"sxwl/3k/internal/scheduler/model"
	"time"

	"github.com/jinzhu/copier"

	"github.com/Masterminds/squirrel"

	"sxwl/3k/internal/scheduler/svc"
	"sxwl/3k/internal/scheduler/types"

	"github.com/zeromicro/go-zero/core/logx"
)

type CpodJobLogic struct {
	logx.Logger
	ctx    context.Context
	svcCtx *svc.ServiceContext
}

func NewCpodJobLogic(ctx context.Context, svcCtx *svc.ServiceContext) *CpodJobLogic {
	return &CpodJobLogic{
		Logger: logx.WithContext(ctx),
		ctx:    ctx,
		svcCtx: svcCtx,
	}
}

func (l *CpodJobLogic) CpodJob(req *types.CpodJobReq) (resp *types.CpodJobResp, err error) {
	CpodMainModel := l.svcCtx.CpodMainModel
	UserJobModel := l.svcCtx.UserJobModel
	InferenceModel := l.svcCtx.InferenceModel
	JupyterlabModel := l.svcCtx.JupyterlabModel

	// check banned
	_, ok := l.svcCtx.Config.BannedCpod[req.CpodId]
	if ok {
		return nil, fmt.Errorf("cpod is illegal")
	}

	resp = &types.CpodJobResp{}
	resp.JobList = make([]map[string]interface{}, 0)
	resp.InferenceServiceList = make([]types.InferenceService, 0)

	gpus, err := CpodMainModel.Find(l.ctx, CpodMainModel.AllFieldsBuilder().Where(squirrel.Eq{
		"cpod_id": req.CpodId,
	}))
	if err != nil {
		l.Logger.Errorf("cpod_main find cpod_id=%s err=%s", req.CpodId, err)
		return nil, err
	}

	jobs, err := UserJobModel.Find(l.ctx, UserJobModel.AllFieldsBuilder().Where(squirrel.Eq{
		"obtain_status": model.JobStatusObtainNeedSend,
		"deleted":       0,
	}))
	if err != nil {
		l.Logger.Errorf("user_job find obtain_status=%d deleted=0 err=%s", model.JobStatusObtainNeedSend, err)
		return nil, err
	}

	activeJobs := make([]*model.SysUserJob, 0)    // 已经在运行中的任务
	assignedJobs := make([]*model.SysUserJob, 0)  // 本次被分配的任务
	assignedGPUs := make([]*model.SysCpodMain, 0) // 本次被分配任务的gpu

	for _, job := range jobs {
		if job.CpodId.String == "" {
			for _, gpu := range gpus {
				if gpu.GpuProd == job.GpuType && gpu.GpuAllocatable.Int64 >= job.GpuNumber.Int64 {
					assignedJobs = append(assignedJobs, job)
					gpu.GpuAllocatable.Int64 -= job.GpuNumber.Int64
					assignedGPUs = append(assignedGPUs, gpu)
					break
				}
			}
		} else {
			if job.CpodId.String == req.CpodId {
				activeJobs = append(activeJobs, job)
			}
		}
	}

	for _, job := range assignedJobs {
		_, err = UserJobModel.UpdateColsByCond(l.ctx, UserJobModel.UpdateBuilder().Where(squirrel.Eq{
			"job_id": job.JobId,
		}).SetMap(map[string]interface{}{
			"cpod_id":     req.CpodId,
			"update_time": sql.NullTime{Time: time.Now(), Valid: true},
		}))
		if err != nil {
			l.Logger.Errorf("user_job assigned job_id=%d job_name=%s cpod_id=%s err=%s",
				job.JobId, job.JobName.String, req.CpodId, err)
			return nil, err
		}

		l.Logger.Infof("user_job assigned job_id=%d job_name=%s cpod_id=%s", job.JobId, job.JobName.String, req.CpodId)

		// set to resp
		cpodJobResp := map[string]any{}
		err = json.Unmarshal([]byte(job.JsonAll.String), &cpodJobResp)
		if err != nil {
			l.Logger.Errorf("unmarshal json=%s err=%s", job.JsonAll.String, err)
			continue
		}
		resp.JobList = append(resp.JobList, cpodJobResp)
	}

	for _, job := range activeJobs {
		// set to resp
		cpodJobResp := map[string]any{}
		err = json.Unmarshal([]byte(job.JsonAll.String), &cpodJobResp)
		if err != nil {
			l.Logger.Errorf("unmarshal json=%s err=%s", job.JsonAll.String, err)
			continue
		}
		resp.JobList = append(resp.JobList, cpodJobResp)
	}

	for _, gpu := range assignedGPUs {
		_, err = CpodMainModel.UpdateColsByCond(l.ctx, CpodMainModel.UpdateBuilder().Where(squirrel.Eq{
			"main_id": gpu.MainId,
		}).SetMap(map[string]interface{}{
			"gpu_allocatable": gpu.GpuAllocatable,
			"update_time":     sql.NullTime{Time: time.Now(), Valid: true},
		}))
		if err != nil {
			l.Logger.Errorf("cpod_main assigned main_id=%d gpu_allocatable=%d gpu_total=%d err=%s",
				gpu.MainId, gpu.GpuAllocatable.Int64, gpu.GpuTotal.Int64, err)
			return nil, err
		}
		l.Logger.Infof("cpod_main assigned main_id=%d gpu_allocatable=%d gpu_total=%d",
			gpu.MainId, gpu.GpuAllocatable.Int64, gpu.GpuTotal.Int64)
	}

	// inference services
	services, err := InferenceModel.FindAll(l.ctx, InferenceModel.AllFieldsBuilder().Where(
		squirrel.Or{
			squirrel.Eq{"status": model.InferStatusWaitDeploy},
			squirrel.And{squirrel.Eq{"cpod_id": req.CpodId}, squirrel.NotEq{"status": model.InferStatusStopped}},
		},
	), "")
	if err != nil {
		l.Errorf("InferenceModel.FindAll err: %s", err)
		return nil, err
	}

	for _, service := range services {
		serviceResp := types.InferenceService{}
		_ = copier.Copy(&serviceResp, service)
		statusDesc, ok := model.InferStatusToDesc[service.Status]
		if ok {
			serviceResp.Status = statusDesc
		}
		serviceResp.Template = service.Template.String
		serviceResp.UserId = service.UserId
		resp.InferenceServiceList = append(resp.InferenceServiceList, serviceResp)

		// 新分配的部署更新cpod_id和status
		// 2024-01-31 目前逻辑简单粗暴，把所有等待部署的任务都下发下去。更完善的方案应该是考虑到推理部署的GPU占用，不要超卖，否则cpod
		// 运行不了那么多的推理部署。
		if service.CpodId == "" && service.Status == model.InferStatusWaitDeploy {
			_, err = InferenceModel.UpdateColsByCond(l.ctx, InferenceModel.UpdateBuilder().Where(squirrel.Eq{
				"id": service.Id,
			}).SetMap(map[string]interface{}{
				"cpod_id": req.CpodId,
				"status":  model.InferStatusDeploying,
			}))
			if err != nil {
				l.Errorf("inference assigned inferId=%d cpod_id=%s err=%s", service.Id, req.CpodId, err)
				return nil, err
			}
			l.Infof("inference assigned inferId=%d cpod_id=%s", service.Id, req.CpodId)
		}
	}

	// jupyterlab
	jupyterlabList, err := JupyterlabModel.Find(l.ctx, JupyterlabModel.AllFieldsBuilder().Where(
		squirrel.Or{
			squirrel.Eq{"status": model.JupyterStatusWaitDeploy},
			squirrel.And{squirrel.Eq{"cpod_id": req.CpodId}, squirrel.NotEq{"status": model.JupyterStatusStopped}},
		},
	))
	if err != nil {
		l.Errorf("JupyterlabModel.FindAll err: %s", err)
		return nil, err
	}

	for _, jupyterlab := range jupyterlabList {
		jupyterlabResp := types.JupyterLab{}
		_ = copier.Copy(&jupyterlabResp, jupyterlab)
		jupyterlabResp.CPU = strconv.FormatInt(jupyterlab.CpuCount, 10)
		jupyterlabResp.Memory = strconv.FormatInt(jupyterlab.MemCount, 10)
		jupyterlabResp.GPU = int(jupyterlab.GpuCount)
		jupyterlabResp.GPUProduct = jupyterlab.GpuProd
		jupyterlabResp.DataVolumeSize = strconv.FormatInt(jupyterlab.DataVolumeSize, 10)
		jupyterlabResp.PretrainModels = make([]types.PretrainModels, 0)
		if jupyterlab.ModelId != "" {
			jupyterlabResp.PretrainModels = append(jupyterlabResp.PretrainModels, types.PretrainModels{
				PretrainModelId:   jupyterlab.ModelId,
				PretrainModelName: jupyterlab.ModelName,
				PretrainModelPath: jupyterlab.ModelPath,
			})
		}

		resp.JupyterlabList = append(resp.JupyterlabList, jupyterlabResp)

		// 新分配的部署更新cpod_id和status
		if jupyterlab.CpodId == "" && jupyterlab.Status == model.InferStatusWaitDeploy {
			_, err = JupyterlabModel.UpdateColsByCond(l.ctx, JupyterlabModel.UpdateBuilder().Where(squirrel.Eq{
				"id": jupyterlab.Id,
			}).SetMap(map[string]interface{}{
				"cpod_id": req.CpodId,
				"status":  model.JupyterStatusDeploying,
			}))
			if err != nil {
				l.Errorf("jupyterlab assigned id=%d cpod_id=%s err=%s", jupyterlab.Id, req.CpodId, err)
				return nil, err
			}
			l.Infof("jupyterlab assigned id=%d cpod_id=%s", jupyterlab.Id, req.CpodId)
		}
	}

	return
}
