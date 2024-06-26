package logic

import (
	"context"
	"database/sql"
	"encoding/json"
	"fmt"
	"strconv"
	"sxwl/3k/internal/scheduler/model"
	"sxwl/3k/pkg/utils/fs"
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

	gpus, err := CpodMainModel.Find(l.ctx, CpodMainModel.AllFieldsBuilder().Where(squirrel.And{
		squirrel.Eq{
			"cpod_id": req.CpodId,
		},
		squirrel.Expr("update_time > NOW() - INTERVAL 30 MINUTE"),
	}))
	if err != nil {
		l.Logger.Errorf("cpod_main find cpod_id=%s err=%s", req.CpodId, err)
		return nil, err
	}

	jobs, err := UserJobModel.Find(l.ctx, UserJobModel.AllFieldsBuilder().Where(squirrel.Eq{
		"obtain_status": model.StatusObtainNeedSend,
		"deleted":       0,
	}))
	if err != nil {
		l.Logger.Errorf("user_job find obtain_status=%d deleted=0 err=%s", model.StatusObtainNeedSend, err)
		return nil, err
	}

	activeJobs := make([]*model.SysUserJob, 0)        // 已经在运行中的任务
	assignedJobs := make([]*model.SysUserJob, 0)      // 本次被分配的任务
	assignedGPUs := make(map[*model.SysCpodMain]bool) // 本次被分配任务的gpu

	for _, job := range jobs {
		if job.CpodId.String == "" {
			for _, gpu := range gpus {
				if gpu.GpuProd == job.GpuType && gpu.GpuAllocatable.Int64 >= job.GpuNumber.Int64 {
					assignedJobs = append(assignedJobs, job)
					gpu.GpuAllocatable.Int64 -= job.GpuNumber.Int64
					assignedGPUs[gpu] = true
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

	// inference services
	services, err := InferenceModel.FindAll(l.ctx, InferenceModel.AllFieldsBuilder().Where(
		squirrel.Or{
			squirrel.Eq{"status": model.StatusNotAssigned},
			squirrel.And{squirrel.Eq{"cpod_id": req.CpodId}, squirrel.NotEq{"status": model.StatusDeleted}},
		},
	), "")
	if err != nil {
		l.Errorf("InferenceModel.FindAll err: %s", err)
		return nil, err
	}

	for _, service := range services {
		serviceResp := types.InferenceService{}
		_ = copier.Copy(&serviceResp, service)
		statusDesc, ok := model.StatusToStr[service.Status]
		if ok {
			serviceResp.Status = statusDesc
		}
		serviceResp.Template = service.Template.String
		serviceResp.UserId = service.NewUserId
		if service.ModelPublic.Int64 == model.CachePrivate {
			serviceResp.ModelIsPublic = false
		} else {
			serviceResp.ModelIsPublic = true
		}

		switch service.Status {
		case model.StatusNotAssigned:
			for _, gpu := range gpus {
				if gpu.GpuProd.String == service.GpuType.String && gpu.GpuAllocatable.Int64 >= service.GpuNumber.Int64 {
					gpu.GpuAllocatable.Int64 -= service.GpuNumber.Int64
					assignedGPUs[gpu] = true

					// new assigned
					_, err = InferenceModel.UpdateColsByCond(l.ctx, InferenceModel.UpdateBuilder().Where(squirrel.Eq{
						"id": service.Id,
					}).SetMap(map[string]interface{}{
						"cpod_id": req.CpodId,
						"status":  model.StatusAssigned,
					}))
					if err != nil {
						l.Errorf("inference assigned inferId=%d cpod_id=%s err=%s", service.Id, req.CpodId, err)
						return nil, err
					}
					l.Infof("inference assigned inferId=%d cpod_id=%s", service.Id, req.CpodId)

					resp.InferenceServiceList = append(resp.InferenceServiceList, serviceResp)
					break
				}
			}
		default:
			resp.InferenceServiceList = append(resp.InferenceServiceList, serviceResp)
		}
	}

	// jupyterlab
	jupyterlabList, err := JupyterlabModel.Find(l.ctx, JupyterlabModel.AllFieldsBuilder().Where(
		squirrel.Or{
			squirrel.Eq{"status": model.StatusNotAssigned},
			squirrel.And{squirrel.Eq{"cpod_id": req.CpodId}, squirrel.NotEq{"status": model.StatusDeleted}},
		},
	))
	if err != nil {
		l.Errorf("JupyterlabModel.FindAll err: %s", err)
		return nil, err
	}

	for _, jupyterlab := range jupyterlabList {
		if model.FinalStatus(jupyterlab.Status) {
			continue
		}
		jupyterlabResp := types.JupyterLab{}
		_ = copier.Copy(&jupyterlabResp, jupyterlab)
		jupyterlabResp.CPUCount = strconv.FormatInt(jupyterlab.CpuCount, 10)
		jupyterlabResp.Memory = fs.BytesToMi(jupyterlab.MemCount)
		jupyterlabResp.GPUCount = int(jupyterlab.GpuCount)
		jupyterlabResp.GPUProduct = jupyterlab.GpuProd
		jupyterlabResp.DataVolumeSize = fs.BytesToMi(jupyterlab.DataVolumeSize)
		err = json.Unmarshal([]byte(jupyterlab.Resource), &jupyterlabResp.Resource)
		if err != nil {
			l.Errorf("json unmarshal jupyterlab: %d err: %s", jupyterlab.Id, err)
			// return nil, ErrSystem
		}
		jupyterlabResp.UserID = jupyterlab.NewUserId
		jupyterlabResp.Replicas = int(jupyterlab.Replicas)

		switch jupyterlab.Status {
		case model.StatusNotAssigned:
			assigned := false
			if jupyterlab.GpuProd == "" {
				assigned = true
			} else {
				for _, gpu := range gpus {
					if gpu.GpuProd.String == jupyterlab.GpuProd && gpu.GpuAllocatable.Int64 >= jupyterlab.GpuCount {
						gpu.GpuAllocatable.Int64 -= jupyterlab.GpuCount
						assignedGPUs[gpu] = true
						assigned = true
						break
					}
				}
			}

			// new assigned
			if assigned {
				_, err = JupyterlabModel.UpdateColsByCond(l.ctx, JupyterlabModel.UpdateBuilder().Where(squirrel.Eq{
					"id": jupyterlab.Id,
				}).SetMap(map[string]interface{}{
					"cpod_id": req.CpodId,
					"status":  model.StatusAssigned,
				}))
				if err != nil {
					l.Errorf("jupyterlab assigned jupyterId=%d cpod_id=%s err=%s", jupyterlab.Id, req.CpodId, err)
					return nil, err
				}
				l.Infof("jupyterlab assigned jupyterId=%d cpod_id=%s", jupyterlab.Id, req.CpodId)

				resp.JupyterlabList = append(resp.JupyterlabList, jupyterlabResp)
			}
		default:
			resp.JupyterlabList = append(resp.JupyterlabList, jupyterlabResp)
		}
	}

	// update GPU usage
	for gpu := range assignedGPUs {
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

	return
}
