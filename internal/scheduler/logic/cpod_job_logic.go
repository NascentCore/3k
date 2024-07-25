package logic

import (
	"context"
	"database/sql"
	"encoding/json"
	"fmt"
	"strconv"
	"sxwl/3k/internal/scheduler/model"
	"sxwl/3k/pkg/storage"
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
	CpodNodeModel := l.svcCtx.CpodNodeModel
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

	nodes, err := CpodNodeModel.Find(l.ctx, CpodNodeModel.AllFieldsBuilder().Where(squirrel.And{
		squirrel.Eq{
			"cpod_id": req.CpodId,
		},
		squirrel.Expr("updated_at > NOW() - INTERVAL 30 MINUTE"),
	}))
	if err != nil {
		l.Logger.Errorf("CpodNodeModel Find cpod_id=%s err=%s", req.CpodId, err)
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

	activeJobs := make([]*model.SysUserJob, 0)         // 已经在运行中的任务
	assignedJobs := make([]*model.SysUserJob, 0)       // 本次被分配的任务
	assignedNodes := make(map[*model.SysCpodNode]bool) // 本次被分配任务的node

	// finetune and cpodjob
	for _, job := range jobs {
		if job.CpodId.String == "" {
			for _, node := range nodes {
				// CPU: 1
				// Memory: 无限制
				if node.CpuAllocatable < 1 {
					continue
				}
				if node.GpuProd == job.GpuType.String && node.GpuAllocatable >= job.GpuNumber.Int64 {
					assignedJobs = append(assignedJobs, job)
					node.GpuAllocatable -= job.GpuNumber.Int64
					node.CpuAllocatable -= 1
					assignedNodes[node] = true
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
		if service.Metadata.Valid {
			_ = json.Unmarshal([]byte(service.Metadata.String), &serviceResp)
		} else {
			continue // 老任务没有metadata，直接忽略掉
		}
		statusDesc, ok := model.StatusToStr[service.Status]
		if ok {
			serviceResp.Status = statusDesc
		}
		serviceResp.CpodId = service.CpodId

		switch service.Status {
		case model.StatusNotAssigned:
			for _, node := range nodes {
				// CPU: 4
				// Memory: 50G
				if node.CpuAllocatable < 4 || node.MemAllocatable < storage.GBToBytes(50) {
					continue
				}
				if node.GpuProd == service.GpuType.String && node.GpuAllocatable >= service.GpuNumber.Int64 {
					node.GpuAllocatable -= service.GpuNumber.Int64
					node.CpuAllocatable -= 4
					node.MemAllocatable -= storage.GBToBytes(50)
					assignedNodes[node] = true

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
		jupyterlabResp.Memory = storage.BytesToHumanReadable(jupyterlab.MemCount)
		jupyterlabResp.GPUCount = int(jupyterlab.GpuCount)
		jupyterlabResp.GPUProduct = jupyterlab.GpuProd
		jupyterlabResp.DataVolumeSize = storage.BytesToHumanReadable(jupyterlab.DataVolumeSize)
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
				for _, node := range nodes {
					if node.CpuAllocatable < jupyterlab.CpuCount || node.MemAllocatable < jupyterlab.MemCount {
						continue
					}
					if node.GpuProd == jupyterlab.GpuProd && node.GpuAllocatable >= jupyterlab.GpuCount {
						node.GpuAllocatable -= jupyterlab.GpuCount
						node.CpuAllocatable -= jupyterlab.CpuCount
						node.MemAllocatable -= jupyterlab.MemCount
						assignedNodes[node] = true
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
	for node := range assignedNodes {
		_, err = CpodNodeModel.UpdateColsByCond(l.ctx, CpodNodeModel.UpdateBuilder().Where(squirrel.Eq{
			"id": node.Id,
		}).SetMap(map[string]interface{}{
			"gpu_allocatable": node.GpuAllocatable,
			"cpu_allocatable": node.CpuAllocatable,
			"mem_allocatable": node.MemAllocatable,
		}))
		if err != nil {
			l.Logger.Errorf("CpodNodeModel assigned id=%d gpu_allocatable=%d cpu_allocatable=%d mem_allocatable=%d err=%s",
				node.Id, node.GpuAllocatable, node.CpuAllocatable, node.MemAllocatable, err)
			return nil, err
		}
		l.Logger.Infof("CpodNodeModel assigned id=%d gpu_allocatable=%d cpu_allocatable=%d mem_allocatable=%d",
			node.Id, node.GpuAllocatable, node.CpuAllocatable, node.MemAllocatable)
	}

	return
}
