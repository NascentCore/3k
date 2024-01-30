package logic

import (
	"context"
	"database/sql"
	"encoding/json"
	"sxwl/3k/internal/scheduler/model"
	"time"

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

	return
}
