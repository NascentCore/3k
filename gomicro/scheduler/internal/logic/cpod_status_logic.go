package logic

import (
	"context"
	"database/sql"
	"net/http"
	"strconv"
	"sxwl/3k/gomicro/pkg/consts"
	"sxwl/3k/gomicro/scheduler/internal/cpod_cache"
	"sxwl/3k/gomicro/scheduler/internal/model"
	"sxwl/3k/gomicro/scheduler/internal/svc"
	"sxwl/3k/gomicro/scheduler/internal/types"
	"sxwl/3k/pkg/job/state"
	"time"

	"github.com/zeromicro/go-zero/rest/httpc"

	"github.com/Masterminds/squirrel"
	"github.com/zeromicro/go-zero/core/logx"
)

type CpodStatusLogic struct {
	logx.Logger
	ctx    context.Context
	svcCtx *svc.ServiceContext
}

func NewCpodStatusLogic(ctx context.Context, svcCtx *svc.ServiceContext) *CpodStatusLogic {
	return &CpodStatusLogic{
		Logger: logx.WithContext(ctx),
		ctx:    ctx,
		svcCtx: svcCtx,
	}
}

func (l *CpodStatusLogic) CpodStatus(req *types.CPODStatusReq) (resp *types.CPODStatusResp, err error) {
	CpodMainModel := l.svcCtx.CpodMainModel
	UserJobModel := l.svcCtx.UserJobModel
	FileURLModel := l.svcCtx.FileURLModel
	CpodCacheModel := l.svcCtx.CpodCacheModel

	// update sys_main_job or insert
	for _, gpu := range req.ResourceInfo.GPUSummaries {
		cpodMains, err := CpodMainModel.Find(l.ctx, CpodMainModel.AllFieldsBuilder().Where(squirrel.Eq{
			"cpod_id":  req.CPODID,
			"gpu_prod": gpu.Prod,
		}))
		if err != nil {
			l.Logger.Errorf("cpod_main find cpod_id=%s gpu_prod=%s err=%s", req.CPODID, gpu.Prod, err)
			return nil, err
		}

		if len(cpodMains) == 0 { // new record insert
			cpodMain := model.SysCpodMain{
				CpodId:         sql.NullString{String: req.CPODID, Valid: true},
				CpodVersion:    sql.NullString{String: req.ResourceInfo.CPODVersion, Valid: true},
				GpuVendor:      sql.NullString{String: gpu.Vendor, Valid: true},
				GpuProd:        sql.NullString{String: gpu.Prod, Valid: true},
				GpuTotal:       sql.NullInt64{Int64: int64(gpu.Total), Valid: true},
				GpuAllocatable: sql.NullInt64{Int64: int64(gpu.Allocatable), Valid: true},
				CreateTime:     sql.NullTime{Time: time.Now(), Valid: true},
				UpdateTime:     sql.NullTime{Time: time.Now(), Valid: true},
				UserId:         sql.NullString{String: strconv.FormatInt(req.UserID, 10), Valid: true},
			}
			_, err := CpodMainModel.Insert(l.ctx, &cpodMain)
			if err != nil {
				l.Logger.Errorf("cpod_main insert err=%s", err)
				return nil, err
			}
			l.Logger.Infof("cpod_main insert cpod_main cpod_id=%s gpu_prod=%s", req.CPODID, gpu.Prod)
		} else if len(cpodMains) == 1 { // already exists update
			_, err := CpodMainModel.UpdateColsByCond(l.ctx, CpodMainModel.UpdateBuilder().SetMap(map[string]interface{}{
				"gpu_total":       gpu.Total,
				"gpu_allocatable": gpu.Allocatable,
				"update_time": sql.NullTime{
					Time:  time.Now(),
					Valid: true,
				},
			}).Where(squirrel.Eq{
				"cpod_id":  req.CPODID,
				"gpu_prod": gpu.Prod,
			}))
			if err != nil {
				l.Logger.Errorf("cpod_main UpdateColsByCond cpod_id=%s gpu_prod=%s err=%s", req.CPODID, gpu.Prod, err)
				return nil, err
			}

			l.Logger.Infof("cpod_main UpdateColsByCond cpod_id=%s gpu_prod=%s total=%d alloc=%d", req.CPODID, gpu.Prod,
				gpu.Total, gpu.Allocatable)
		} else { // multi duplicate records log
			l.Logger.Infof("cpod_main multi duplicate rows cpod_id=%s gpu_prod=%s", req.CPODID, gpu.Prod)
		}
	}

	// update sys_user_job
	for _, job := range req.JobStatus {
		dbWorkStatus, ok := statusToDBMap[job.JobStatus]
		if !ok {
			l.Logger.Errorf("statusToDBMap not exists status=%s", job.JobStatus)
			continue
		}
		userJobs, err := UserJobModel.Find(l.ctx, UserJobModel.AllFieldsBuilder().Where(squirrel.Eq{
			"cpod_id":  req.CPODID,
			"job_name": job.Name,
		}))
		if err != nil {
			l.Logger.Errorf("user_job find cpod_id=%s job_name=%s err=%s", req.CPODID, job.Name, err)
			return nil, err
		}
		if len(userJobs) != 1 {
			l.Logger.Errorf("user_job find cpod_id=%s job_name=%s len=%d", req.CPODID, job.Name, len(userJobs))
			// 这条任务数据记录异常
			continue
		}

		if userJobs[0].WorkStatus == int64(dbWorkStatus) {
			l.Logger.Infof("user_job status same cpod_id=%s job_name=%s status=%d", req.CPODID, job.Name, dbWorkStatus)
			continue
		}

		updateBuilder := UserJobModel.UpdateBuilder().Where(
			squirrel.Eq{
				"cpod_id":  req.CPODID,
				"job_name": job.Name,
			},
		)
		setMap := map[string]interface{}{
			"obtain_status": model.JobStatusObtainNotNeedSend,
			"update_time": sql.NullTime{
				Time:  time.Now(),
				Valid: true,
			},
			"work_status": dbWorkStatus,
		}

		result, err := UserJobModel.UpdateColsByCond(l.ctx, updateBuilder.SetMap(setMap))
		if err != nil {
			l.Logger.Errorf("user_job update cpod_id=%s job_name=%s err=%s", req.CPODID, job.Name, err)
			return nil, err
		}

		rows, err := result.RowsAffected()
		if err != nil {
			l.Logger.Errorf("user_job update cpod_id=%s job_name=%s RowsAffected err=%s", req.CPODID, job.Name, err)
			return nil, err
		}
		if rows != 1 {
			l.Logger.Errorf("user_job update rows=%d cpod_id=%s job_name=%s err=%s", rows, req.CPODID, job.Name, err)
			continue // no update
		}

		// callback
		if dbWorkStatus != model.JobStatusWorkerFail && dbWorkStatus != model.JobStatusWorkerUrlSuccess {
			continue
		}

		callBackReq := types.JobCallBackReq{}
		go func(cpodID, jobName string) {
			logger := logx.WithContext(context.Background())
			userJob, err := UserJobModel.FindOneByQuery(context.Background(),
				UserJobModel.AllFieldsBuilder().Where(
					squirrel.Eq{
						"cpod_id":  cpodID,
						"job_name": jobName,
					},
				),
			)
			if err != nil {
				logger.Errorf("user_job findOne cpod_id=%s job_name=%s err=%s", cpodID, jobName, err)
				return
			}

			if userJob.CallbackUrl.String == "" {
				logger.Infof("user_job callback is empty cpod_id=%s job_name=%s", cpodID, jobName)
				return
			}

			var url string
			if dbWorkStatus == model.JobStatusWorkerUrlSuccess {
				fileURL, err := FileURLModel.FindOneByQuery(context.Background(),
					FileURLModel.AllFieldsBuilder().Where(
						squirrel.Eq{
							"job_name": jobName,
						},
					),
				)
				if err != nil {
					logger.Errorf("file_url findOne job_name=%s err=%s", jobName, err)
					return
				}
				url = fileURL.FileUrl.String
			}

			switch dbWorkStatus {
			case model.JobStatusWorkerFail:
				callBackReq.Status = consts.JobFail
			case model.JobStatusWorkerUrlSuccess:
				callBackReq.Status = consts.JobSuccess
				callBackReq.URL = url
				callBackReq.JobID = jobName
			}

			callBackResp, err := httpc.Do(context.Background(), http.MethodPost, userJob.CallbackUrl.String, callBackReq)
			if err != nil {
				logger.Errorf("callback job_name=%s url=%s err=%s", jobName, userJob.CallbackUrl.String, err)
				return
			}
			logger.Infof("callback job_name=%s url=%s status=%s", jobName,
				userJob.CallbackUrl.String, callBackResp.Status)
		}(req.CPODID, job.Name)
	}

	// update cache
	cacheList, err := CpodCacheModel.Find(l.ctx, CpodCacheModel.AllFieldsBuilder().Where(squirrel.And{
		squirrel.Eq{"cpod_id": req.CPODID},
	}))
	if err != nil {
		l.Logger.Errorf("cpod_cache find cpod_id=%s err=%s", req.CPODID, err)
		return nil, err
	}

	currentCache := make(map[string]bool)
	for _, cache := range cacheList {
		currentCache[cpod_cache.Encode(cache.DataType, cache.DataId)] = true
	}

	reportCache := make(map[string]bool)
	for _, cachedModel := range req.ResourceInfo.CachedModels {
		reportCache[cpod_cache.Encode(model.CacheModel, cachedModel)] = true
	}
	for _, cachedDataset := range req.ResourceInfo.CachedDatasets {
		reportCache[cpod_cache.Encode(model.CacheDataset, cachedDataset)] = true
	}
	for _, cachedImage := range req.ResourceInfo.CachedImages {
		reportCache[cpod_cache.Encode(model.CacheImage, cachedImage)] = true
	}

	// update cache insert
	for rc := range reportCache {
		if _, exists := currentCache[rc]; !exists {
			dataType, dataId, err := cpod_cache.Decode(rc)
			if err != nil {
				l.Logger.Errorf("cpod_cache decode encoded=%s err=%s", rc, err)
				return nil, err
			}
			_, err = CpodCacheModel.Insert(l.ctx, &model.SysCpodCache{
				CpodId:      req.CPODID,
				CpodVersion: req.ResourceInfo.CPODVersion,
				DataType:    dataType,
				DataId:      dataId,
			})
			if err != nil {
				l.Logger.Errorf("cpod_cache insert cpod_id=%s data_type=%d data_id=%s err=%s", req.CPODID,
					dataType, dataId, err)
				return nil, err
			}
		}
		delete(currentCache, rc)
	}

	// update cache delete
	for rc := range currentCache {
		dataType, dataId, err := cpod_cache.Decode(rc)
		if err != nil {
			l.Logger.Errorf("cpod_cache decode encoded=%s err=%s", rc, err)
			return nil, err
		}
		err = CpodCacheModel.DeleteByCond(l.ctx, CpodCacheModel.DeleteBuilder().Where(squirrel.Eq{
			"cpod_id":   req.CPODID,
			"data_type": dataType,
			"data_id":   dataId,
		}))
		if err != nil {
			l.Logger.Errorf("cpod_cache delete cpod_id=%s data_type=%d data_id=%s err=%s", req.CPODID,
				dataType, dataId, err)
			return nil, err
		}
	}

	resp = &types.CPODStatusResp{Message: "cpod_status success"}
	return
}

var statusToDBMap = map[string]int{
	string(state.JobStatusCreateFailed):  model.JobStatusWorkerFail,
	string(state.JobStatusFailed):        model.JobStatusWorkerFail,
	string(state.JobStatusSucceed):       model.JobStatusWorkerSuccess,
	string(state.JobStatusModelUploaded): model.JobStatusWorkerUrlSuccess,
}
