package logic

import (
	"context"
	"database/sql"
	"net/http"
	"strconv"
	"sxwl/3k/internal/scheduler/model"
	"sxwl/3k/internal/scheduler/svc"
	"sxwl/3k/internal/scheduler/types"
	"sxwl/3k/pkg/consts"
	"time"

	"github.com/zeromicro/go-zero/rest/httpc"

	"github.com/Masterminds/squirrel"
	"github.com/zeromicro/go-zero/core/logx"
)

type JobStatus string

// 通用的任务状态定义
const (
	JobStatusCreated        JobStatus = "created"        //任务在Cpod中被创建（在K8S中被创建），pod在启动过程中
	JobStatusCreateFailed   JobStatus = "createfailed"   //任务在创建时直接失败（因为配置原因）
	JobStatusRunning        JobStatus = "running"        //Pod全部创建成功，并正常运行
	JobStatusPending        JobStatus = "pending"        //因为资源不足，在等待
	JobStatusErrorLoop      JobStatus = "crashloop"      //进入crashloop
	JobStatusModelUploaded  JobStatus = "modeluploaded"  //模型文件（训练结果）已上传
	JobStatusModelUploading JobStatus = "modeluploading" //模型文件（训练结果）正在上传
	JobStatusSucceed        JobStatus = "succeeded"      //所有工作成功完成
	JobStatusFailed         JobStatus = "failed"         //在中途以失败中止
	JobStatusUnknown        JobStatus = "unknown"        //无法获取任务状态，状态未知
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
				GpuMem:         sql.NullInt64{Int64: int64(gpu.MemSize)},
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

	currentCache := make(map[string]int64)
	for _, cache := range cacheList {
		currentCache[cache.DataId] = cache.Id
	}

	reportCache := make(map[string]types.Cache)
	for _, cache := range req.ResourceInfo.Caches {
		reportCache[cache.DataId] = cache
	}

	// update cache insert
	for id, cache := range reportCache {
		if _, exists := currentCache[id]; !exists {
			dataType, ok := cacheTypeToDBMap[cache.DataType]
			if !ok {
				l.Logger.Errorf("cpod_cache data_type not match cpod_id=%s data_type=%s", req.CPODID, cache.DataType)
				continue
			}
			_, err = CpodCacheModel.Insert(l.ctx, &model.SysCpodCache{
				CpodId:      req.CPODID,
				CpodVersion: req.ResourceInfo.CPODVersion,
				DataType:    int64(dataType),
				DataName:    cache.DataName,
				DataId:      cache.DataId,
				DataSource:  cache.DataSource,
			})
			if err != nil {
				l.Logger.Errorf("cpod_cache insert cpod_id=%s data_type=%s data_name=%s data_id=%s data_source=%s err=%s",
					req.CPODID, cache.DataType, cache.DataName, cache.DataId, cache.DataSource, err)
				return nil, err
			}
		}
		delete(currentCache, id)
	}

	// update cache delete
	for dataId, id := range currentCache {
		err = CpodCacheModel.Delete(l.ctx, id)
		if err != nil {
			l.Logger.Errorf("cpod_cache delete cpod_id=%s id=%d data_id=%s err=%s", req.CPODID, id, dataId, err)
			return nil, err
		}
	}

	resp = &types.CPODStatusResp{Message: "cpod_status success"}
	return
}

var statusToDBMap = map[string]int{
	string(JobStatusCreateFailed):  model.JobStatusWorkerFail,
	string(JobStatusFailed):        model.JobStatusWorkerFail,
	string(JobStatusSucceed):       model.JobStatusWorkerSuccess,
	string(JobStatusModelUploaded): model.JobStatusWorkerUrlSuccess,
}

var cacheTypeToDBMap = map[string]int{
	consts.CacheModel:   model.CacheModel,
	consts.CacheDataSet: model.CacheDataset,
	consts.CacheImage:   model.CacheImage,
}

var dbToCacheTypeMap = map[int]string{
	model.CacheModel:   consts.CacheModel,
	model.CacheDataset: consts.CacheDataSet,
	model.CacheImage:   consts.CacheImage,
}
