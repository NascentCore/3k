package logic

import (
	"context"
	"database/sql"
	"fmt"
	"sxwl/3k/internal/scheduler/model"
	"sxwl/3k/internal/scheduler/svc"
	"sxwl/3k/internal/scheduler/types"
	"sxwl/3k/pkg/consts"
	"sxwl/3k/pkg/orm"
	"time"

	"github.com/Masterminds/squirrel"
	"github.com/zeromicro/go-zero/core/logx"
)

type JobStatus string

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
	// FileURLModel := l.svcCtx.FileURLModel
	CpodCacheModel := l.svcCtx.CpodCacheModel
	InferModel := l.svcCtx.InferenceModel
	JupyterlabModel := l.svcCtx.JupyterlabModel

	// check banned
	_, ok := l.svcCtx.Config.BannedCpod[req.CPODID]
	if ok {
		return nil, fmt.Errorf("cpod is illegal")
	}

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
				UserId:         orm.NullString(req.UserID),
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
				"gpu_mem":         gpu.MemSize,
				"user_id":         req.UserID,
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
		status, ok := model.StatusToInt[job.JobStatus]
		if !ok {
			l.Logger.Errorf("StatusToInt not exists status=%s", job.JobStatus)
			continue
		}
		userJobs, err := UserJobModel.Find(l.ctx, UserJobModel.AllFieldsBuilder().Where(squirrel.Eq{
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

		if userJobs[0].WorkStatus == int64(status) {
			// l.Logger.Infof("user_job status same cpod_id=%s job_name=%s status=%d", req.CPODID, job.Name, status)
			continue
		}

		updateBuilder := UserJobModel.UpdateBuilder().Where(
			squirrel.Eq{
				"job_name": job.Name,
			},
		)
		setMap := map[string]interface{}{
			"update_time": sql.NullTime{
				Time:  time.Now(),
				Valid: true,
			},
			"work_status": status,
		}

		// 终态不再下发
		switch status {
		case model.StatusFailed, model.StatusSucceeded:
			setMap["obtain_status"] = model.StatusObtainNotNeedSend
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

		// 2024-06-14 目前没有回调入口，暂时关闭
		// // callback
		// if dbWorkStatus != model.JobStatusWorkerFail && dbWorkStatus != model.JobStatusWorkerUrlSuccess {
		// 	continue
		// }
		//
		// callBackReq := types.JobCallBackReq{}
		// go func(cpodID, jobName string) {
		// 	logger := logx.WithContext(context.Background())
		// 	userJob, err := UserJobModel.FindOneByQuery(context.Background(),
		// 		UserJobModel.AllFieldsBuilder().Where(
		// 			squirrel.Eq{
		// 				"cpod_id":  cpodID,
		// 				"job_name": jobName,
		// 			},
		// 		),
		// 	)
		// 	if err != nil {
		// 		logger.Errorf("user_job findOne cpod_id=%s job_name=%s err=%s", cpodID, jobName, err)
		// 		return
		// 	}
		//
		// 	if userJob.CallbackUrl.String == "" {
		// 		logger.Infof("user_job callback is empty cpod_id=%s job_name=%s", cpodID, jobName)
		// 		return
		// 	}
		//
		// 	var url string
		// 	if dbWorkStatus == model.JobStatusWorkerUrlSuccess {
		// 		fileURL, err := FileURLModel.FindOneByQuery(context.Background(),
		// 			FileURLModel.AllFieldsBuilder().Where(
		// 				squirrel.Eq{
		// 					"job_name": jobName,
		// 				},
		// 			),
		// 		)
		// 		if err != nil {
		// 			logger.Errorf("file_url findOne job_name=%s err=%s", jobName, err)
		// 			return
		// 		}
		// 		url = fileURL.FileUrl.String
		// 	}
		//
		// 	switch dbWorkStatus {
		// 	case model.JobStatusWorkerFail:
		// 		callBackReq.Status = consts.JobFail
		// 	case model.JobStatusWorkerUrlSuccess:
		// 		callBackReq.Status = consts.JobSuccess
		// 		callBackReq.URL = url
		// 		callBackReq.JobID = jobName
		// 	}
		//
		// 	callBackResp, err := httpc.Do(context.Background(), http.MethodPost, userJob.CallbackUrl.String, callBackReq)
		// 	if err != nil {
		// 		logger.Errorf("callback job_name=%s url=%s err=%s", jobName, userJob.CallbackUrl.String, err)
		// 		return
		// 	}
		// 	logger.Infof("callback job_name=%s url=%s status=%s", jobName,
		// 		userJob.CallbackUrl.String, callBackResp.Status)
		// }(req.CPODID, job.Name)
	}

	// update inference
	for _, reqInfer := range req.InferenceStatus {
		// get int status
		status, ok := model.StatusToInt[reqInfer.Status]
		if !ok {
			l.Logger.Errorf("StatusToInt not exists status=%s", reqInfer.Status)
			continue
		}

		// query the inference in db
		infer, err := InferModel.FindOneByQuery(l.ctx, InferModel.AllFieldsBuilder().Where(squirrel.Eq{
			"service_name": reqInfer.ServiceName,
		}))
		if err != nil {
			l.Logger.Errorf("infer find service_name=%s err=%s", reqInfer.ServiceName, err)
			return nil, err
		}

		// if status the same continue
		if infer.Status == int64(status) {
			continue
		}

		// if jupyterlab is deleted continue 终态不再更新
		if model.FinalStatus(infer.Status) {
			continue
		}

		updateBuilder := InferModel.UpdateBuilder().Where(squirrel.Eq{
			"service_name": infer.ServiceName,
		})

		var setMap = map[string]interface{}{}
		switch status {
		case model.StatusRunning:
			setMap = map[string]interface{}{
				"start_time": orm.NullTime(time.Now()),
				"status":     status,
				"url":        reqInfer.URL,
			}
		case model.StatusSucceeded, model.StatusFailed:
			setMap = map[string]interface{}{
				"end_time":      orm.NullTime(time.Now()),
				"status":        status,
				"obtain_status": model.StatusObtainNotNeedSend,
			}
		case model.StatusDataPreparing:
			setMap = map[string]interface{}{
				"status": status,
			}
		default:
			continue
		}

		result, err := InferModel.UpdateColsByCond(l.ctx, updateBuilder.SetMap(setMap))
		if err != nil {
			l.Logger.Errorf("inference update cpod_id=%s service_name=%s err=%s", req.CPODID, infer.ServiceName, err)
			return nil, err
		}
		rows, err := result.RowsAffected()
		if err != nil {
			l.Logger.Errorf("inference update cpod_id=%s service_name=%s RowsAffected err=%s", req.CPODID, infer.ServiceName, err)
			return nil, err
		}
		if rows != 1 {
			l.Logger.Errorf("inference update rows=%d cpod_id=%s service_name=%s err=%s", rows, req.CPODID, infer.ServiceName, err)
			continue // no update
		}
	}

	// update jupyterlab
	for _, reqJupyter := range req.JupyterlabStatus {
		// get int status
		status, ok := model.StatusToInt[reqJupyter.Status]
		if !ok {
			l.Logger.Errorf("StatusToInt not exists status=%s", reqJupyter.Status)
			continue
		}

		// query the jupyterlab in db
		jupyter, err := JupyterlabModel.FindOneByQuery(l.ctx, JupyterlabModel.AllFieldsBuilder().Where(squirrel.Eq{
			"job_name": reqJupyter.JobName,
		}))
		if err != nil {
			l.Logger.Errorf("jupyter find job_name=%s err=%s", reqJupyter.JobName, err)
			return nil, err
		}

		// if status the same continue
		if jupyter.Status == int64(status) {
			continue
		}

		// if jupyterlab is deleted continue 终态不再更新
		if model.FinalStatus(jupyter.Status) {
			continue
		}

		updateBuilder := JupyterlabModel.UpdateBuilder().Where(squirrel.Eq{
			"job_name": reqJupyter.JobName,
		})

		var setMap = map[string]interface{}{}
		switch status {
		case model.StatusRunning:
			setMap = map[string]interface{}{
				"start_time": orm.NullTime(time.Now()),
				"status":     status,
				"url":        reqJupyter.URL,
			}
		case model.StatusSucceeded, model.StatusFailed:
			setMap = map[string]interface{}{
				"end_time": orm.NullTime(time.Now()),
				"status":   status,
			}
		case model.StatusDataPreparing, model.StatusPaused, model.StatusPausing:
			setMap = map[string]interface{}{
				"status": status,
			}
		default:
			continue
		}

		result, err := JupyterlabModel.UpdateColsByCond(l.ctx, updateBuilder.SetMap(setMap))
		if err != nil {
			l.Logger.Errorf("reqJupyter update cpod_id=%s name=%s err=%s", req.CPODID, reqJupyter.JobName, err)
			return nil, err
		}
		rows, err := result.RowsAffected()
		if err != nil {
			l.Logger.Errorf("reqJupyter update cpod_id=%s name=%s RowsAffected err=%s", req.CPODID, reqJupyter.JobName, err)
			return nil, err
		}
		if rows != 1 {
			l.Logger.Errorf("reqJupyter update rows=%d cpod_id=%s name=%s err=%s", rows, req.CPODID, reqJupyter.JobName, err)
			continue // no update
		}
	}

	// update cache
	cacheList, err := CpodCacheModel.Find(l.ctx, CpodCacheModel.AllFieldsBuilder().Where(squirrel.And{
		squirrel.Eq{"cpod_id": req.CPODID},
	}))
	if err != nil {
		l.Logger.Errorf("cpod_cache find cpod_id=%s err=%s", req.CPODID, err)
		return nil, err
	}

	currentCache := make(map[string]*model.SysCpodCache)
	for _, cache := range cacheList {
		currentCache[cache.DataId] = cache
	}

	reportCache := make(map[string]types.Cache)
	for _, cache := range req.ResourceInfo.Caches {
		reportCache[cache.DataId] = cache
	}

	// update cache insert
	for id, cache := range reportCache {
		if _, exists := currentCache[id]; !exists {
			// 不存在就插入记录
			dataType, ok := cacheTypeToDBMap[cache.DataType]
			if !ok {
				l.Logger.Errorf("cpod_cache data_type not match cpod_id=%s data_type=%s", req.CPODID, cache.DataType)
				continue
			}
			sysCpodCache := model.SysCpodCache{
				CpodId:            req.CPODID,
				CpodVersion:       req.ResourceInfo.CPODVersion,
				DataType:          int64(dataType),
				DataName:          cache.DataName,
				DataId:            cache.DataId,
				DataSize:          cache.DataSize,
				DataSource:        cache.DataSource,
				Template:          cache.Template,
				FinetuneGpuCount:  cache.FinetuneGPUCount,
				InferenceGpuCount: cache.InferenceGPUCount,
			}
			if cache.IsPublic {
				sysCpodCache.Public = model.CachePublic
			} else {
				sysCpodCache.Public = model.CachePrivate
				sysCpodCache.NewUserId = orm.NullString(cache.UserID)
			}
			_, err = CpodCacheModel.Insert(l.ctx, &sysCpodCache)
			if err != nil {
				l.Logger.Errorf("cpod_cache insert cpod_id=%s data_type=%s data_name=%s data_id=%s data_source=%s err=%s",
					req.CPODID, cache.DataType, cache.DataName, cache.DataId, cache.DataSource, err)
				return nil, err
			}
		} else {
			// 存在的检查下template和size是否变更
			isPublic := true
			public := model.CachePublic
			if currentCache[id].Public == model.CachePrivate {
				isPublic = false
				public = model.CachePrivate
			}
			if currentCache[id].Template != cache.Template ||
				currentCache[id].DataSize != cache.DataSize ||
				currentCache[id].FinetuneGpuCount != cache.FinetuneGPUCount ||
				currentCache[id].InferenceGpuCount != cache.InferenceGPUCount ||
				isPublic != cache.IsPublic ||
				currentCache[id].NewUserId.String != cache.UserID {
				_, err = CpodCacheModel.UpdateColsByCond(l.ctx, CpodCacheModel.UpdateBuilder().Where(squirrel.Eq{
					"id": currentCache[id].Id,
				}).SetMap(map[string]interface{}{
					"data_size":           cache.DataSize,
					"template":            cache.Template,
					"finetune_gpu_count":  cache.FinetuneGPUCount,
					"inference_gpu_count": cache.InferenceGPUCount,
					"public":              public,
					"new_user_id":         cache.UserID,
				}))
				if err != nil {
					l.Logger.Errorf("cpod_cache update cpod_id=%s data_type=%s data_name=%s data_id=%s data_source=%s err=%s",
						req.CPODID, cache.DataType, cache.DataName, cache.DataId, cache.DataSource, err)
					return nil, err
				}
			}
		}
		delete(currentCache, id)
	}

	// update cache delete
	for dataId, cache := range currentCache {
		err = CpodCacheModel.Delete(l.ctx, cache.Id)
		if err != nil {
			l.Logger.Errorf("cpod_cache delete cpod_id=%s id=%d data_id=%s err=%s", req.CPODID, cache.Id, dataId, err)
			return nil, err
		}
	}

	resp = &types.CPODStatusResp{Message: "cpod_status success"}
	return
}

var cacheTypeToDBMap = map[string]int{
	consts.CacheModel:   model.CacheModel,
	consts.CacheDataSet: model.CacheDataset,
	consts.CacheImage:   model.CacheImage,
}

// var dbToCacheTypeMap = map[int]string{
// 	model.CacheModel:   consts.CacheModel,
// 	model.CacheDataset: consts.CacheDataSet,
// 	model.CacheImage:   consts.CacheImage,
// }
