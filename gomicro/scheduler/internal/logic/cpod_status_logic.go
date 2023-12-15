package logic

import (
	"context"
	"database/sql"
	"net/http"
	"strconv"
	"sxwl/3k/gomicro/pkg/consts"
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

	// update sys_main_job or insert
	for _, gpu := range req.ResourceInfo.GPUSummaries {
		result, err := CpodMainModel.UpdateColsByCond(l.ctx, CpodMainModel.UpdateBuilder().SetMap(map[string]interface{}{
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
		l.Logger.Infof("cpod_main UpdateColsByCond cpod_id=%s gpu_prod=%s err=%s", req.CPODID, gpu.Prod, err)

		rows, err := result.RowsAffected()
		if err != nil {
			l.Logger.Errorf("cpod_main UpdateColsByCond rows cpod_id=%s gpu_prod=%s err=%s", req.CPODID, gpu.Prod, err)
			return nil, err
		}

		if rows == 0 { // 没有更新，就插入新记录
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
			l.Logger.Infof("cpod_main insert cpod_main cpod_id=%s gpu_prod=%s err=%s", req.CPODID, gpu.Prod, err)
		}
	}

	// update sys_user_job
	for _, job := range req.JobStatus {
		dbWorkStatus, ok := statusToDBMap[job.JobStatus]
		if !ok {
			continue
		}
		updateBuilder := UserJobModel.UpdateBuilder().Where(
			squirrel.And{
				squirrel.Eq{
					"cpod_id":  req.CPODID,
					"job_name": job.Name,
				}, squirrel.NotEq{
					"work_status": dbWorkStatus,
				},
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
			l.Logger.Errorf("user_job update job_name=%s err=%s", job.Name, err)
			return nil, err
		}
		if rows, err := result.RowsAffected(); err != nil {
			l.Logger.Errorf("user_job update job_name=%s RowsAffected err=%s", job.Name, err)
			return nil, err
		} else if rows != 1 {
			continue // no update
		}

		// fail then callback
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
				callBackReq.Status = consts.CallBackJobFail
			case model.JobStatusWorkerUrlSuccess:
				callBackReq.Status = consts.CallBackJobSuccess
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

	resp = &types.CPODStatusResp{Message: "cpod_status success"}
	return
}

var statusToDBMap = map[string]int{
	string(state.JobStatusCreateFailed):  model.JobStatusWorkerFail,
	string(state.JobStatusFailed):        model.JobStatusWorkerFail,
	string(state.JobStatusSucceed):       model.JobStatusWorkerSuccess,
	string(state.JobStatusModelUploaded): model.JobStatusWorkerUrlSuccess,
}
