package logic

import (
	"context"
	"sxwl/3k/pkg/math"
	"time"

	"sxwl/3k/internal/scheduler/svc"
	"sxwl/3k/internal/scheduler/types"

	"github.com/Masterminds/squirrel"
	"github.com/zeromicro/go-zero/core/logx"
)

type BillingTasksLogic struct {
	logx.Logger
	ctx    context.Context
	svcCtx *svc.ServiceContext
}

func NewBillingTasksLogic(ctx context.Context, svcCtx *svc.ServiceContext) *BillingTasksLogic {
	return &BillingTasksLogic{
		Logger: logx.WithContext(ctx),
		ctx:    ctx,
		svcCtx: svcCtx,
	}
}

func (l *BillingTasksLogic) BillingTasks(req *types.BillingTasksReq) (resp *types.BillingTasksResp, err error) {
	UserModel := l.svcCtx.UserModel
	BillingModel := l.svcCtx.UserBillingModel
	JobModel := l.svcCtx.UserJobModel
	InferModel := l.svcCtx.InferenceModel
	JupyterModel := l.svcCtx.JupyterlabModel

	isAdmin, err := UserModel.IsAdmin(l.ctx, req.SxUserID)
	if err != nil {
		l.Errorf("UserModel.IsAdmin userID=%s err=%s", req.SxUserID, err)
		return nil, ErrNotAdmin
	}
	if !isAdmin && req.SxUserID != req.UserID {
		l.Infof("UserModel.IsAdmin userID=%s is not admin", req.SxUserID)
		return nil, ErrNotAdmin
	}

	// 默认只查询最近一个月内的消费记录
	jobIDList := make([]string, 0)
	trainJobs, err := JobModel.Find(l.ctx, JobModel.AllFieldsBuilder().Where(squirrel.And{
		squirrel.Eq{
			"new_user_id": req.UserID,
		},
		squirrel.Expr("create_time >= DATE_SUB(NOW(), INTERVAL 1 MONTH)"),
	}).OrderBy("job_id DESC"))
	if err != nil {
		l.Errorf("trainjobs find user_id=%s err=%s", req.UserID, err)
		return nil, ErrDBFind
	}
	for _, job := range trainJobs {
		jobIDList = append(jobIDList, job.JobName.String)
	}

	inferJobs, err := InferModel.FindAll(l.ctx, InferModel.AllFieldsBuilder().Where(squirrel.And{
		squirrel.Eq{
			"new_user_id": req.UserID,
		},
		squirrel.Expr("created_at >= DATE_SUB(NOW(), INTERVAL 1 MONTH)"),
	}), "")
	if err != nil {
		l.Errorf("inferJobs find user_id=%s err=%s", req.UserID, err)
		return nil, ErrDBFind
	}
	for _, job := range inferJobs {
		jobIDList = append(jobIDList, job.ServiceName)
	}

	jupyterJobs, err := JupyterModel.Find(l.ctx, JupyterModel.AllFieldsBuilder().Where(squirrel.And{
		squirrel.Eq{
			"new_user_id": req.UserID,
		},
		squirrel.Expr("created_at >= DATE_SUB(NOW(), INTERVAL 1 MONTH)"),
	}).OrderBy("id DESC"))
	if err != nil {
		l.Errorf("jupyterJobs find user_id=%s err=%s", req.UserID, err)
		return nil, ErrDBFind
	}
	for _, job := range jupyterJobs {
		jobIDList = append(jobIDList, job.JobName)
	}

	if req.Page <= 0 {
		req.Page = 1
	}
	if req.PageSize <= 0 || req.PageSize > 100 {
		req.PageSize = 10
	}
	offset := (req.Page - 1) * req.PageSize

	// Ensure offset and offset + req.PageSize do not exceed the bounds of jobIDList
	if offset > int64(len(jobIDList)) {
		jobIDList = []string{}
	} else if offset+req.PageSize > int64(len(jobIDList)) {
		jobIDList = jobIDList[offset:]
	} else {
		jobIDList = jobIDList[offset : offset+req.PageSize]
	}

	billings, err := BillingModel.Find(l.ctx, BillingModel.AllFieldsBuilder().Where(squirrel.Eq{
		"job_id": jobIDList,
	}).OrderBy("id ASC"))
	if err != nil {
		l.Errorf("BillingModel find user_id=%s err=%s", req.UserID, err)
		return nil, ErrDBFind
	}

	jobRespMap := make(map[string]*types.TaskBilling) // job_id: amount
	for _, billing := range billings {
		_, ok := jobRespMap[billing.JobId]
		if !ok {
			jobRespMap[billing.JobId] = &types.TaskBilling{
				UserId:    billing.NewUserId,
				JobId:     billing.JobId,
				JobType:   billing.JobType,
				Amount:    math.Round(billing.Amount, 2),
				StartTime: billing.BillingTime.Format(time.DateTime),
				EndTime:   "",
			}
		} else {
			jobRespMap[billing.JobId].Amount = math.Round(jobRespMap[billing.JobId].Amount+billing.Amount, 2)
			jobRespMap[billing.JobId].EndTime = billing.BillingTime.Format(time.DateTime)
		}
	}

	resp = &types.BillingTasksResp{Data: make([]types.TaskBilling, 0), Total: int64(len(jobRespMap))}
	for _, jobID := range jobIDList {
		jobResp, ok := jobRespMap[jobID]
		if !ok {
			continue
		}
		if jobResp.Amount <= 0. {
			continue
		}

		resp.Data = append(resp.Data, *jobResp)
	}

	return
}
