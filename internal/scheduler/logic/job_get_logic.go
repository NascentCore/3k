package logic

import (
	"context"
	"fmt"
	"sxwl/3k/internal/scheduler/svc"
	"sxwl/3k/internal/scheduler/types"
	"time"

	"github.com/Masterminds/squirrel"
	"github.com/jinzhu/copier"
	"github.com/zeromicro/go-zero/core/logx"
)

type JobGetLogic struct {
	logx.Logger
	ctx    context.Context
	svcCtx *svc.ServiceContext
}

func NewJobGetLogic(ctx context.Context, svcCtx *svc.ServiceContext) *JobGetLogic {
	return &JobGetLogic{
		Logger: logx.WithContext(ctx),
		ctx:    ctx,
		svcCtx: svcCtx,
	}
}

func (l *JobGetLogic) JobGet(req *types.JobGetReq) (resp *types.JobGetResp, err error) {
	UserJobModel := l.svcCtx.UserJobModel
	userJobs, total, err := UserJobModel.FindPageListByPage(l.ctx, squirrel.Eq{"new_user_id": req.UserID},
		int64(req.Current), int64(req.Size), "")
	if err != nil {
		l.Errorf("get userJobs user_id: %s err: %s", req.UserID, err)
		return nil, err
	}

	resp = &types.JobGetResp{}
	resp.TotalElements = total
	resp.Content = make([]types.Job, 0)
	for _, userJob := range userJobs {
		job := types.Job{}
		_ = copier.Copy(&job, userJob)
		job.CreateTime = userJob.CreateTime.Time.Format(time.DateTime)
		job.UpdateTime = userJob.UpdateTime.Time.Format(time.DateTime)
		job.UserId = userJob.NewUserId
		job.TensorURL = fmt.Sprintf("%s/tensorboard/%s/#timeseries", l.svcCtx.Config.K8S.BaseUrl, req.UserID)
		resp.Content = append(resp.Content, job)
	}

	return
}
