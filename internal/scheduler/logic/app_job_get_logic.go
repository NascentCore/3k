package logic

import (
	"context"
	"sxwl/3k/internal/scheduler/model"
	"time"

	"sxwl/3k/internal/scheduler/svc"
	"sxwl/3k/internal/scheduler/types"

	"github.com/Masterminds/squirrel"
	"github.com/jinzhu/copier"
	"github.com/zeromicro/go-zero/core/logx"
)

type AppJobGetLogic struct {
	logx.Logger
	ctx    context.Context
	svcCtx *svc.ServiceContext
}

func NewAppJobGetLogic(ctx context.Context, svcCtx *svc.ServiceContext) *AppJobGetLogic {
	return &AppJobGetLogic{
		Logger: logx.WithContext(ctx),
		ctx:    ctx,
		svcCtx: svcCtx,
	}
}

func (l *AppJobGetLogic) AppJobGet(req *types.BaseReq) (resp *types.AppJobGetResp, err error) {
	AppJobModel := l.svcCtx.AppJobModel

	jobs, err := AppJobModel.Find(l.ctx, AppJobModel.AllFieldsBuilder().Where(squirrel.Eq{
		"user_id": req.UserID,
	}))
	if err != nil {
		l.Errorf("AppJobGetLogic Find user_id: %s err: %v", req.UserID, err)
		return nil, err
	}

	resp = &types.AppJobGetResp{
		Data: make([]types.AppJob, 0),
	}
	for _, job := range jobs {
		respJob := types.AppJob{}
		_ = copier.Copy(&respJob, &job)

		// status
		status, ok := model.StatusToStr[job.Status]
		if !ok {
			l.Errorf("AppJobGetLogic StatusToStr user_id: %s status: %d err: %v", req.UserID, job.Status, err)
			return nil, ErrSystem
		}
		respJob.Status = status

		// time
		if job.StartTime.Valid {
			respJob.StartTime = job.StartTime.Time.Format(time.DateTime)
		}
		if job.EndTime.Valid {
			respJob.EndTime = job.EndTime.Time.Format(time.DateTime)
		}
		respJob.CreatedAt = job.CreatedAt.Format(time.DateTime)
		respJob.UpdatedAt = job.UpdatedAt.Format(time.DateTime)

		resp.Data = append(resp.Data, respJob)
	}

	return
}
