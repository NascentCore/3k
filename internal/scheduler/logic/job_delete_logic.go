package logic

import (
	"context"
	"sxwl/3k/internal/scheduler/model"

	"sxwl/3k/internal/scheduler/svc"

	"github.com/zeromicro/go-zero/core/logx"
)

type JobDeleteLogic struct {
	logx.Logger
	ctx    context.Context
	svcCtx *svc.ServiceContext
}

func NewJobDeleteLogic(ctx context.Context, svcCtx *svc.ServiceContext) *JobDeleteLogic {
	return &JobDeleteLogic{
		Logger: logx.WithContext(ctx),
		ctx:    ctx,
		svcCtx: svcCtx,
	}
}

func (l *JobDeleteLogic) JobDelete(ids []int64) (err error) {
	if len(ids) == 0 {
		return nil
	}

	for _, id := range ids {
		err = l.svcCtx.UserJobModel.DeleteSoft(l.ctx, &model.SysUserJob{JobId: id})
		if err != nil {
			return err
		}
	}

	return nil
}
