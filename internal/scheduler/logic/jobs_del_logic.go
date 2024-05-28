package logic

import (
	"context"
	"sxwl/3k/internal/scheduler/model"
	"sxwl/3k/pkg/orm"
	"time"

	"sxwl/3k/internal/scheduler/svc"
	"sxwl/3k/internal/scheduler/types"

	"github.com/Masterminds/squirrel"
	"github.com/zeromicro/go-zero/core/logx"
)

type JobsDelLogic struct {
	logx.Logger
	ctx    context.Context
	svcCtx *svc.ServiceContext
}

func NewJobsDelLogic(ctx context.Context, svcCtx *svc.ServiceContext) *JobsDelLogic {
	return &JobsDelLogic{
		Logger: logx.WithContext(ctx),
		ctx:    ctx,
		svcCtx: svcCtx,
	}
}

func (l *JobsDelLogic) JobsDel(req *types.JobsDelReq) (resp *types.JobsDelResp, err error) {
	// UserModel := l.svcCtx.UserModel
	JobModel := l.svcCtx.UserJobModel
	InferModel := l.svcCtx.InferenceModel
	JupyterlabModel := l.svcCtx.JupyterlabModel

	// isAdmin, err := UserModel.IsAdmin(l.ctx, req.UserID)
	// if err != nil {
	//     l.Errorf("UserModel.IsAdmin userID=%d err=%s", req.UserID, err)
	//     return nil, ErrNotAdmin
	// }
	// if !isAdmin {
	//     l.Infof("UserModel.IsAdmin userID=%d is not admin", req.UserID)
	//     return nil, ErrNotAdmin
	// }

	// cpodjob and finetune
	_, err = JobModel.UpdateColsByCond(l.ctx, JobModel.UpdateBuilder().Where(squirrel.Eq{
		"new_user_id": req.ToUser,
	}).Set("deleted", model.JobDeleted))
	if err != nil {
		l.Errorf("job deleted userID=%s err=%s", req.ToUser, err)
		return nil, err
	}

	// inference
	_, err = InferModel.UpdateColsByCond(l.ctx, InferModel.UpdateBuilder().Where(
		squirrel.Eq{"new_user_id": req.ToUser},
	).SetMap(map[string]interface{}{
		"status":   model.InferStatusStopped,
		"end_time": orm.NullTime(time.Now()),
	}))
	if err != nil {
		l.Errorf("inference stop user_id=%s err=%s", req.ToUser, err)
		return nil, err
	}

	// jupyterlab
	_, err = JupyterlabModel.UpdateColsByCond(l.ctx, JupyterlabModel.UpdateBuilder().Where(
		squirrel.Eq{"new_user_id": req.ToUser},
	).SetMap(map[string]interface{}{
		"status":   model.JupyterStatusStopped,
		"end_time": orm.NullTime(time.Now()),
	}))
	if err != nil {
		l.Errorf("jupyterlab stop user_id=%s err=%s", req.ToUser, err)
		return nil, err
	}

	return &types.JobsDelResp{Message: MsgJobsDelSuccess}, nil
}
