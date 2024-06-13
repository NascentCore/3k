package logic

import (
	"context"
	"database/sql"
	"encoding/json"
	"fmt"
	"sxwl/3k/internal/scheduler/job"
	"sxwl/3k/internal/scheduler/model"
	"sxwl/3k/pkg/consts"
	"sxwl/3k/pkg/orm"
	"sxwl/3k/pkg/storage"
	uuid2 "sxwl/3k/pkg/uuid"
	"time"

	"github.com/Masterminds/squirrel"
	"github.com/jinzhu/copier"

	"sxwl/3k/internal/scheduler/svc"
	"sxwl/3k/internal/scheduler/types"

	"github.com/zeromicro/go-zero/core/logx"
)

type JobCreateLogic struct {
	logx.Logger
	ctx    context.Context
	svcCtx *svc.ServiceContext
}

func NewJobCreateLogic(ctx context.Context, svcCtx *svc.ServiceContext) *JobCreateLogic {
	return &JobCreateLogic{
		Logger: logx.WithContext(ctx),
		ctx:    ctx,
		svcCtx: svcCtx,
	}
}

func (l *JobCreateLogic) JobCreate(req *types.JobCreateReq) (resp *types.JobCreateResp, err error) {
	BalanceModel := l.svcCtx.UserBalanceModel
	UserJobModel := l.svcCtx.UserJobModel

	// check balance
	balance, err := BalanceModel.FindOneByQuery(l.ctx, BalanceModel.AllFieldsBuilder().Where(squirrel.Eq{
		"new_user_id": req.UserID,
	}))
	if err != nil {
		return nil, err
	}
	if balance.Balance < 0.0 {
		return nil, fmt.Errorf("余额不足")
	}

	// check quota
	ok, left, err := job.CheckQuota(l.ctx, l.svcCtx, req.UserID, req.GpuType, req.GpuNumber)
	if err != nil {
		l.Errorf("JobCreate CheckQuota userId: %s GpuType: %s err: %s", req.UserID, req.GpuType, err)
		return nil, err
	}
	if !ok {
		err = fmt.Errorf("JobCreate CheckQuota userId: %s gpu: %s left: %d need: %d", req.UserID, req.GpuType, left, req.GpuNumber)
		l.Error(err)
		return nil, err
	}

	if req.TrainedModelName == "" {
		req.TrainedModelName, err = uuid2.WithPrefix("model")
		if err != nil {
			l.Errorf("create trained model name err=%s", err)
			return nil, ErrSystem
		}
	}

	userJob := &model.SysUserJob{}
	_ = copier.Copy(userJob, req)
	userJob.NewUserId = req.UserID
	jobName, err := uuid2.WithPrefix("train")
	if err != nil {
		l.Errorf("create train job name err=%s", err)
		return nil, ErrSystem
	}
	userJob.JobName = orm.NullString(jobName)
	if req.ModelName != "" {
		userJob.PretrainedModelName = orm.NullString(req.ModelName)
	}
	if req.ModelName != "" {
		userJob.DatasetName = orm.NullString(req.ModelName)
	}
	userJob.CreateTime = orm.NullTime(time.Now())
	userJob.UpdateTime = orm.NullTime(time.Now())
	userJob.BillingStatus = model.BillingStatusContinue

	// jsonAll := make(map[string]any)
	jsonAll := &model.JobJson{
		JobName:               jobName,
		PretrainModelId:       req.ModelId,
		PretrainModelName:     req.ModelName,
		PretrainModelIsPublic: req.ModelIsPublic,
		PretrainModelSize:     req.ModelSize,
		PretrainModelPath:     req.ModelPath,
		PretrainModelUrl: storage.OssPathToOssURL(
			l.svcCtx.Config.OSS.Bucket,
			storage.ResourceToOSSPath(consts.Model, req.ModelName)),
		PretrainModelTemplate: req.ModelTemplate,
		DatasetUrl: storage.OssPathToOssURL(
			l.svcCtx.Config.OSS.Bucket,
			storage.ResourceToOSSPath(consts.Dataset, req.DatasetName)),
		BackoffLimit: 1,
		UserID:       req.UserID,
	}

	// copy the other parts of req
	_ = copier.Copy(jsonAll, req)

	bytes, err := json.Marshal(jsonAll)
	if err != nil {
		l.Errorf("marshal jsonAll userId: %s err: %s", req.UserID, err)
		return nil, err
	}
	userJob.JsonAll = sql.NullString{String: string(bytes), Valid: true}

	_, err = UserJobModel.Insert(l.ctx, userJob)
	if err != nil {
		l.Errorf("insert userId: %s err: %s", req.UserID, err)
		return nil, err
	}

	resp = &types.JobCreateResp{JobId: userJob.JobName.String}
	return resp, nil
}
