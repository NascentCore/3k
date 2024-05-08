package logic

import (
	"context"
	"database/sql"
	"encoding/json"
	"fmt"
	"sxwl/3k/internal/scheduler/job"
	"sxwl/3k/internal/scheduler/model"
	"sxwl/3k/pkg/orm"
	"time"

	"github.com/jinzhu/copier"

	"github.com/google/uuid"

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
	UserJobModel := l.svcCtx.UserJobModel

	// check quota
	ok, left, err := job.CheckQuota(l.ctx, l.svcCtx, req.UserID, req.GpuType, req.GpuNumber)
	if err != nil {
		l.Errorf("JobCreate CheckQuota userId: %d GpuType: %s err: %s", req.UserID, req.GpuType, err)
		return nil, err
	}
	if !ok {
		err = fmt.Errorf("JobCreate CheckQuota userId: %d gpu: %s left: %d need: %d", req.UserID, req.GpuType, left, req.GpuNumber)
		l.Error(err)
		return nil, err
	}

	newUUID, err := uuid.NewRandom()
	if err != nil {
		l.Errorf("new uuid userId: %d err: %s", req.UserID, err)
		return nil, err
	}

	if req.TrainedModelName == "" {
		req.TrainedModelName = fmt.Sprintf("model-%s", newUUID.String())
	}

	userJob := &model.SysUserJob{}
	_ = copier.Copy(userJob, req)
	userJob.UserId = req.UserID
	userJob.JobName = sql.NullString{String: "ai" + newUUID.String(), Valid: true}
	if req.PretrainedModelId != "" {
		userJob.PretrainedModelName = orm.NullString(req.PretrainedModelId)
	}
	if req.DatasetId != "" {
		userJob.DatasetName = orm.NullString(req.DatasetId)
	}
	userJob.CreateTime = orm.NullTime(time.Now())
	userJob.UpdateTime = orm.NullTime(time.Now())

	jsonAll := make(map[string]any)
	bytes, err := json.Marshal(req)
	if err != nil {
		l.Errorf("marshal userJob userId: %d err: %s", req.UserID, err)
		return nil, err
	}

	err = json.Unmarshal(bytes, &jsonAll)
	if err != nil {
		l.Errorf("unmarshal userId: %d err: %s", req.UserID, err)
		return nil, err
	}

	jsonAll["userId"] = req.UserID
	jsonAll["jobName"] = userJob.JobName.String
	jsonAll["backoffLimit"] = 1
	jsonAll["pretrainModelIsPublic"] = req.PretrainedModelIsPublic
	jsonAll["datasetIsPublic"] = req.DatasetIsPublic

	bytes, err = json.Marshal(jsonAll)
	if err != nil {
		l.Errorf("marshal jsonAll userId: %d err: %s", req.UserID, err)
		return nil, err
	}
	userJob.JsonAll = sql.NullString{String: string(bytes), Valid: true}

	_, err = UserJobModel.Insert(l.ctx, userJob)
	if err != nil {
		l.Errorf("insert userId: %d err: %s", req.UserID, err)
		return nil, err
	}

	resp = &types.JobCreateResp{JobId: userJob.JobName.String}
	return resp, nil
}
