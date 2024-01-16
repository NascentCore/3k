package logic

import (
	"context"
	"database/sql"
	"encoding/json"
	"strconv"
	"sxwl/3k/internal/scheduler/model"
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

	newUUID, err := uuid.NewRandom()
	if err != nil {
		l.Errorf("new uuid userId: %d err: %s", req.UserID, err)
		return nil, err
	}

	userJob := &model.SysUserJob{}
	_ = copier.Copy(userJob, req)
	userJob.UserId = req.UserID
	userJob.JobName = sql.NullString{String: "ai" + newUUID.String(), Valid: true}
	stopType, err := strconv.Atoi(req.StopType)
	if err != nil {
		l.Errorf("stopType convert userId: %d stopType: %s err: %s", req.UserID, req.StopType, err)
		return nil, err
	}
	userJob.StopType = sql.NullInt64{Int64: int64(stopType), Valid: true}
	userJob.CreateTime = sql.NullTime{Time: time.Now(), Valid: true}
	userJob.UpdateTime = sql.NullTime{Time: time.Now(), Valid: true}

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

	jsonAll["jobName"] = userJob.JobName.String
	jsonAll["modelVol"], err = strconv.Atoi(userJob.ModelVol.String)
	if err != nil {
		l.Errorf("modelVol convert userId: %d modelVol: %s err: %s", req.UserID, userJob.ModelVol.String, err)
		return nil, err
	}
	jsonAll["ckptVol"], err = strconv.Atoi(userJob.CkptVol.String)
	if err != nil {
		l.Errorf("ckptVol convert userId: %d ckptVol: %s err: %s", req.UserID, userJob.CkptVol.String, err)
		return nil, err
	}
	jsonAll["stopType"] = userJob.StopType.Int64

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
