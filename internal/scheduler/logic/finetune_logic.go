package logic

import (
	"context"
	"database/sql"
	"encoding/json"
	"fmt"
	"sxwl/3k/internal/scheduler/config"
	"sxwl/3k/internal/scheduler/job"
	"sxwl/3k/internal/scheduler/model"
	"sxwl/3k/pkg/consts"
	"sxwl/3k/pkg/orm"
	"sxwl/3k/pkg/storage"
	"time"

	"github.com/jinzhu/copier"

	"github.com/Masterminds/squirrel"

	"sxwl/3k/internal/scheduler/svc"
	"sxwl/3k/internal/scheduler/types"

	"github.com/zeromicro/go-zero/core/logx"
)

type FinetuneLogic struct {
	logx.Logger
	ctx    context.Context
	svcCtx *svc.ServiceContext
}

func NewFinetuneLogic(ctx context.Context, svcCtx *svc.ServiceContext) *FinetuneLogic {
	return &FinetuneLogic{
		Logger: logx.WithContext(ctx),
		ctx:    ctx,
		svcCtx: svcCtx,
	}
}

func (l *FinetuneLogic) Finetune(req *types.FinetuneReq) (resp *types.FinetuneResp, err error) {
	UserJobModel := l.svcCtx.UserJobModel
	CpodMainModel := l.svcCtx.CpodMainModel

	// check model is in list
	_, ok := l.svcCtx.Config.FinetuneModel[req.Model]
	if !ok {
		l.Errorf("finetune model %s is not supported userId: %d", req.Model, req.UserID)
		return nil, fmt.Errorf("model: %s for codeless fine-tune is not supported", req.Model)
	}

	userJob := &model.SysUserJob{}
	userJob.UserId = req.UserID
	jobName, err := job.NewJobName()
	if err != nil {
		l.Errorf("new uuid userId: %d err: %s", req.UserID, err)
		return nil, err
	}
	userJob.JobName = orm.NullString(jobName)

	// model
	userJob.PretrainedModelName = orm.NullString(req.Model)

	// dataset
	datasetOSSPath := storage.ResourceToOSSPath(consts.Dataset, req.TrainingFile)
	ok, datasetSize, err := storage.ExistDir(l.svcCtx.Config.OSS.Bucket, datasetOSSPath)
	if err != nil {
		l.Errorf("dataset storage.ExistDir userID: %d dataset: %s err: %s", req.UserID, req.TrainingFile, err)
		return nil, err
	}
	if !ok {
		l.Errorf("dataset not exists userID: %d dataset: %s err: %s", req.UserID, req.TrainingFile, err)
		return nil, fmt.Errorf("dataset not exists dataset: %s", req.TrainingFile)
	}
	userJob.DatasetName = orm.NullString(storage.DatasetCRDName(datasetOSSPath))

	// job_type
	userJob.JobType = orm.NullString(consts.JobTypeCodeless)

	// fill gpu
	if req.GpuModel == "" {
		cpodMain, err := CpodMainModel.FindOneByQuery(l.ctx, CpodMainModel.AllFieldsBuilder().Where(squirrel.And{
			squirrel.GtOrEq{
				"update_time": orm.NullTime(time.Now().Add(-30 * time.Minute)),
			},
		}))
		if err != nil {
			l.Errorf("finetune no gpu match model: %s userId: %d", req.Model, req.UserID)
			return nil, fmt.Errorf("finetune no gpu match model: %s userId: %d", req.Model, req.UserID)
		}
		userJob.GpuType = cpodMain.GpuProd
		userJob.GpuNumber = orm.NullInt64(1) // 无代码微调暂时都写1
	} else {
		userJob.GpuType = orm.NullString(req.GpuModel)
		userJob.GpuNumber = orm.NullInt64(req.GpuCount)
	}

	// check gpu quota
	ok, left, err := job.CheckQuota(l.ctx, l.svcCtx, req.UserID, userJob.GpuType.String, userJob.GpuNumber.Int64)
	if err != nil {
		l.Errorf("finetune CheckQuota userId: %d GpuType: %s err: %s", req.UserID, userJob.GpuType.String, err)
		return nil, err
	}
	if !ok {
		err = fmt.Errorf("finetune CheckQuota userId: %d gpu: %s left: %d need: %d", req.UserID, userJob.GpuType.String, left, userJob.GpuNumber.Int64)
		l.Error(err)
		return nil, err
	}

	// trainedModelName
	if req.TrainedModelName == "" {
		req.TrainedModelName = fmt.Sprintf("%s-%s", req.Model, time.Now().Format(consts.JobTimestampFormat))
	}

	// time
	userJob.CreateTime = sql.NullTime{Time: time.Now(), Valid: true}
	userJob.UpdateTime = sql.NullTime{Time: time.Now(), Valid: true}

	// Hyperparameters
	epochs, ok := req.Hyperparameters[config.ParamEpochs]
	if !ok {
		epochs = "3.0"
	}
	batchSize, ok := req.Hyperparameters[config.ParamBatchSize]
	if !ok {
		batchSize = "4"
	}
	learningRate, ok := req.Hyperparameters[config.ParamLearningRate]
	if !ok {
		learningRate = "5e-5"
	}

	// json_all
	createReq := types.JobCreateReq{}
	_ = copier.Copy(&createReq, &userJob)

	jsonAll := make(map[string]any)
	bytes, err := json.Marshal(createReq)
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
	jsonAll["userId"] = req.UserID
	jsonAll["datasetId"] = userJob.DatasetName.String
	jsonAll["datasetName"] = req.TrainingFile
	jsonAll["datasetUrl"] = storage.OssPathToOssURL(
		l.svcCtx.Config.OSS.Bucket,
		storage.ResourceToOSSPath(consts.Dataset, req.TrainingFile))
	jsonAll["datasetSize"] = datasetSize
	jsonAll["pretrainedModelName"] = req.Model
	jsonAll["trainedModelName"] = req.TrainedModelName
	jsonAll["backoffLimit"] = 1 // 重试次数，默认为1
	jsonAll["ckptVol"] = 0      // 改为数值型默认值
	jsonAll["modelVol"] = 0     // 改为数值型默认值
	jsonAll["stopType"] = 0     // 改为数值型默认值
	jsonAll["epochs"] = epochs
	jsonAll["batchSize"] = batchSize
	jsonAll["learningRate"] = learningRate

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

	resp = &types.FinetuneResp{JobId: userJob.JobName.String}
	return resp, nil
}
