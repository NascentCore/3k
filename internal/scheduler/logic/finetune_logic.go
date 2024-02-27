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

	userJob := &model.SysUserJob{}
	userJob.UserId = req.UserID
	jobName, err := job.NewJobName()
	if err != nil {
		l.Errorf("new uuid userId: %d err: %s", req.UserID, err)
		return nil, err
	}
	userJob.JobName = orm.NullString(jobName)

	// model
	modelOSSPath := storage.ResourceToOSSPath(consts.Model, req.Model)
	ok, modelSize, err := storage.ExistDir(l.svcCtx.Config.OSS.Bucket, modelOSSPath)
	if err != nil {
		l.Errorf("model storage.ExistDir userID: %d model: %s err: %s", req.UserID, req.Model, err)
		return nil, err
	}
	if !ok {
		l.Errorf("model not exists userID: %d model: %s err: %s", req.UserID, req.Model, err)
		return nil, fmt.Errorf("model not exists model: %s", req.Model)
	}
	userJob.PretrainedModelName = orm.NullString(storage.ModelCRDName(modelOSSPath))
	userJob.PretrainedModelPath = orm.NullString("/data/model")

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
	userJob.DatasetPath = orm.NullString("/data/dataset/custom")

	// image
	ftModel, ok := l.svcCtx.Config.FinetuneModel[req.Model]
	if !ok {
		l.Errorf("finetune model %s with no image userId: %d", req.Model, req.UserID)
		return nil, fmt.Errorf("model: %s for codeless fine-tune is not supported", req.Model)
	}
	userJob.ImagePath = orm.NullString(ftModel.Image)

	// output
	userJob.ModelPath = orm.NullString("/data/save")
	userJob.ModelVol = orm.NullString(fmt.Sprintf("%d", ftModel.ModelVol))

	// job_type
	userJob.JobType = orm.NullString(consts.JobTypePytorch)

	// fill gpu
	cpodMain, err := CpodMainModel.FindOneByQuery(l.ctx, CpodMainModel.AllFieldsBuilder().Where(squirrel.And{
		squirrel.GtOrEq{
			"update_time": orm.NullTime(time.Now().Add(-30 * time.Minute)),
		},
		squirrel.GtOrEq{
			"gpu_mem": orm.NullInt64(ftModel.GPUMem),
		},
	}))
	if err != nil {
		l.Errorf("finetune no gpu match model: %s mem: %d userId: %d", req.Model, ftModel.GPUMem, req.UserID)
		return nil, fmt.Errorf("finetune no gpu match model: %s mem: %d userId: %d",
			req.Model, ftModel.GPUMem, req.UserID)
	}
	userJob.GpuType = cpodMain.GpuProd
	userJob.GpuNumber = orm.NullInt64(ftModel.GPUNum)

	// time
	userJob.CreateTime = sql.NullTime{Time: time.Now(), Valid: true}
	userJob.UpdateTime = sql.NullTime{Time: time.Now(), Valid: true}

	// run_command
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
	cmd := fmt.Sprintf(ftModel.Command,
		epochs, learningRate, batchSize)
	for param, value := range req.Config {
		cmd = cmd + fmt.Sprintf(" %s=%s", param, value)
	}
	userJob.RunCommand = orm.NullString(cmd)

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
	jsonAll["datasetSize"] = datasetSize
	jsonAll["pretrainedModelId"] = userJob.PretrainedModelName.String
	jsonAll["pretrainedModelName"] = req.Model
	jsonAll["pretrainedModelSize"] = modelSize
	jsonAll["ckptVol"] = 0
	jsonAll["modelVol"] = ftModel.ModelVol
	jsonAll["stopType"] = 0
	jsonAll["backoffLimit"] = 1 // 重试次数，默认为1

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
