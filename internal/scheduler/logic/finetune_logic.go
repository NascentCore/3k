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
	CpodCacheModel := l.svcCtx.CpodCacheModel
	CpodMainModel := l.svcCtx.CpodMainModel

	userJob := &model.SysUserJob{}
	userJob.UserId = req.UserID
	jobName, err := job.NewJobName()
	if err != nil {
		l.Errorf("new uuid userId: %d err: %s", req.UserID, err)
		return nil, err
	}
	userJob.JobName = orm.NullString(jobName)

	// modelstorage id
	cache, err := CpodCacheModel.FindOneByQuery(l.ctx, CpodCacheModel.AllFieldsBuilder().Where(squirrel.Eq{
		"data_type": model.CacheModel,
		"data_name": req.Model,
	}))
	if err != nil {
		l.Errorf("finetune find model err %s model: %s userId: %d", err, req.Model, req.UserID)
		return nil, err
	}
	userJob.PretrainedModelName = orm.NullString(cache.DataId)
	userJob.PretrainedModelPath = orm.NullString("/data/model")

	// dataset
	cache, err = CpodCacheModel.FindOneByQuery(l.ctx, CpodCacheModel.AllFieldsBuilder().Where(squirrel.Eq{
		"data_type": model.CacheDataset,
		"data_name": req.TrainingFile,
	}))
	if err != nil {
		l.Errorf("finetune find dataset err %s training_file: %s userId: %d", err, req.TrainingFile, req.UserID)
		return nil, err
	}
	userJob.DatasetName = orm.NullString(cache.DataId)
	userJob.DatasetPath = orm.NullString("/data/dataset/custom")

	// output
	userJob.ModelPath = orm.NullString("/data/save")
	userJob.ModelVol = orm.NullString("200")

	// image
	ftModel, ok := l.svcCtx.Config.FinetuneModel[req.Model]
	if !ok {
		l.Errorf("finetune model %s with no image userId: %d", req.Model, req.UserID)
		return nil, fmt.Errorf("model: %s for codeless fine-tune is not supported", req.Model)
	}
	userJob.ImagePath = orm.NullString(ftModel.Image)

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
	userJob.GpuNumber = orm.NullInt64(1)

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
	jsonAll["datasetId"] = userJob.DatasetName.String
	jsonAll["pretrainedModelId"] = userJob.PretrainedModelName.String
	jsonAll["ckptVol"] = 0
	jsonAll["modelVol"] = 200
	jsonAll["stopType"] = 0

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
