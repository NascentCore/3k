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
	uuid2 "sxwl/3k/pkg/uuid"
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
	BalanceModel := l.svcCtx.UserBalanceModel
	UserJobModel := l.svcCtx.UserJobModel
	CpodMainModel := l.svcCtx.CpodMainModel

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

	// check model is in list
	_, ok := l.svcCtx.Config.FinetuneModel[req.ModelName]
	if !ok {
		l.Errorf("finetune model %s is not supported userId: %s", req.ModelName, req.UserID)
		return nil, fmt.Errorf("model: %s for finetune is not supported", req.ModelName)
	}

	// // template
	// template := ""
	// fileList, err := storage.ListFiles(l.svcCtx.Config.OSS.Bucket, storage.ResourceToOSSPath(consts.Model, req.ModelName))
	// if err != nil {
	// 	l.Errorf("model storage.ListFiles userID: %s model: %s err: %s", req.UserID, req.ModelName, err)
	// 	return nil, err
	// }
	// for file := range fileList {
	// 	if strings.Contains(file, "sxwl-infer-template-") {
	// 		template = storage.ExtractTemplate(file)
	// 		break
	// 	}
	// }

	userJob := &model.SysUserJob{}
	userJob.NewUserId = req.UserID
	jobName, err := uuid2.WithPrefix("finetune")
	if err != nil {
		l.Errorf("new uuid userId: %s err: %s", req.UserID, err)
		return nil, err
	}
	userJob.JobName = orm.NullString(jobName)

	// model
	userJob.PretrainedModelName = orm.NullString(req.ModelName)

	// dataset
	// datasetOSSPath := storage.ResourceToOSSPath(consts.Dataset, req.DatasetName)
	// ok, datasetSize, err := storage.ExistDir(l.svcCtx.Config.OSS.Bucket, datasetOSSPath)
	// if err != nil {
	// 	l.Errorf("dataset storage.ExistDir userID: %s dataset: %s err: %s", req.UserID, req.TrainingFile, err)
	// 	return nil, err
	// }
	// if !ok {
	// 	l.Errorf("dataset not exists userID: %s dataset: %s err: %s", req.UserID, req.TrainingFile, err)
	// 	return nil, fmt.Errorf("dataset not exists dataset: %s", req.TrainingFile)
	// }
	// userJob.DatasetName = orm.NullString(storage.DatasetCRDName(datasetOSSPath))
	userJob.DatasetName = orm.NullString(req.DatasetId)

	// job_type
	userJob.JobType = orm.NullString(consts.JobTypeFinetune)

	// fill gpu
	if req.GpuModel == "" {
		cpodMain, err := CpodMainModel.FindOneByQuery(l.ctx, CpodMainModel.AllFieldsBuilder().Where(squirrel.And{
			squirrel.GtOrEq{
				"update_time": orm.NullTime(time.Now().Add(-30 * time.Minute)),
			},
		}))
		if err != nil {
			l.Errorf("finetune no gpu match model: %s userId: %s", req.ModelName, req.UserID)
			return nil, fmt.Errorf("finetune no gpu match model: %s userId: %s", req.ModelName, req.UserID)
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
		l.Errorf("finetune CheckQuota userId: %s GpuType: %s err: %s", req.UserID, userJob.GpuType.String, err)
		return nil, err
	}
	if !ok {
		err = fmt.Errorf("finetune CheckQuota userId: %s gpu: %s left: %d need: %d", req.UserID, userJob.GpuType.String, left, userJob.GpuNumber.Int64)
		l.Error(err)
		return nil, err
	}

	// trainedModelName
	if req.TrainedModelName == "" {
		req.TrainedModelName = fmt.Sprintf("%s-%s", req.ModelName, time.Now().Format(consts.JobTimestampFormat))
	}

	// time
	userJob.CreateTime = sql.NullTime{Time: time.Now(), Valid: true}
	userJob.UpdateTime = sql.NullTime{Time: time.Now(), Valid: true}
	// billing_status
	userJob.BillingStatus = model.BillingStatusContinue

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
		l.Errorf("marshal userJob userId: %s err: %s", req.UserID, err)
		return nil, err
	}

	err = json.Unmarshal(bytes, &jsonAll)
	if err != nil {
		l.Errorf("unmarshal userId: %s err: %s", req.UserID, err)
		return nil, err
	}

	jsonAll["jobName"] = userJob.JobName.String
	jsonAll["userId"] = req.UserID
	jsonAll["datasetId"] = userJob.DatasetName.String
	jsonAll["datasetName"] = req.DatasetName
	jsonAll["datasetUrl"] = storage.OssPathToOssURL(
		l.svcCtx.Config.OSS.Bucket,
		storage.ResourceToOSSPath(consts.Dataset, req.DatasetName))
	jsonAll["datasetSize"] = req.DatasetSize
	jsonAll["datasetIsPublic"] = req.DatasetIsPublic
	jsonAll["pretrainedModelName"] = req.ModelName
	jsonAll["pretrainedModelIsPublic"] = req.ModelIsPublic
	jsonAll["pretrainedModelTemplate"] = req.ModelTemplate
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
		l.Errorf("marshal jsonAll userId: %s err: %s", req.UserID, err)
		return nil, err
	}
	userJob.JsonAll = sql.NullString{String: string(bytes), Valid: true}

	_, err = UserJobModel.Insert(l.ctx, userJob)
	if err != nil {
		l.Errorf("insert userId: %s err: %s", req.UserID, err)
		return nil, err
	}

	resp = &types.FinetuneResp{JobId: userJob.JobName.String}
	return resp, nil
}
