package logic

import (
	"context"
	"fmt"
	"strings"
	"sxwl/3k/internal/scheduler/job"
	"sxwl/3k/internal/scheduler/model"
	"sxwl/3k/pkg/consts"
	"sxwl/3k/pkg/orm"
	"sxwl/3k/pkg/storage"

	"github.com/Masterminds/squirrel"
	"github.com/google/uuid"

	"sxwl/3k/internal/scheduler/svc"
	"sxwl/3k/internal/scheduler/types"

	"github.com/zeromicro/go-zero/core/logx"
)

type InferenceDeployLogic struct {
	logx.Logger
	ctx    context.Context
	svcCtx *svc.ServiceContext
}

func NewInferenceDeployLogic(ctx context.Context, svcCtx *svc.ServiceContext) *InferenceDeployLogic {
	return &InferenceDeployLogic{
		Logger: logx.WithContext(ctx),
		ctx:    ctx,
		svcCtx: svcCtx,
	}
}

func (l *InferenceDeployLogic) InferenceDeploy(req *types.InferenceDeployReq) (resp *types.InferenceDeployResp, err error) {
	BalanceModel := l.svcCtx.UserBalanceModel
	InferenceModel := l.svcCtx.InferenceModel
	CpodCacheModel := l.svcCtx.CpodCacheModel

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

	// check gpu quota
	ok, left, err := job.CheckQuota(l.ctx, l.svcCtx, req.UserID, req.GpuModel, req.GpuCount)
	if err != nil {
		l.Errorf("InferenceDeploy CheckQuota userId: %s GpuType: %s err: %s", req.UserID, req.GpuModel, err)
		return nil, err
	}
	if !ok {
		err = fmt.Errorf("InferenceDeploy CheckQuota userId: %s gpu: %s left: %d need: %d", req.UserID, req.GpuModel, left, req.GpuCount)
		l.Error(err)
		return nil, err
	}

	newUUID, err := uuid.NewRandom()
	if err != nil {
		l.Errorf("new uuid userId: %s err: %s", req.UserID, err)
		return nil, err
	}
	serviceName := "infer-" + newUUID.String()

	// model
	var modelSize, modelPublic int64
	var modelOSSPath, modelId, template string
	if l.svcCtx.Config.OSS.LocalMode {
		cacheModel, err := CpodCacheModel.FindOneByQuery(l.ctx, CpodCacheModel.AllFieldsBuilder().Where(
			squirrel.And{squirrel.Eq{"data_type": model.CacheModel}, squirrel.Eq{"data_name": req.ModelName}},
		))
		if err != nil {
			l.Errorf("model not exists userID: %s model: %s err: %s", req.UserID, req.ModelName, err)
			return nil, fmt.Errorf("model not exists model: %s", req.ModelName)
		}
		modelSize = cacheModel.DataSize
		modelId = cacheModel.DataId
		template = cacheModel.Template
		modelPublic = cacheModel.Public
	} else {
		ossPath := storage.ResourceToOSSPath(consts.Model, req.ModelName)
		ok, size, err := storage.ExistDir(l.svcCtx.Config.OSS.Bucket, ossPath)
		if err != nil {
			l.Errorf("model storage.ExistDir userID: %s model: %s err: %s", req.UserID, req.ModelName, err)
			return nil, err
		}
		if !ok {
			l.Errorf("model not exists userID: %s model: %s err: %s", req.UserID, req.ModelName, err)
			return nil, fmt.Errorf("model not exists model: %s", req.ModelName)
		}
		modelSize = size
		modelOSSPath = ossPath
		modelId = storage.ModelCRDName(ossPath)
		isPublic := storage.ResourceIsPublic(req.ModelName)
		if isPublic {
			modelPublic = model.CachePublic
		} else {
			modelPublic = model.CachePrivate
		}
		// template
		fileList, err := storage.ListFiles(l.svcCtx.Config.OSS.Bucket, modelOSSPath)
		if err != nil {
			l.Errorf("model storage.ListFiles userID: %s model: %s err: %s", req.UserID, req.ModelName, err)
			return nil, err
		}
		for file := range fileList {
			if strings.Contains(file, "sxwl-infer-template-") {
				template = storage.ExtractTemplate(file)
				break
			}
		}
	}

	infer := &model.SysInference{
		ServiceName:   serviceName,
		NewUserId:     req.UserID,
		ModelName:     orm.NullString(req.ModelName),
		ModelId:       orm.NullString(modelId),
		ModelSize:     orm.NullInt64(modelSize),
		ModelPublic:   orm.NullInt64(modelPublic),
		GpuType:       orm.NullString(req.GpuModel),
		GpuNumber:     orm.NullInt64(req.GpuCount),
		Template:      orm.NullString(template),
		BillingStatus: model.BillingStatusContinue,
	}

	_, err = InferenceModel.Insert(l.ctx, infer)
	if err != nil {
		l.Errorf("insert userId: %s err: %s", req.UserID, err)
		return nil, err
	}

	resp = &types.InferenceDeployResp{ServiceName: serviceName}

	return
}
