package logic

import (
	"context"
	"encoding/json"
	"fmt"
	"sxwl/3k/internal/scheduler/job"
	"sxwl/3k/internal/scheduler/model"
	"sxwl/3k/pkg/orm"

	"github.com/jinzhu/copier"

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

	var modelPublic int64
	if req.ModelIsPublic {
		modelPublic = model.CachePublic
	} else {
		modelPublic = model.CachePrivate
	}

	// meta
	meta := types.InferenceService{}
	_ = copier.Copy(&meta, req)
	meta.GpuType = req.GpuModel
	meta.GpuNumber = req.GpuCount
	meta.Template = req.ModelTemplate
	meta.ModelMeta = req.ModelMeta
	meta.ServiceName = serviceName
	meta.MinInstances = req.MinInstances
	meta.MaxInstances = req.MaxInstances

	bytes, err := json.Marshal(meta)
	if err != nil {
		l.Errorf("json marshal meta err: %v", err)
		return nil, err
	}

	infer := &model.SysInference{
		ServiceName:   serviceName,
		NewUserId:     req.UserID,
		ModelName:     orm.NullString(req.ModelName),
		ModelId:       orm.NullString(req.ModelId),
		ModelSize:     orm.NullInt64(req.ModelSize),
		ModelPublic:   orm.NullInt64(modelPublic),
		ModelMeta:     orm.NullString(req.ModelMeta),
		GpuType:       orm.NullString(req.GpuModel),
		GpuNumber:     orm.NullInt64(req.GpuCount),
		Template:      orm.NullString(req.ModelTemplate),
		BillingStatus: model.BillingStatusContinue,
		Metadata:      orm.NullString(string(bytes)),
	}

	_, err = InferenceModel.Insert(l.ctx, infer)
	if err != nil {
		l.Errorf("insert userId: %s err: %s", req.UserID, err)
		return nil, err
	}

	resp = &types.InferenceDeployResp{ServiceName: serviceName}

	return
}
