package logic

import (
	"context"
	"fmt"
	"sxwl/3k/internal/scheduler/model"
	"sxwl/3k/pkg/consts"
	"sxwl/3k/pkg/orm"
	"sxwl/3k/pkg/storage"

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
	InferenceModel := l.svcCtx.InferenceModel

	newUUID, err := uuid.NewRandom()
	if err != nil {
		l.Errorf("new uuid userId: %d err: %s", req.UserID, err)
		return nil, err
	}
	serviceName := "infer-" + newUUID.String()

	// model
	modelOSSPath := storage.ResourceToOSSPath(consts.Model, req.ModelName)
	ok, modelSize, err := storage.ExistDir(l.svcCtx.Config.OSS.Bucket, modelOSSPath)
	if err != nil {
		l.Errorf("model storage.ExistDir userID: %d model: %s err: %s", req.UserID, req.ModelName, err)
		return nil, err
	}
	if !ok {
		l.Errorf("model not exists userID: %d model: %s err: %s", req.UserID, req.ModelName, err)
		return nil, fmt.Errorf("model not exists model: %s", req.ModelName)
	}

	infer := &model.SysInference{
		ServiceName: serviceName,
		UserId:      req.UserID,
		ModelName:   orm.NullString(req.ModelName),
		ModelId:     orm.NullString(storage.ModelCRDName(modelOSSPath)),
		ModelSize:   orm.NullInt64(modelSize),
	}

	_, err = InferenceModel.Insert(l.ctx, infer)
	if err != nil {
		l.Errorf("insert userId: %d err: %s", req.UserID, err)
		return nil, err
	}

	resp = &types.InferenceDeployResp{ServiceName: serviceName}

	return
}
