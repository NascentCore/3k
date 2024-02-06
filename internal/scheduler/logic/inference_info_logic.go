package logic

import (
	"context"
	"fmt"
	"sxwl/3k/internal/scheduler/model"

	"github.com/jinzhu/copier"

	"github.com/Masterminds/squirrel"

	"sxwl/3k/internal/scheduler/svc"
	"sxwl/3k/internal/scheduler/types"

	"github.com/zeromicro/go-zero/core/logx"
)

type InferenceInfoLogic struct {
	logx.Logger
	ctx    context.Context
	svcCtx *svc.ServiceContext
}

func NewInferenceInfoLogic(ctx context.Context, svcCtx *svc.ServiceContext) *InferenceInfoLogic {
	return &InferenceInfoLogic{
		Logger: logx.WithContext(ctx),
		ctx:    ctx,
		svcCtx: svcCtx,
	}
}

func (l *InferenceInfoLogic) InferenceInfo(req *types.InferenceInfoReq) (resp *types.InferenceInfoResp, err error) {
	InferenceModel := l.svcCtx.InferenceModel

	selectBuilder := InferenceModel.AllFieldsBuilder()
	selectBuilder = selectBuilder.Where(squirrel.Eq{"user_id": req.UserID})
	if req.ServiceName != "" {
		selectBuilder = selectBuilder.Where(squirrel.Eq{"service_name": req.ServiceName})
	}
	infers, err := InferenceModel.FindAll(l.ctx, selectBuilder, "")
	if err != nil {
		l.Errorf("InferenceModel.FindAll user_id: %d service_name: %s err: %s", req.UserID, req.ServiceName, err)
		return nil, err
	}

	resp = &types.InferenceInfoResp{}
	resp.Data = make([]types.SysInference, 0)
	for _, infer := range infers {
		inferResp := types.SysInference{}
		_ = copier.Copy(&inferResp, &infer)
		statusDesc, ok := model.InferStatusToDesc[infer.Status]
		if ok {
			inferResp.Status = statusDesc
		}
		inferResp.Url = fmt.Sprintf("http://%s:30005/v1/chat/completions", infer.Url)
		resp.Data = append(resp.Data, inferResp)
	}

	return
}
