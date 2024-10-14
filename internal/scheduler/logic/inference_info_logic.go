package logic

import (
	"context"
	"encoding/json"
	"fmt"
	"sxwl/3k/internal/scheduler/model"
	"time"

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
	selectBuilder = selectBuilder.Where(squirrel.Eq{"new_user_id": req.UserID})
	if req.ServiceName != "" {
		selectBuilder = selectBuilder.Where(squirrel.Eq{"service_name": req.ServiceName})
	}
	infers, err := InferenceModel.FindAll(l.ctx, selectBuilder, "")
	if err != nil {
		l.Errorf("InferenceModel.FindAll user_id: %s service_name: %s err: %s", req.UserID, req.ServiceName, err)
		return nil, err
	}

	resp = &types.InferenceInfoResp{}
	resp.Data = make([]types.SysInference, 0)
	for _, infer := range infers {
		inferResp := types.SysInference{}
		if infer.Metadata.Valid {
			_ = json.Unmarshal([]byte(infer.Metadata.String), &inferResp)
		}
		_ = copier.Copy(&inferResp, &infer)
		statusDesc, ok := model.StatusToStr[infer.Status]
		if ok {
			inferResp.Status = statusDesc
		}
		// if infer.ModelPublic.Int64 == model.CachePrivate {
		//	inferResp.ModelIsPublic = false
		// } else {
		//	inferResp.ModelIsPublic = true
		// }
		if infer.Url != "" {
			inferResp.Url = fmt.Sprintf("%s%s", l.svcCtx.Config.K8S.BaseUrl, infer.Url)
			// http://test.llm.nascentcore.net:30004/inference/api/infer-da9d08c4-0314-475e-bb74-b02d546e74a7/v1/chat/completions
			inferResp.API = fmt.Sprintf("%s/inference/api/%s/v1/chat/completions", l.svcCtx.Config.K8S.BaseUrl, infer.ServiceName)
		}
		if infer.StartTime.Valid {
			inferResp.StartTime = infer.StartTime.Time.Format(time.DateTime)
		}
		if infer.EndTime.Valid {
			inferResp.EndTime = infer.EndTime.Time.Format(time.DateTime)
		}

		resp.Data = append(resp.Data, inferResp)
	}

	return
}
