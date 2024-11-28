package logic

import (
	"context"
	"encoding/json"
	"fmt"
	"time"

	"sxwl/3k/internal/scheduler/model"
	"sxwl/3k/internal/scheduler/svc"
	"sxwl/3k/internal/scheduler/types"

	"github.com/Masterminds/squirrel"
	"github.com/jinzhu/copier"
	"github.com/zeromicro/go-zero/core/logx"
)

type InferencePlaygroundLogic struct {
	logx.Logger
	ctx    context.Context
	svcCtx *svc.ServiceContext
}

func NewInferencePlaygroundLogic(ctx context.Context, svcCtx *svc.ServiceContext) *InferencePlaygroundLogic {
	return &InferencePlaygroundLogic{
		Logger: logx.WithContext(ctx),
		ctx:    ctx,
		svcCtx: svcCtx,
	}
}

func (l *InferencePlaygroundLogic) InferencePlayground(req *types.BaseReq) (resp *types.InferenceInfoResp, err error) {
	InferenceModel := l.svcCtx.InferenceModel
	UserModel := l.svcCtx.UserModel

	user, err := UserModel.FindOneByQuery(l.ctx, UserModel.AllFieldsBuilder().Where(squirrel.Eq{"username": "playground@sxwl.ai"}))
	if err != nil {
		l.Errorf("UserModel.FindOneByQuery user_id: %s err: %s", req.UserID, err)
		return nil, ErrSystem
	}

	selectBuilder := InferenceModel.AllFieldsBuilder()
	selectBuilder = selectBuilder.Where(squirrel.Eq{"new_user_id": user.NewUserId})
	infers, err := InferenceModel.FindAll(l.ctx, selectBuilder, "")
	if err != nil {
		l.Errorf("InferenceModel.FindAll user_id: %s err: %s", user.NewUserId, err)
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
		inferResp.GPUModel = infer.GpuType.String
		inferResp.GPUCount = int(infer.GpuNumber.Int64)
		if infer.Url != "" {
			inferResp.Url = fmt.Sprintf("%s%s", l.svcCtx.Config.K8S.BaseUrl, infer.Url)
			inferResp.API = fmt.Sprintf("%s/inference/api/%s/v1/chat/completions", l.svcCtx.Config.K8S.BaseUrl, infer.ServiceName)
		}
		if infer.StartTime.Valid {
			inferResp.StartTime = infer.StartTime.Time.Format(time.DateTime)
		}
		if infer.EndTime.Valid {
			inferResp.EndTime = infer.EndTime.Time.Format(time.DateTime)
		}
		inferResp.CreateTime = infer.CreatedAt.Format(time.DateTime)

		resp.Data = append(resp.Data, inferResp)
	}

	return
}
