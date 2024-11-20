package logic

import (
	"context"
	"fmt"

	"sxwl/3k/internal/scheduler/model"
	"sxwl/3k/internal/scheduler/svc"
	"sxwl/3k/internal/scheduler/types"

	"github.com/Masterminds/squirrel"
	"github.com/zeromicro/go-zero/core/logx"
)

type InferenceStatusLogic struct {
	logx.Logger
	ctx    context.Context
	svcCtx *svc.ServiceContext
}

func NewInferenceStatusLogic(ctx context.Context, svcCtx *svc.ServiceContext) *InferenceStatusLogic {
	return &InferenceStatusLogic{
		Logger: logx.WithContext(ctx),
		ctx:    ctx,
		svcCtx: svcCtx,
	}
}

func (l *InferenceStatusLogic) InferenceStatus(req *types.InferenceStatusReq) (resp *types.InferenceStatusResp, err error) {
	InferenceModel := l.svcCtx.InferenceModel

	infer, err := InferenceModel.FindOneByQuery(l.ctx, InferenceModel.AllFieldsBuilder().Where(squirrel.Eq{
		"service_name": req.ServiceName,
	}))
	if err != nil {
		l.Logger.Errorf("find inference: %s err: %s", req.ServiceName, err)
		return nil, ErrDBFind
	}

	// 检查是否是该用户的任务
	if infer.NewUserId != req.UserID {
		return nil, ErrPermissionDenied
	}

	// 查询推理任务状态并返回
	resp = &types.InferenceStatusResp{}
	resp.ServiceName = infer.ServiceName
	resp.Status = model.StatusToStr[infer.Status]
	if infer.Url != "" {
		resp.ChatURL = fmt.Sprintf("%s%s", l.svcCtx.Config.K8S.BaseUrl, infer.Url)
		resp.APIURL = fmt.Sprintf("%s/inference/api/%s/v1/chat/completions", l.svcCtx.Config.K8S.BaseUrl, infer.ServiceName)
	}

	return
}
