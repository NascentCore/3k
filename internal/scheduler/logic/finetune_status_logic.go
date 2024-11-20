package logic

import (
	"context"
	"encoding/json"
	"fmt"
	"sxwl/3k/internal/scheduler/model"
	"sxwl/3k/internal/scheduler/svc"
	"sxwl/3k/internal/scheduler/types"
	"sxwl/3k/pkg/consts"

	"github.com/Masterminds/squirrel"
	"github.com/zeromicro/go-zero/core/logx"
)

type FinetuneStatusLogic struct {
	logx.Logger
	ctx    context.Context
	svcCtx *svc.ServiceContext
}

func NewFinetuneStatusLogic(ctx context.Context, svcCtx *svc.ServiceContext) *FinetuneStatusLogic {
	return &FinetuneStatusLogic{
		Logger: logx.WithContext(ctx),
		ctx:    ctx,
		svcCtx: svcCtx,
	}
}

func (l *FinetuneStatusLogic) FinetuneStatus(req *types.FinetuneStatusReq) (resp *types.FinetuneStatusResp, err error) {
	UserJobModel := l.svcCtx.UserJobModel
	ResourceModel := l.svcCtx.OssResourceModel

	job, err := UserJobModel.FindOneByQuery(l.ctx, UserJobModel.AllFieldsBuilder().Where(squirrel.Eq{
		"job_name": req.JobId,
	}))
	if err != nil {
		l.Logger.Errorf("find job_id: %s err: %s", req.JobId, err)
		return nil, ErrDBFind
	}

	// 检查是否是该用户的任务
	if job.NewUserId != req.UserID {
		return nil, ErrPermissionDenied
	}

	// 查询任务状态并返回
	resp = &types.FinetuneStatusResp{}
	resp.Status = model.StatusToStr[job.WorkStatus]
	if job.WorkStatus != model.StatusSucceeded {
		return resp, nil
	}

	// 获取生成适配器信息
	jsonAll := map[string]any{}
	err = json.Unmarshal([]byte(job.JsonAll.String), &jsonAll)
	if err != nil {
		l.Logger.Errorf("unmarshal json=%s err=%s", job.JsonAll.String, err)
		return nil, err
	}

	adapterName, ok := jsonAll["trainedModelName"]
	if !ok {
		l.Logger.Errorf("trainedModelName not found in json=%s", job.JsonAll.String)
		return nil, err
	}

	resource, err := ResourceModel.FindOneByQuery(l.ctx, ResourceModel.AllFieldsBuilder().Where(squirrel.Eq{
		"resource_name": fmt.Sprintf("%s/%s", req.UserID, adapterName),
		"resource_type": consts.Adapter,
	}))
	if err != nil {
		l.Logger.Errorf("find resource_name: %s err: %s", adapterName, err)
		return nil, ErrDBFind
	}

	resp.AdapterId = resource.ResourceId
	resp.AdapterName = resource.ResourceName
	resp.AdapterSize = resource.ResourceSize
	resp.AdapterIsPublic = resource.Public == 1

	return resp, nil
}
