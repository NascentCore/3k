package logic

import (
	"context"
	"sxwl/3k/internal/scheduler/model"

	"sxwl/3k/internal/scheduler/svc"
	"sxwl/3k/internal/scheduler/types"

	"github.com/Masterminds/squirrel"
	"github.com/zeromicro/go-zero/core/logx"
)

type QuotaListLogic struct {
	logx.Logger
	ctx    context.Context
	svcCtx *svc.ServiceContext
}

func NewQuotaListLogic(ctx context.Context, svcCtx *svc.ServiceContext) *QuotaListLogic {
	return &QuotaListLogic{
		Logger: logx.WithContext(ctx),
		ctx:    ctx,
		svcCtx: svcCtx,
	}
}

func (l *QuotaListLogic) QuotaList(req *types.QuotaListReq) (resp *types.QuotaListResp, err error) {
	QuotaModel := l.svcCtx.QuotaModel

	var quotaList []*model.SysQuota
	if req.UserId != 0 {
		quotaList, err = QuotaModel.Find(l.ctx, QuotaModel.AllFieldsBuilder().Where(squirrel.Eq{
			"user_id": req.UserId,
		}))
		if err != nil {
			l.Errorf("QuotaList Find user_id=%d err=%s", req.UserId, err)
			return nil, err
		}
	} else {
		quotaList, err = QuotaModel.FindAll(l.ctx, "")
		if err != nil {
			l.Errorf("QuotaList FindAll user_id=%d err=%s", req.UserId, err)
			return nil, err
		}
	}

	resp = &types.QuotaListResp{
		Data: make([]types.QuotaResp, 0),
	}
	for _, quota := range quotaList {
		resp.Data = append(resp.Data, types.QuotaResp{
			Id: quota.Id,
			Quota: types.Quota{
				UserId:   quota.UserId,
				Resource: quota.Resource,
				Quota:    quota.Quota,
			},
		})
	}

	return
}
