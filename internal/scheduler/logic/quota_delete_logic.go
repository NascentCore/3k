package logic

import (
	"context"

	"sxwl/3k/internal/scheduler/svc"
	"sxwl/3k/internal/scheduler/types"

	"github.com/zeromicro/go-zero/core/logx"
)

type QuotaDeleteLogic struct {
	logx.Logger
	ctx    context.Context
	svcCtx *svc.ServiceContext
}

func NewQuotaDeleteLogic(ctx context.Context, svcCtx *svc.ServiceContext) *QuotaDeleteLogic {
	return &QuotaDeleteLogic{
		Logger: logx.WithContext(ctx),
		ctx:    ctx,
		svcCtx: svcCtx,
	}
}

func (l *QuotaDeleteLogic) QuotaDelete(req *types.QuotaDeleteReq) (resp *types.QuotaDeleteResp, err error) {
	QuotaModel := l.svcCtx.QuotaModel
	err = QuotaModel.Delete(l.ctx, req.Id)
	if err != nil {
		l.Errorf("QuotaDelete id=%d user_id=%d err=%s", req.Id, req.UserID, err)
		return nil, err
	}

	return &types.QuotaDeleteResp{Message: "ok"}, nil
}
