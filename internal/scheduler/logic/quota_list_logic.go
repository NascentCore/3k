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
	UserModel := l.svcCtx.UserModel
	QuotaModel := l.svcCtx.QuotaModel

	isAdmin, err := UserModel.IsAdmin(l.ctx, req.UserID)
	if err != nil {
		l.Errorf("UserModel.IsAdmin userID=%s err=%s", req.UserID, err)
		return nil, ErrNotAdmin
	}
	if !isAdmin {
		l.Infof("UserModel.IsAdmin userID=%s is not admin", req.UserID)
		return nil, ErrNotAdmin
	}

	var quotaList []*model.SysQuota
	if req.ToUser != "" {
		quotaList, err = QuotaModel.Find(l.ctx, QuotaModel.AllFieldsBuilder().Where(squirrel.Eq{
			"new_user_id": req.ToUser,
		}))
		if err != nil {
			l.Errorf("QuotaList Find user_id=%s err=%s", req.ToUser, err)
			return nil, err
		}
	} else {
		quotaList, err = QuotaModel.FindAll(l.ctx, "")
		if err != nil {
			l.Errorf("QuotaList FindAll user_id=%s err=%s", req.UserID, err)
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
				UserId:   quota.NewUserId,
				Resource: quota.Resource,
				Quota:    quota.Quota,
			},
		})
	}

	return
}
