package logic

import (
	"context"

	"sxwl/3k/internal/scheduler/model"
	"sxwl/3k/internal/scheduler/svc"
	"sxwl/3k/internal/scheduler/types"

	"github.com/Masterminds/squirrel"
	"github.com/zeromicro/go-zero/core/logx"
)

type ClusterCpodNamePutLogic struct {
	logx.Logger
	ctx    context.Context
	svcCtx *svc.ServiceContext
}

func NewClusterCpodNamePutLogic(ctx context.Context, svcCtx *svc.ServiceContext) *ClusterCpodNamePutLogic {
	return &ClusterCpodNamePutLogic{
		Logger: logx.WithContext(ctx),
		ctx:    ctx,
		svcCtx: svcCtx,
	}
}

func (l *ClusterCpodNamePutLogic) ClusterCpodNamePut(req *types.ClusterCpodNamePutReq) (resp *types.BaseResp, err error) {
	UserModel := l.svcCtx.UserModel
	CpodNodeModel := l.svcCtx.CpodNodeModel

	// Check if the user is an admin
	isAdmin, err := UserModel.IsAdmin(l.ctx, req.UserID)
	if err != nil {
		l.Errorf("UserModel.IsAdmin userID=%s err=%s", req.UserID, err)
		return nil, ErrNotAdmin
	}
	if !isAdmin {
		l.Infof("UserModel.IsAdmin userID=%s is not admin", req.UserID)
		return nil, ErrNotAdmin
	}

	// check if the cpod exists
	_, err = CpodNodeModel.FindOneByQuery(l.ctx, CpodNodeModel.AllFieldsBuilder().Where(squirrel.Eq{
		"cpod_id": req.CpodID,
	}))
	if err != nil && err != model.ErrNotFound {
		return nil, ErrCpodNotFound
	}
	if err != nil {
		l.Errorf("CpodNodeModel.FindOneByCond cpodID=%s err=%s", req.CpodID, err)
		return nil, ErrDBFind
	}

	// update the cpod name
	builder := CpodNodeModel.UpdateBuilder().Where(squirrel.Eq{
		"cpod_id": req.CpodID,
	}).Set("cpod_name", req.CpodName)
	_, err = CpodNodeModel.UpdateColsByCond(l.ctx, builder)
	if err != nil {
		l.Errorf("CpodModel.UpdateCpodName cpodID=%s name=%s err=%s", req.CpodID, req.CpodName, err)
		return nil, ErrDB
	}

	return
}
