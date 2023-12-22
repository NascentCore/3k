package logic

import (
	"context"

	"github.com/Masterminds/squirrel"

	"sxwl/3k/gomicro/scheduler/internal/svc"
	"sxwl/3k/gomicro/scheduler/internal/types"

	"github.com/zeromicro/go-zero/core/logx"
)

type CpodJobLogic struct {
	logx.Logger
	ctx    context.Context
	svcCtx *svc.ServiceContext
}

func NewCpodJobLogic(ctx context.Context, svcCtx *svc.ServiceContext) *CpodJobLogic {
	return &CpodJobLogic{
		Logger: logx.WithContext(ctx),
		ctx:    ctx,
		svcCtx: svcCtx,
	}
}

func (l *CpodJobLogic) CpodJob(req *types.CPODJobReq) (resp []types.CPODJobResp, err error) {
	CpodMainModel := l.svcCtx.CpodMainModel
	//UserJobModel := l.svcCtx.UserJobModel

	_, err = CpodMainModel.Find(l.ctx, CpodMainModel.AllFieldsBuilder().Where(squirrel.Eq{
		"cpod_id": req.CPODID,
	}))
	if err != nil {
		l.Logger.Errorf("cpod_main find cpod_id=%s err=%s", req.CPODID, err)
		return nil, err
	}

	//UserJobModel.Find(l.ctx, UserJobModel.AllFieldsBuilder().Where(squirrel.Eq{
	//	""
	//}))

	return
}
