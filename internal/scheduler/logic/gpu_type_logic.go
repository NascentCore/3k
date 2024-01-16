package logic

import (
	"context"

	"sxwl/3k/internal/scheduler/svc"
	"sxwl/3k/internal/scheduler/types"

	"github.com/zeromicro/go-zero/core/logx"
)

type GpuTypeLogic struct {
	logx.Logger
	ctx    context.Context
	svcCtx *svc.ServiceContext
}

func NewGpuTypeLogic(ctx context.Context, svcCtx *svc.ServiceContext) *GpuTypeLogic {
	return &GpuTypeLogic{
		Logger: logx.WithContext(ctx),
		ctx:    ctx,
		svcCtx: svcCtx,
	}
}

func (l *GpuTypeLogic) GpuType(req *types.GPUTypeReq) (resp []types.GPUTypeResp, err error) {
	CpodMainModel := l.svcCtx.CpodMainModel

	gpuPrices, err := CpodMainModel.GpuTypeAndPrice(l.ctx)
	if err != nil {
		return nil, err
	}

	resp = []types.GPUTypeResp{}
	for _, gpuPrice := range gpuPrices {
		resp = append(resp, types.GPUTypeResp{
			Amount:  gpuPrice.Amount,
			GPUProd: gpuPrice.GPUProd,
		})
	}

	return resp, nil
}
