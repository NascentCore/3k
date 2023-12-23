package logic

import (
	"testing"

	"sxwl/3k/gomicro/scheduler/internal/svc"
	"sxwl/3k/gomicro/scheduler/internal/types"
)

func TestUploadStatusLogic_UploadStatus(t *testing.T) {
	ctx := svc.ServiceContext{}
	uploads := UploadStatusLogic{
		svcCtx: &ctx,
	}
	modelurlReq := types.ModelUrlReq{
		DownloadUrls: []string{},
	}
	_, err := uploads.UploadStatus(&modelurlReq)
	if err != nil {
		t.Fail()
	}
}
