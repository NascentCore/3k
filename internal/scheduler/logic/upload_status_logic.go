package logic

import (
	"context"
	"database/sql"
	"sxwl/3k/internal/scheduler/model"
	"time"

	"sxwl/3k/internal/scheduler/svc"
	"sxwl/3k/internal/scheduler/types"

	"github.com/zeromicro/go-zero/core/logx"
)

type UploadStatusLogic struct {
	logx.Logger
	ctx    context.Context
	svcCtx *svc.ServiceContext
}

func NewUploadStatusLogic(ctx context.Context, svcCtx *svc.ServiceContext) *UploadStatusLogic {
	return &UploadStatusLogic{
		Logger: logx.WithContext(ctx),
		ctx:    ctx,
		svcCtx: svcCtx,
	}
}

func (l *UploadStatusLogic) UploadStatus(req *types.ModelUrlReq) (resp *types.ModelUrlResp, err error) {
	// todo: add your logic here and delete this line
	FileURLModel := l.svcCtx.FileURLModel
	for _, url := range req.DownloadUrls {
		modeUrl := model.SysFileurl{
			JobName:    req.JobName,
			FileUrl:    sql.NullString{String: url, Valid: true},
			CreateTime: sql.NullTime{Time: time.Now(), Valid: true},
			UpdateTime: sql.NullTime{Time: time.Now(), Valid: true},
		}
		_, err := FileURLModel.Insert(l.ctx, &modeUrl)
		if err != nil {
			l.Logger.Errorf("modeUrl insert err=%s", err)
			return nil, err
		}
		l.Logger.Infof("modeUrl insert sys_fileurl job_name=%s file_url=%s err=%s", req.JobName, url, err)
	}
	return
}
