package synchronizer

import (
	"context"
	"fmt"
	"time"

	"github.com/NascentCore/cpodoperator/pkg/provider/sxwl"

	"github.com/go-logr/logr"
)

// Uploader定时上报心跳信息, 数据来自Ch
type Uploader struct {
	ch            <-chan sxwl.HeartBeatPayload
	scheduler     sxwl.Scheduler
	cachedPayload sxwl.HeartBeatPayload
	logger        logr.Logger
}

func NewUploader(ch <-chan sxwl.HeartBeatPayload, scheduler sxwl.Scheduler, interval time.Duration, logger logr.Logger) *Uploader {
	return &Uploader{
		ch:        ch,
		scheduler: scheduler,
		logger:    logger,
	}
}

func (u *Uploader) Start(ctx context.Context) {
	u.logger.Info("uploader")
	select {
	case u.cachedPayload = <-u.ch:
	case <-ctx.Done():
		u.logger.Info("uploader stopped")
		return
	default:
		u.logger.Info("no new data")
	}
	if u.cachedPayload.CPodID == "" {
		u.logger.Info("no data to upload")
		return
	}
	u.logger.Info(fmt.Sprintf("data updated at %d seconds ago", int(time.Now().Sub(u.cachedPayload.UpdateTime).Seconds())))
	u.logger.Info("ready to upload", "payload", u.cachedPayload)
	err := u.scheduler.HeartBeat(u.cachedPayload)
	if err != nil {
		u.logger.Error(err, "upload cpod status data failed")
	} else {
		u.logger.Info("uploaded cpod status data")
	}
}
