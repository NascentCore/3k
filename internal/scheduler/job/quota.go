package job

import (
	"context"
	"errors"
	"sxwl/3k/internal/scheduler/model"
	"sxwl/3k/internal/scheduler/svc"

	"github.com/Masterminds/squirrel"
)

func CheckQuota(ctx context.Context, svx *svc.ServiceContext, userID string, resource string, need int64) (ok bool, left int64, err error) {
	QuotaModel := svx.QuotaModel
	UserJobModel := svx.UserJobModel
	InferenceModel := svx.InferenceModel

	// check user has quota
	quota, err := QuotaModel.FindOneByQuery(ctx, QuotaModel.AllFieldsBuilder().Where(squirrel.Eq{
		"new_user_id": userID,
		"resource":    resource,
	}))
	if err != nil {
		if errors.Is(err, model.ErrNotFound) {
			return true, 0, nil // 没有配额记录，不需要检查
		} else {
			return false, 0, err
		}
	}

	// check quota left
	jobs, err := UserJobModel.Find(ctx, UserJobModel.AllFieldsBuilder().Where(squirrel.Eq{
		"new_user_id": userID,
		"work_status": model.JobStatusWorkerRunning,
		"gpu_type":    resource,
		"deleted":     0,
	}))
	if err != nil {
		return false, 0, err
	}

	infers, err := InferenceModel.FindAll(ctx, InferenceModel.AllFieldsBuilder().Where(squirrel.And{
		squirrel.Eq{
			"new_user_id": userID,
			"gpu_type":    resource,
		},
		squirrel.NotEq{
			"status": model.InferStatusStopped,
		},
	}), "")
	if err != nil {
		return false, 0, err
	}

	var sum int64
	for _, j := range jobs {
		sum += j.GpuNumber.Int64
	}
	for _, i := range infers {
		sum += i.GpuNumber.Int64
	}

	if sum+need > quota.Quota {
		return false, quota.Quota - sum, nil
	}

	return true, quota.Quota - sum, nil
}
