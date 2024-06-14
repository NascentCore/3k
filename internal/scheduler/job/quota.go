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
	JupyterModel := svx.JupyterlabModel

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

	// job used quota
	jobs, err := UserJobModel.Find(ctx, UserJobModel.AllFieldsBuilder().Where(squirrel.Eq{
		"new_user_id": userID,
		"work_status": model.StatusRunning,
		"gpu_type":    resource,
		"deleted":     0,
	}))
	if err != nil {
		return false, 0, err
	}

	// inference used quota
	infers, err := InferenceModel.FindAll(ctx, InferenceModel.AllFieldsBuilder().Where(squirrel.And{
		squirrel.Eq{
			"new_user_id": userID,
			"gpu_type":    resource,
		},
		squirrel.Eq{
			"status": model.StatusRunning,
		},
	}), "")
	if err != nil {
		return false, 0, err
	}

	// jupyterlab used quota
	jupyters, err := JupyterModel.Find(ctx, JupyterModel.AllFieldsBuilder().Where(squirrel.And{
		squirrel.Eq{
			"new_user_id": userID,
			"gpu_prod":    resource,
		},
		squirrel.Eq{
			"status": model.StatusRunning,
		},
	}))
	if err != nil {
		return false, 0, err
	}

	var sum int64
	for _, i := range jobs {
		sum += i.GpuNumber.Int64
	}
	for _, i := range infers {
		sum += i.GpuNumber.Int64
	}
	for _, i := range jupyters {
		sum += i.GpuCount
	}

	if sum+need > quota.Quota {
		return false, quota.Quota - sum, nil
	}

	return true, quota.Quota - sum, nil
}
