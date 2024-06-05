package pay

import (
	"context"
	"errors"
	"fmt"
	"math"
	"sxwl/3k/internal/scheduler/model"
	"sxwl/3k/internal/scheduler/svc"
	"sxwl/3k/pkg/orm"
	"sxwl/3k/pkg/time"
	"sxwl/3k/pkg/uuid"
	time2 "time"

	"github.com/Masterminds/squirrel"
	"github.com/zeromicro/go-zero/core/logx"
)

type BillingManager struct {
	ctx context.Context
	logx.Logger
	svcCtx *svc.ServiceContext
}

func NewBillingManager(svcCtx *svc.ServiceContext) *BillingManager {
	ctx := context.Background()
	return &BillingManager{
		ctx:    ctx,
		Logger: logx.WithContext(ctx),
		svcCtx: svcCtx,
	}
}

func (bm *BillingManager) Update() {
	if !bm.svcCtx.Config.Billing.CronBilling {
		return
	}

	PriceModel := bm.svcCtx.PriceModel
	JobModel := bm.svcCtx.UserJobModel
	InferenceModel := bm.svcCtx.InferenceModel
	JupyterlabModel := bm.svcCtx.JupyterlabModel
	UserBillingModel := bm.svcCtx.UserBillingModel

	billingList := make([]model.UserBilling, 0)
	prices := make(map[string]float64)

	// prices
	priceList, err := PriceModel.FindAll(bm.ctx, "")
	if err != nil {
		bm.Errorf("price FindAll err=%s", err)
		return
	}
	for _, price := range priceList {
		prices[price.GpuProd.String] = price.Amount.Float64
	}

	// training job
	completeJobs := make([]int64, 0)
	jobs, err := JobModel.Find(bm.ctx, JobModel.AllFieldsBuilder().Where(squirrel.Eq{
		"billing_status": model.BillingStatusContinue, // 未结清账单的任务
		"deleted":        model.JobValid,
	}))
	if err != nil {
		bm.Errorf("job find err=%s", err)
		return
	}

	for _, job := range jobs {
		price, ok := prices[job.GpuType.String]
		if !ok {
			bm.Errorf("price gpu %s not exists", job.GpuType.String)
			price = 5.0 // default price
		}

		startTime := time.GetNearestMinute(job.CreateTime.Time)
		endTime := time.GetNearestMinute(job.UpdateTime.Time)
		if job.WorkStatus == model.JobStatusWorkerRunning {
			endTime = time.GetNearestMinute(time2.Now())
		}
		lastBilling, err := UserBillingModel.FindOneByQuery(bm.ctx, UserBillingModel.AllFieldsBuilder().Where(squirrel.Eq{
			"job_id": job.JobName.String,
		}).OrderBy("billing_time DESC"))
		if err != nil && !errors.Is(err, model.ErrNotFound) {
			bm.Errorf("userBilling FindOne err=%s", err)
			continue
		}
		if lastBilling != nil {
			startTime = time.GetNearestMinute(lastBilling.BillingTime.Add(time2.Minute))
		}

		// 生成账单
		amount := roundToTwoDecimalPlaces(price * float64(job.GpuNumber.Int64) / 60)
		for t := startTime; t.Before(endTime); t = t.Add(time2.Minute) {
			billingID, err := newBillingID()
			if err != nil {
				bm.Errorf("newBillingID prod=%s err=%s", job.GpuType.String, err)
				return
			}
			billingList = append(billingList, model.UserBilling{
				BillingId:     billingID,
				NewUserId:     job.NewUserId,
				Amount:        amount,
				BillingStatus: model.BillingStatusUnpaid,
				JobId:         job.JobName.String,
				JobType:       model.BillingTypeTraining,
				BillingTime:   t,
				Description: orm.NullString(fmt.Sprintf(`{"GPUProd": %s, "GPUCount": %d, "Amount": %f}`,
					job.GpuType.String,
					job.GpuNumber.Int64,
					amount)),
			})
		}

		if job.WorkStatus > model.JobStatusWorkerRunning {
			completeJobs = append(completeJobs, job.JobId)
		}
	}

	err = bm.BillCompleteTraining(completeJobs)
	if err != nil {
		bm.Errorf("BillCompleteTraining err=%s", err)
	}

	// inference job
	completeInfers := make([]int64, 0)
	infers, err := InferenceModel.FindAll(bm.ctx, InferenceModel.AllFieldsBuilder().Where(squirrel.Eq{
		"billing_status": model.BillingStatusContinue, // 未结清账单的任务
	}), "")
	if err != nil {
		bm.Errorf("infers FindAll err=%s", err)
		return
	}

	for _, infer := range infers {
		price, ok := prices[infer.GpuType.String]
		if !ok {
			bm.Errorf("price gpu %s not exists", infer.GpuType.String)
			price = 5.0 // default price
		}

		if !infer.StartTime.Valid {
			continue // 没部署成功，跳过
		}

		startTime := time.GetNearestMinute(infer.StartTime.Time)
		endTime := time.GetNearestMinute(infer.EndTime.Time)
		if infer.Status == model.InferStatusDeployed {
			endTime = time.GetNearestMinute(time2.Now())
		}
		lastBilling, err := UserBillingModel.FindOneByQuery(bm.ctx, UserBillingModel.AllFieldsBuilder().Where(squirrel.Eq{
			"job_id": infer.ServiceName,
		}).OrderBy("billing_time DESC"))
		if err != nil && !errors.Is(err, model.ErrNotFound) {
			bm.Errorf("userBilling FindOne err=%s", err)
			continue
		}
		if lastBilling != nil {
			startTime = time.GetNearestMinute(lastBilling.BillingTime.Add(time2.Minute))
		}

		// 生成账单
		amount := roundToTwoDecimalPlaces(price * float64(infer.GpuNumber.Int64) / 60)
		for t := startTime; t.Before(endTime); t = t.Add(time2.Minute) {
			billingID, err := newBillingID()
			if err != nil {
				bm.Errorf("newBillingID prod=%s err=%s", infer.GpuType.String, err)
				return
			}
			billingList = append(billingList, model.UserBilling{
				BillingId:     billingID,
				NewUserId:     infer.NewUserId,
				Amount:        amount,
				BillingStatus: model.BillingStatusUnpaid,
				JobId:         infer.ServiceName,
				JobType:       model.BillingTypeInference,
				BillingTime:   t,
				Description: orm.NullString(fmt.Sprintf(`{"GPUProd": %s, "GPUCount": %d, "Amount": %f}`,
					infer.GpuType.String,
					infer.GpuNumber.Int64,
					amount)),
			})
		}

		if infer.Status > model.InferStatusDeployed {
			completeInfers = append(completeInfers, infer.Id)
		}
	}
	err = bm.BillCompleteInfer(completeInfers)
	if err != nil {
		bm.Errorf("BillCompleteInfer err=%s", err)
	}

	// jupyterlab
	completeJupyters := make([]int64, 0)
	jupyters, err := JupyterlabModel.Find(bm.ctx, JupyterlabModel.AllFieldsBuilder().Where(squirrel.Eq{
		"billing_status": model.BillingStatusContinue,
	}))
	if err != nil {
		bm.Errorf("jupyters FindAll err=%s", err)
		return
	}

	for _, jupyter := range jupyters {
		if jupyter.GpuProd == "" {
			continue // 没有占用GPU的jupyterlab跳过
		}
		price, ok := prices[jupyter.GpuProd]
		if !ok {
			bm.Errorf("price gpu %s not exists", jupyter.GpuProd)
			price = 5.0 // default price
		}

		if !jupyter.StartTime.Valid {
			continue // 没部署成功，跳过
		}

		startTime := time.GetNearestMinute(jupyter.StartTime.Time)
		endTime := time.GetNearestMinute(jupyter.EndTime.Time)
		if jupyter.Status == model.JupyterStatusDeployed {
			endTime = time.GetNearestMinute(time2.Now())
		}
		lastBilling, err := UserBillingModel.FindOneByQuery(bm.ctx, UserBillingModel.AllFieldsBuilder().Where(squirrel.Eq{
			"job_id": jupyter.JobName,
		}).OrderBy("billing_time DESC"))
		if err != nil && !errors.Is(err, model.ErrNotFound) {
			bm.Errorf("userBilling FindOne err=%s", err)
			continue
		}
		if lastBilling != nil {
			startTime = time.GetNearestMinute(lastBilling.BillingTime.Add(time2.Minute))
		}

		// 生成账单
		amount := roundToTwoDecimalPlaces(price * float64(jupyter.GpuCount) / 60)
		for t := startTime; t.Before(endTime); t = t.Add(time2.Minute) {
			billingID, err := newBillingID()
			if err != nil {
				bm.Errorf("newBillingID prod=%s err=%s", jupyter.GpuProd, err)
				return
			}
			billingList = append(billingList, model.UserBilling{
				BillingId:     billingID,
				UserId:        jupyter.UserId,
				Amount:        amount,
				BillingStatus: model.BillingStatusUnpaid,
				JobId:         jupyter.JobName,
				JobType:       model.BillingTypeJupyterlab,
				BillingTime:   t,
				Description: orm.NullString(fmt.Sprintf(`{"GPUProd": %s, "GPUCount": %d, "Amount": %f}`,
					jupyter.GpuProd,
					jupyter.GpuCount,
					amount)),
			})
		}

		if jupyter.Status > model.JupyterStatusDeployed {
			completeJupyters = append(completeJupyters, jupyter.Id)
		}
	}
	err = bm.BillCompleteJupyter(completeJupyters)
	if err != nil {
		bm.Errorf("BillCompleteJupyter err=%s", err)
	}

	// save billing
	err = bm.SaveBilling(billingList)
	if err != nil {
		bm.Errorf("SaveBilling err=%s", err)
		return
	}
}

func (bm *BillingManager) SaveBilling(billings []model.UserBilling) error {
	insertBuilder := bm.svcCtx.UserBillingModel.InsertBuilder().Columns(
		"billing_id",
		"new_user_id",
		"amount",
		"billing_status",
		"job_id",
		"job_type",
		"billing_time",
		"description",
	)
	for _, billing := range billings {
		insertBuilder = insertBuilder.Values(
			billing.BillingId,
			billing.NewUserId,
			billing.Amount,
			billing.BillingStatus,
			billing.JobId,
			billing.JobType,
			billing.BillingTime,
			billing.Description,
		)
	}

	query, args, err := insertBuilder.ToSql()
	if err != nil {
		bm.Errorf("insertBuilder.ToSql err=%s", err)
		return err
	}

	_, err = bm.svcCtx.DB.Exec(query, args...)
	if err != nil {
		bm.Errorf("insert exec err=%s", err)
		return err
	}

	return nil
}

func (bm *BillingManager) BillCompleteTraining(ids []int64) error {
	if len(ids) == 0 {
		return nil
	}

	sql, args, err := bm.svcCtx.UserJobModel.UpdateBuilder().Where(squirrel.Eq{
		"job_id": ids,
	}).Set("billing_status", model.BillingStatusComplete).ToSql()
	if err != nil {
		return err
	}

	result, err := bm.svcCtx.DB.ExecCtx(bm.ctx, sql, args...)
	if err != nil {
		return err
	}
	rowsAffected, err := result.RowsAffected()
	if err != nil {
		return err
	}
	if rowsAffected == 0 {
		return errors.New("jobs not found or already billing complete")
	}

	return nil
}

func (bm *BillingManager) BillCompleteInfer(ids []int64) error {
	if len(ids) == 0 {
		return nil
	}

	sql, args, err := bm.svcCtx.InferenceModel.UpdateBuilder().Where(squirrel.Eq{
		"id": ids,
	}).Set("billing_status", model.BillingStatusComplete).ToSql()
	if err != nil {
		return err
	}

	result, err := bm.svcCtx.DB.ExecCtx(bm.ctx, sql, args...)
	if err != nil {
		return err
	}
	rowsAffected, err := result.RowsAffected()
	if err != nil {
		return err
	}
	if rowsAffected == 0 {
		return errors.New("infers not found or already billing complete")
	}

	return nil
}

func (bm *BillingManager) BillCompleteJupyter(ids []int64) error {
	if len(ids) == 0 {
		return nil
	}

	sql, args, err := bm.svcCtx.JupyterlabModel.UpdateBuilder().Where(squirrel.Eq{
		"id": ids,
	}).Set("billing_status", model.BillingStatusComplete).ToSql()
	if err != nil {
		return err
	}

	result, err := bm.svcCtx.DB.ExecCtx(bm.ctx, sql, args...)
	if err != nil {
		return err
	}
	rowsAffected, err := result.RowsAffected()
	if err != nil {
		return err
	}
	if rowsAffected == 0 {
		return errors.New("jupyters not found or already billing complete")
	}

	return nil
}

func roundToTwoDecimalPlaces(f float64) float64 {
	return math.Round(f*100) / 100
}

func newBillingID() (string, error) {
	return uuid.WithPrefix("bill")
}
