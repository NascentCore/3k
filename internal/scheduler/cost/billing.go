package cost

import (
	"context"
	"fmt"
	"math"
	"sxwl/3k/internal/scheduler/model"
	"sxwl/3k/internal/scheduler/svc"
	"sxwl/3k/pkg/orm"
	"sxwl/3k/pkg/time"
	"sxwl/3k/pkg/uuid"

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

	JobModel := bm.svcCtx.UserJobModel
	InferenceModel := bm.svcCtx.InferenceModel
	PriceModel := bm.svcCtx.PriceModel
	JupyterlabModel := bm.svcCtx.JupyterlabModel
	BillingModel := bm.svcCtx.UserBillingModel

	billingList := make([]model.UserBilling, 0)
	billingTime := time.GetNearestMinute()
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
	jobs, err := JobModel.Find(bm.ctx, JobModel.AllFieldsBuilder().Where(squirrel.Eq{
		"work_status": model.JobStatusWorkerRunning,
		"deleted":     model.JobValid,
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

		// 生成账单
		amount := roundToTwoDecimalPlaces(price * float64(job.GpuNumber.Int64) / 60)
		billingID, err := newBillingID()
		if err != nil {
			bm.Errorf("newBillingID prod=%s err=%s", job.GpuType.String, err)
			return
		}
		billingList = append(billingList, model.UserBilling{
			BillingId:     billingID,
			UserId:        job.UserId,
			Amount:        amount,
			BillingStatus: model.BillingStatusUnpaid,
			JobId:         job.JobName.String,
			JobType:       model.BillingTypeTraining,
			BillingTime:   billingTime,
			Description: orm.NullString(fmt.Sprintf(`{"GPUProd": %s, "GPUCount": %d, "Amount": %f}`,
				job.GpuType.String,
				job.GpuNumber.Int64,
				amount)),
		})
	}

	// inference job
	infers, err := InferenceModel.FindAll(bm.ctx, InferenceModel.AllFieldsBuilder().Where(squirrel.Eq{
		"status": model.InferStatusDeployed,
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

		// 生成账单
		amount := roundToTwoDecimalPlaces(price * float64(infer.GpuNumber.Int64) / 60)
		billingID, err := newBillingID()
		if err != nil {
			bm.Errorf("newBillingID prod=%s err=%s", infer.GpuType.String, err)
			return
		}
		billingList = append(billingList, model.UserBilling{
			BillingId:     billingID,
			UserId:        infer.UserId,
			Amount:        amount,
			BillingStatus: model.BillingStatusUnpaid,
			JobId:         infer.ServiceName,
			JobType:       model.BillingTypeInference,
			BillingTime:   billingTime,
			Description: orm.NullString(fmt.Sprintf(`{"GPUProd": %s, "GPUCount": %d, "Amount": %f}`,
				infer.GpuType.String,
				infer.GpuNumber.Int64,
				amount)),
		})
	}

	// jupyterlab
	jupyters, err := JupyterlabModel.Find(bm.ctx, JupyterlabModel.AllFieldsBuilder().Where(squirrel.Eq{
		"status": model.JupyterStatusDeployed,
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

		// 生成账单
		amount := roundToTwoDecimalPlaces(price * float64(jupyter.GpuCount) / 60)
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
			JobType:       model.BillingTypeInference,
			BillingTime:   billingTime,
			Description: orm.NullString(fmt.Sprintf(`{"GPUProd": %s, "GPUCount": %d, "Amount": %.2f}`,
				jupyter.GpuProd,
				jupyter.GpuCount,
				amount)),
		})
	}

	// insert into user_billing
	insertBuilder := BillingModel.InsertBuilder().Columns(
		"billing_id",
		"user_id",
		"amount",
		"billing_status",
		"job_id",
		"job_type",
		"billing_time",
		"description",
	)
	for _, billing := range billingList {
		insertBuilder = insertBuilder.Values(
			billing.BillingId,
			billing.UserId,
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
		return
	}

	_, err = bm.svcCtx.DB.Exec(query, args...)
	if err != nil {
		bm.Errorf("insert exec err=%s", err)
		return
	}
}

// Compensate 检查是否有某些任务有缺失的账单，例如scheduler宕机时漏掉的账单
func (bm *BillingManager) Compensate() {
	if !bm.svcCtx.Config.Billing.CronBilling {
		return
	}

	// TODO
}

func roundToTwoDecimalPlaces(f float64) float64 {
	return math.Round(f*100) / 100
}

func newBillingID() (string, error) {
	return uuid.UUIDWithPrefix("bill")
}
