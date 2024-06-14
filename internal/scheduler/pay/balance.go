package pay

import (
	"context"
	"errors"
	"sxwl/3k/internal/scheduler/logic"
	"sxwl/3k/internal/scheduler/model"
	"sxwl/3k/internal/scheduler/svc"
	"sxwl/3k/internal/scheduler/types"
	"sxwl/3k/pkg/orm"
	"time"

	"github.com/Masterminds/squirrel"
	"github.com/zeromicro/go-zero/core/logx"
	"github.com/zeromicro/go-zero/core/stores/sqlx"
)

type BalanceManager struct {
	ctx context.Context
	logx.Logger
	svcCtx *svc.ServiceContext
}

func NewBalanceManager(svcCtx *svc.ServiceContext) *BalanceManager {
	ctx := context.Background()
	return &BalanceManager{
		ctx:    ctx,
		Logger: logx.WithContext(ctx),
		svcCtx: svcCtx,
	}
}

func (bm *BalanceManager) Update() {
	if !bm.svcCtx.Config.Billing.CronBalance {
		return
	}

	BillingModel := bm.svcCtx.UserBillingModel
	BalanceModel := bm.svcCtx.UserBalanceModel

	billingList, err := BillingModel.Find(bm.ctx, BillingModel.AllFieldsBuilder().Where(squirrel.Eq{
		"billing_status": model.BillingStatusUnpaid,
	}))
	if err != nil {
		bm.Errorf("billing Find err=%s", err)
		return
	}

	// 按用户聚合
	userAmountMap := make(map[string]float64)
	userBillingMap := make(map[string][]int64)
	for _, billing := range billingList {
		_, ok := userAmountMap[billing.NewUserId]
		if !ok {
			userAmountMap[billing.NewUserId] = billing.Amount
			userBillingMap[billing.NewUserId] = []int64{billing.Id}
			continue
		}

		userAmountMap[billing.NewUserId] += billing.Amount
		userBillingMap[billing.NewUserId] = append(userBillingMap[billing.NewUserId], billing.Id)
	}

	for newUserID, amount := range userAmountMap {
		err = bm.svcCtx.DB.Transact(func(session sqlx.Session) error {
			// 扣减用户余额
			sql, args, err := BalanceModel.UpdateBuilder().Where(squirrel.Eq{
				"new_user_id": newUserID,
			}).Set("balance", squirrel.Expr("balance - ?", amount)).ToSql()
			if err != nil {
				return err
			}

			result, err := session.Exec(sql, args...)
			if err != nil {
				return err
			}
			rowsAffected, err := result.RowsAffected()
			if err != nil {
				return err
			}
			if rowsAffected == 0 {
				bm.Errorf("user %s userBalance not found", newUserID)
			}

			// 设置账单状态为已支付
			sql, args, err = BillingModel.UpdateBuilder().Where(squirrel.Eq{
				"id": userBillingMap[newUserID],
			}).SetMap(map[string]interface{}{
				"billing_status": model.BillingStatusPaid,
				"payment_time":   orm.NullTime(time.Now()),
			}).ToSql()
			if err != nil {
				return err
			}

			result, err = session.Exec(sql, args...)
			if err != nil {
				return err
			}
			rowsAffected, err = result.RowsAffected()
			if err != nil {
				return err
			}
			if rowsAffected == 0 {
				return errors.New("billing not found or already paid")
			}

			return nil
		})
		if err != nil {
			bm.Errorf("Transact err=%s", err)
		}
	}

	// 用户欠费，终止任务
	userIDList := make([]string, 0)
	for userID := range userAmountMap {
		userIDList = append(userIDList, userID)
	}

	userBalanceList, err := BalanceModel.Find(bm.ctx, BalanceModel.AllFieldsBuilder().Where(squirrel.Eq{
		"new_user_id": userIDList,
	}))
	if err != nil {
		bm.Errorf("userBalance Find err=%s", err)
		return
	}

	for _, userBalance := range userBalanceList {
		if userBalance.Balance < 0.0 {
			// 欠费，终止任务
			_, err = logic.NewJobsDelLogic(bm.ctx, bm.svcCtx).JobsDel(&types.JobsDelReq{
				ToUser: userBalance.NewUserId,
			})
			if err != nil {
				bm.Errorf("user overdraw delete jobs user_id=%s err=%s", userBalance.NewUserId, err)
			}
		}
	}
}
