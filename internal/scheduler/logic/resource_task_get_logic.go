package logic

import (
	"context"
	"fmt"
	"strings"

	"sxwl/3k/internal/scheduler/model"
	"sxwl/3k/internal/scheduler/svc"
	"sxwl/3k/internal/scheduler/types"

	"github.com/zeromicro/go-zero/core/logx"
	"github.com/zeromicro/go-zero/core/stores/sqlx"
)

type ResourceTaskGetLogic struct {
	logx.Logger
	ctx    context.Context
	svcCtx *svc.ServiceContext
}

func NewResourceTaskGetLogic(ctx context.Context, svcCtx *svc.ServiceContext) *ResourceTaskGetLogic {
	return &ResourceTaskGetLogic{
		Logger: logx.WithContext(ctx),
		ctx:    ctx,
		svcCtx: svcCtx,
	}
}

func (l *ResourceTaskGetLogic) ResourceTaskGet(req *types.BaseReq) (resp *types.ResourceSyncTaskResp, err error) {
	UserModel := l.svcCtx.UserModel

	// 1. 检查管理员权限
	isAdmin, err := UserModel.IsAdmin(l.ctx, req.UserID)
	if err != nil {
		l.Errorf("UserModel.IsAdmin userID=%s err=%s", req.UserID, err)
		return nil, ErrNotAdmin
	}
	if !isAdmin {
		l.Infof("UserModel.IsAdmin userID=%s is not admin", req.UserID)
		return nil, ErrNotAdmin
	}

	var tasks []*model.ResourceSyncTask

	// 2. 使用事务和SELECT FOR UPDATE来确保任务分配的原子性
	err = l.svcCtx.DB.TransactCtx(l.ctx, func(ctx context.Context, session sqlx.Session) error {
		// 使用 SELECT FOR UPDATE 锁定要处理的记录
		query := `SELECT * FROM resource_sync_task 
			WHERE status = ? 
			LIMIT 100 
			FOR UPDATE`

		err := session.QueryRowsCtx(ctx, &tasks, query, model.ResourceSyncTaskStatusGettingMetaDone)
		if err == sqlx.ErrNotFound {
			return nil
		}
		if err != nil {
			return err
		}

		// 批量更新状态
		if len(tasks) > 0 {
			var placeholders []string
			var args []interface{}

			for _, task := range tasks {
				placeholders = append(placeholders, "?")
				args = append(args, task.Id)
			}

			updateQuery := fmt.Sprintf(`UPDATE resource_sync_task 
				SET status = ?, executor_id = ? 
				WHERE id IN (%s)`,
				strings.Join(placeholders, ","))

			args = append([]interface{}{
				model.ResourceSyncTaskStatusTransfering,
				req.UserID,
			}, args...)

			result, err := session.ExecCtx(ctx, updateQuery, args...)
			if err != nil {
				return err
			}

			affected, err := result.RowsAffected()
			if err != nil {
				return err
			}
			if affected == 0 {
				return fmt.Errorf("no tasks were updated")
			}
		}

		return nil
	})

	if err != nil {
		l.Errorf("Transaction failed: %s", err)
		return nil, ErrDB
	}

	// 3. 构造返回数据
	resp = &types.ResourceSyncTaskResp{
		Data: make([]types.ResourceSyncTask, 0, len(tasks)),
	}
	for _, task := range tasks {
		resp.Data = append(resp.Data, types.ResourceSyncTask{
			ResourceID:   task.ResourceId,
			ResourceType: task.ResourceType,
			Source:      task.Source,
		})
	}
	resp.Total = int64(len(resp.Data))

	return resp, nil
}
