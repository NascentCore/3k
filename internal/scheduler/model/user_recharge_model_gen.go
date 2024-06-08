// Code generated by goctl. DO NOT EDIT.

package model

import (
	"context"
	"database/sql"
	"fmt"
	"strings"
	"time"

	"github.com/zeromicro/go-zero/core/stores/builder"
	"github.com/zeromicro/go-zero/core/stores/sqlc"
	"github.com/zeromicro/go-zero/core/stores/sqlx"
	"github.com/zeromicro/go-zero/core/stringx"
)

var (
	userRechargeFieldNames          = builder.RawFieldNames(&UserRecharge{})
	userRechargeRows                = strings.Join(userRechargeFieldNames, ",")
	userRechargeRowsExpectAutoSet   = strings.Join(stringx.Remove(userRechargeFieldNames, "`id`", "`create_at`", "`create_time`", "`created_at`", "`update_at`", "`update_time`", "`updated_at`"), ",")
	userRechargeRowsWithPlaceHolder = strings.Join(stringx.Remove(userRechargeFieldNames, "`id`", "`create_at`", "`create_time`", "`created_at`", "`update_at`", "`update_time`", "`updated_at`"), "=?,") + "=?"
)

type (
	userRechargeModel interface {
		Insert(ctx context.Context, data *UserRecharge) (sql.Result, error)
		FindOne(ctx context.Context, id int64) (*UserRecharge, error)
		Update(ctx context.Context, data *UserRecharge) error
		Delete(ctx context.Context, id int64) error
	}

	defaultUserRechargeModel struct {
		conn  sqlx.SqlConn
		table string
	}

	UserRecharge struct {
		Id            int64        `db:"id"`             // ID
		RechargeId    string       `db:"recharge_id"`    // 充值记录id
		UserId        string       `db:"user_id"`        // 用户ID
		Amount        float64      `db:"amount"`         // 充值金额
		BeforeBalance float64      `db:"before_balance"` // 充值前余额
		AfterBalance  float64      `db:"after_balance"`  // 充值后余额
		Description   string       `db:"description"`    // 描述
		CreatedAt     time.Time    `db:"created_at"`     // 创建时间
		UpdatedAt     time.Time    `db:"updated_at"`     // 更新时间
		DeletedAt     sql.NullTime `db:"deleted_at"`     // 删除时间
	}
)

func newUserRechargeModel(conn sqlx.SqlConn) *defaultUserRechargeModel {
	return &defaultUserRechargeModel{
		conn:  conn,
		table: "`user_recharge`",
	}
}

func (m *defaultUserRechargeModel) Delete(ctx context.Context, id int64) error {
	query := fmt.Sprintf("delete from %s where `id` = ?", m.table)
	_, err := m.conn.ExecCtx(ctx, query, id)
	return err
}

func (m *defaultUserRechargeModel) FindOne(ctx context.Context, id int64) (*UserRecharge, error) {
	query := fmt.Sprintf("select %s from %s where `id` = ? limit 1", userRechargeRows, m.table)
	var resp UserRecharge
	err := m.conn.QueryRowCtx(ctx, &resp, query, id)
	switch err {
	case nil:
		return &resp, nil
	case sqlc.ErrNotFound:
		return nil, ErrNotFound
	default:
		return nil, err
	}
}

func (m *defaultUserRechargeModel) Insert(ctx context.Context, data *UserRecharge) (sql.Result, error) {
	query := fmt.Sprintf("insert into %s (%s) values (?, ?, ?, ?, ?, ?, ?)", m.table, userRechargeRowsExpectAutoSet)
	ret, err := m.conn.ExecCtx(ctx, query, data.RechargeId, data.UserId, data.Amount, data.BeforeBalance, data.AfterBalance, data.Description, data.DeletedAt)
	return ret, err
}

func (m *defaultUserRechargeModel) Update(ctx context.Context, data *UserRecharge) error {
	query := fmt.Sprintf("update %s set %s where `id` = ?", m.table, userRechargeRowsWithPlaceHolder)
	_, err := m.conn.ExecCtx(ctx, query, data.RechargeId, data.UserId, data.Amount, data.BeforeBalance, data.AfterBalance, data.Description, data.DeletedAt, data.Id)
	return err
}

func (m *defaultUserRechargeModel) tableName() string {
	return m.table
}