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
	userBalanceFieldNames          = builder.RawFieldNames(&UserBalance{})
	userBalanceRows                = strings.Join(userBalanceFieldNames, ",")
	userBalanceRowsExpectAutoSet   = strings.Join(stringx.Remove(userBalanceFieldNames, "`id`", "`create_at`", "`create_time`", "`created_at`", "`update_at`", "`update_time`", "`updated_at`"), ",")
	userBalanceRowsWithPlaceHolder = strings.Join(stringx.Remove(userBalanceFieldNames, "`id`", "`create_at`", "`create_time`", "`created_at`", "`update_at`", "`update_time`", "`updated_at`"), "=?,") + "=?"
)

type (
	userBalanceModel interface {
		Insert(ctx context.Context, data *UserBalance) (sql.Result, error)
		FindOne(ctx context.Context, id int64) (*UserBalance, error)
		Update(ctx context.Context, data *UserBalance) error
		Delete(ctx context.Context, id int64) error
	}

	defaultUserBalanceModel struct {
		conn  sqlx.SqlConn
		table string
	}

	UserBalance struct {
		Id        int64     `db:"id"`          // ID
		UserId    int64     `db:"user_id"`     // 用户ID
		NewUserId string    `db:"new_user_id"` // 用户ID
		Balance   float64   `db:"balance"`     // 当前余额
		CreatedAt time.Time `db:"created_at"`  // 创建时间
		UpdatedAt time.Time `db:"updated_at"`  // 更新时间
	}
)

func newUserBalanceModel(conn sqlx.SqlConn) *defaultUserBalanceModel {
	return &defaultUserBalanceModel{
		conn:  conn,
		table: "`user_balance`",
	}
}

func (m *defaultUserBalanceModel) Delete(ctx context.Context, id int64) error {
	query := fmt.Sprintf("delete from %s where `id` = ?", m.table)
	_, err := m.conn.ExecCtx(ctx, query, id)
	return err
}

func (m *defaultUserBalanceModel) FindOne(ctx context.Context, id int64) (*UserBalance, error) {
	query := fmt.Sprintf("select %s from %s where `id` = ? limit 1", userBalanceRows, m.table)
	var resp UserBalance
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

func (m *defaultUserBalanceModel) Insert(ctx context.Context, data *UserBalance) (sql.Result, error) {
	query := fmt.Sprintf("insert into %s (%s) values (?, ?, ?)", m.table, userBalanceRowsExpectAutoSet)
	ret, err := m.conn.ExecCtx(ctx, query, data.UserId, data.NewUserId, data.Balance)
	return ret, err
}

func (m *defaultUserBalanceModel) Update(ctx context.Context, data *UserBalance) error {
	query := fmt.Sprintf("update %s set %s where `id` = ?", m.table, userBalanceRowsWithPlaceHolder)
	_, err := m.conn.ExecCtx(ctx, query, data.UserId, data.NewUserId, data.Balance, data.Id)
	return err
}

func (m *defaultUserBalanceModel) tableName() string {
	return m.table
}
