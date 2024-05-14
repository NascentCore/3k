package model

import (
	"context"
	"database/sql"
	"time"

	"github.com/Masterminds/squirrel"
	"github.com/zeromicro/go-zero/core/stores/sqlx"
)

var _ UserBalanceModel = (*customUserBalanceModel)(nil)

type (
	// UserBalanceModel is an interface to be customized, add more methods here,
	// and implement the added methods in customUserBalanceModel.
	UserBalanceModel interface {
		userBalanceModel
		AllFieldsBuilder() squirrel.SelectBuilder
		UpdateBuilder() squirrel.UpdateBuilder

		DeleteSoft(ctx context.Context, data *UserBalance) error
		FindOneByQuery(ctx context.Context, selectBuilder squirrel.SelectBuilder) (*UserBalance, error)
		FindOneById(ctx context.Context, data *UserBalance) (*UserBalance, error)
		FindAll(ctx context.Context, orderBy string) ([]*UserBalance, error)
		Find(ctx context.Context, selectBuilder squirrel.SelectBuilder) ([]*UserBalance, error)
		FindPageListByPage(ctx context.Context, selectBuilder squirrel.SelectBuilder, page, pageSize int64, orderBy string) ([]*UserBalance, error)
		UpdateColsByCond(ctx context.Context, updateBuilder squirrel.UpdateBuilder) (sql.Result, error)
	}

	customUserBalanceModel struct {
		*defaultUserBalanceModel
	}
)

// NewUserBalanceModel returns a model for the database table.
func NewUserBalanceModel(conn sqlx.SqlConn) UserBalanceModel {
	return &customUserBalanceModel{
		defaultUserBalanceModel: newUserBalanceModel(conn),
	}
}

// AllFieldsBuilder return a SelectBuilder with select("*")
func (m *defaultUserBalanceModel) AllFieldsBuilder() squirrel.SelectBuilder {
	return squirrel.Select("*").From(m.table)
}

// UpdateBuilder return a empty UpdateBuilder
func (m *defaultUserBalanceModel) UpdateBuilder() squirrel.UpdateBuilder {
	return squirrel.Update(m.table)
}

// DeleteSoft set deleted_at with CURRENT_TIMESTAMP
func (m *defaultUserBalanceModel) DeleteSoft(ctx context.Context, data *UserBalance) error {
	builder := squirrel.Update(m.table)
	builder = builder.Set("deleted_at", sql.NullTime{
		Time:  time.Now(),
		Valid: true,
	})
	builder = builder.Where("id = ?", data.Id)
	query, args, err := builder.ToSql()
	if err != nil {
		return err
	}

	if _, err := m.conn.ExecCtx(ctx, query, args...); err != nil {
		return err
	}
	return nil
}

// FindOneByQuery if table has deleted_at use FindOneByQuery instead of FindOne
func (m *defaultUserBalanceModel) FindOneByQuery(ctx context.Context, selectBuilder squirrel.SelectBuilder) (*UserBalance, error) {
	selectBuilder = selectBuilder.Where("deleted_at is null").Limit(1)
	query, args, err := selectBuilder.ToSql()
	if err != nil {
		return nil, err
	}

	var resp UserBalance
	err = m.conn.QueryRowCtx(ctx, &resp, query, args...)
	switch err {
	case nil:
		return &resp, nil
	default:
		return nil, err
	}
}

// FindOneById just like FindOneByQuery but use data.Id as query condition
func (m *defaultUserBalanceModel) FindOneById(ctx context.Context, data *UserBalance) (*UserBalance, error) {
	return m.FindOneByQuery(ctx, m.AllFieldsBuilder().Where("id = ?", data.Id))
}

// FindAll returns all valid rows in the table
func (m *defaultUserBalanceModel) FindAll(ctx context.Context, orderBy string) ([]*UserBalance, error) {
	selectBuilder := m.AllFieldsBuilder()
	if orderBy == "" {
		selectBuilder = selectBuilder.OrderBy("id DESC")
	} else {
		selectBuilder = selectBuilder.OrderBy(orderBy)
	}

	query, args, err := selectBuilder.Where("deleted_at is null").ToSql()
	if err != nil {
		return nil, err
	}

	var resp []*UserBalance
	err = m.conn.QueryRowsCtx(ctx, &resp, query, args...)
	switch err {
	case nil:
		return resp, nil
	default:
		return nil, err
	}
}

// Find returns all valid rows matched in the table
func (m *defaultUserBalanceModel) Find(ctx context.Context, selectBuilder squirrel.SelectBuilder) ([]*UserBalance, error) {
	query, args, err := selectBuilder.Where("deleted_at is null").ToSql()
	if err != nil {
		return nil, err
	}

	var resp []*UserBalance
	err = m.conn.QueryRowsCtx(ctx, &resp, query, args...)
	switch err {
	case nil:
		return resp, nil
	default:
		return nil, err
	}
}

// FindPageListByPage -
func (m *defaultUserBalanceModel) FindPageListByPage(ctx context.Context, selectBuilder squirrel.SelectBuilder, page, pageSize int64, orderBy string) ([]*UserBalance, error) {
	if orderBy == "" {
		selectBuilder = selectBuilder.OrderBy("id DESC")
	} else {
		selectBuilder = selectBuilder.OrderBy(orderBy)
	}

	if page < 1 {
		page = 1
	}
	offset := (page - 1) * pageSize

	query, args, err := selectBuilder.Where("deleted_at is null").Offset(uint64(offset)).Limit(uint64(pageSize)).ToSql()
	if err != nil {
		return nil, err
	}

	var resp []*UserBalance
	err = m.conn.QueryRowsCtx(ctx, &resp, query, args...)
	switch err {
	case nil:
		return resp, nil
	default:
		return nil, err
	}
}

// UpdateColsByCond -
func (m *defaultUserBalanceModel) UpdateColsByCond(ctx context.Context, updateBuilder squirrel.UpdateBuilder) (sql.Result, error) {
	query, args, err := updateBuilder.ToSql()
	if err != nil {
		return nil, err
	}

	return m.conn.ExecCtx(ctx, query, args...)
}
