package model

import (
	"context"
	"database/sql"

	"github.com/Masterminds/squirrel"
	"github.com/zeromicro/go-zero/core/stores/sqlx"
)

var _ SysPriceModel = (*customSysPriceModel)(nil)

type (
	// SysPriceModel is an interface to be customized, add more methods here,
	// and implement the added methods in customSysPriceModel.
	SysPriceModel interface {
		sysPriceModel
		AllFieldsBuilder() squirrel.SelectBuilder
		UpdateBuilder() squirrel.UpdateBuilder

		FindOneByQuery(ctx context.Context, selectBuilder squirrel.SelectBuilder) (*SysPrice, error)
		FindOneById(ctx context.Context, data *SysPrice) (*SysPrice, error)
		FindAll(ctx context.Context, orderBy string) ([]*SysPrice, error)
		Find(ctx context.Context, selectBuilder squirrel.SelectBuilder) ([]*SysPrice, error)
		FindPageListByPage(ctx context.Context, selectBuilder squirrel.SelectBuilder, page, pageSize int64, orderBy string) ([]*SysPrice, error)
		UpdateColsByCond(ctx context.Context, updateBuilder squirrel.UpdateBuilder) (sql.Result, error)
	}

	customSysPriceModel struct {
		*defaultSysPriceModel
	}
)

// NewSysPriceModel returns a model for the database table.
func NewSysPriceModel(conn sqlx.SqlConn) SysPriceModel {
	return &customSysPriceModel{
		defaultSysPriceModel: newSysPriceModel(conn),
	}
}

// AllFieldsBuilder return a SelectBuilder with select("*")
func (m *defaultSysPriceModel) AllFieldsBuilder() squirrel.SelectBuilder {
	return squirrel.Select("*").From(m.table)
}

// UpdateBuilder return a empty UpdateBuilder
func (m *defaultSysPriceModel) UpdateBuilder() squirrel.UpdateBuilder {
	return squirrel.Update(m.table)
}

// FindOneByQuery if table has deleted_at use FindOneByQuery instead of FindOne
func (m *defaultSysPriceModel) FindOneByQuery(ctx context.Context, selectBuilder squirrel.SelectBuilder) (*SysPrice, error) {
	selectBuilder = selectBuilder.Limit(1)
	query, args, err := selectBuilder.ToSql()
	if err != nil {
		return nil, err
	}

	var resp SysPrice
	err = m.conn.QueryRowCtx(ctx, &resp, query, args...)
	switch err {
	case nil:
		return &resp, nil
	default:
		return nil, err
	}
}

// FindOneById just like FindOneByQuery but use data.Id as query condition
func (m *defaultSysPriceModel) FindOneById(ctx context.Context, data *SysPrice) (*SysPrice, error) {
	return m.FindOneByQuery(ctx, m.AllFieldsBuilder().Where("price_id = ?", data.PriceId))
}

// FindAll returns all valid rows in the table
func (m *defaultSysPriceModel) FindAll(ctx context.Context, orderBy string) ([]*SysPrice, error) {
	selectBuilder := m.AllFieldsBuilder()
	if orderBy == "" {
		selectBuilder = selectBuilder.OrderBy("price_id DESC")
	} else {
		selectBuilder = selectBuilder.OrderBy(orderBy)
	}

	query, args, err := selectBuilder.ToSql()
	if err != nil {
		return nil, err
	}

	var resp []*SysPrice
	err = m.conn.QueryRowsCtx(ctx, &resp, query, args...)
	switch err {
	case nil:
		return resp, nil
	default:
		return nil, err
	}
}

// Find returns all valid rows matched in the table
func (m *defaultSysPriceModel) Find(ctx context.Context, selectBuilder squirrel.SelectBuilder) ([]*SysPrice, error) {
	query, args, err := selectBuilder.ToSql()
	if err != nil {
		return nil, err
	}

	var resp []*SysPrice
	err = m.conn.QueryRowsCtx(ctx, &resp, query, args...)
	switch err {
	case nil:
		return resp, nil
	default:
		return nil, err
	}
}

// FindPageListByPage -
func (m *defaultSysPriceModel) FindPageListByPage(ctx context.Context, selectBuilder squirrel.SelectBuilder, page, pageSize int64, orderBy string) ([]*SysPrice, error) {
	if orderBy == "" {
		selectBuilder = selectBuilder.OrderBy("price_id DESC")
	} else {
		selectBuilder = selectBuilder.OrderBy(orderBy)
	}

	if page < 1 {
		page = 1
	}
	offset := (page - 1) * pageSize

	query, args, err := selectBuilder.Offset(uint64(offset)).Limit(uint64(pageSize)).ToSql()
	if err != nil {
		return nil, err
	}

	var resp []*SysPrice
	err = m.conn.QueryRowsCtx(ctx, &resp, query, args...)
	switch err {
	case nil:
		return resp, nil
	default:
		return nil, err
	}
}

// UpdateColsByCond -
func (m *defaultSysPriceModel) UpdateColsByCond(ctx context.Context, updateBuilder squirrel.UpdateBuilder) (sql.Result, error) {
	query, args, err := updateBuilder.ToSql()
	if err != nil {
		return nil, err
	}

	return m.conn.ExecCtx(ctx, query, args...)
}
