package model

import (
	"context"
	"database/sql"
	"time"

	"github.com/Masterminds/squirrel"
	"github.com/zeromicro/go-zero/core/stores/sqlx"
)

var _ SysCpodNodeModel = (*customSysCpodNodeModel)(nil)

type (
	// SysCpodNodeModel is an interface to be customized, add more methods here,
	// and implement the added methods in customSysCpodNodeModel.
	SysCpodNodeModel interface {
		sysCpodNodeModel
		AllFieldsBuilder() squirrel.SelectBuilder
		UpdateBuilder() squirrel.UpdateBuilder

		DeleteSoft(ctx context.Context, data *SysCpodNode) error
		FindOneByQuery(ctx context.Context, selectBuilder squirrel.SelectBuilder) (*SysCpodNode, error)
		FindOneById(ctx context.Context, data *SysCpodNode) (*SysCpodNode, error)
		FindAll(ctx context.Context, orderBy string) ([]*SysCpodNode, error)
		Find(ctx context.Context, selectBuilder squirrel.SelectBuilder) ([]*SysCpodNode, error)
		FindPageListByPage(ctx context.Context, selectBuilder squirrel.SelectBuilder, page, pageSize int64, orderBy string) ([]*SysCpodNode, error)
		UpdateColsByCond(ctx context.Context, updateBuilder squirrel.UpdateBuilder) (sql.Result, error)
	}

	customSysCpodNodeModel struct {
		*defaultSysCpodNodeModel
	}
)

// NewSysCpodNodeModel returns a model for the database table.
func NewSysCpodNodeModel(conn sqlx.SqlConn) SysCpodNodeModel {
	return &customSysCpodNodeModel{
		defaultSysCpodNodeModel: newSysCpodNodeModel(conn),
	}
}

// AllFieldsBuilder return a SelectBuilder with select("*")
func (m *defaultSysCpodNodeModel) AllFieldsBuilder() squirrel.SelectBuilder {
	return squirrel.Select("*").From(m.table)
}

// UpdateBuilder return a empty UpdateBuilder
func (m *defaultSysCpodNodeModel) UpdateBuilder() squirrel.UpdateBuilder {
	return squirrel.Update(m.table)
}

// DeleteSoft set deleted_at with CURRENT_TIMESTAMP
func (m *defaultSysCpodNodeModel) DeleteSoft(ctx context.Context, data *SysCpodNode) error {
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
func (m *defaultSysCpodNodeModel) FindOneByQuery(ctx context.Context, selectBuilder squirrel.SelectBuilder) (*SysCpodNode, error) {
	selectBuilder = selectBuilder.Where("deleted_at is null").Limit(1)
	query, args, err := selectBuilder.ToSql()
	if err != nil {
		return nil, err
	}

	var resp SysCpodNode
	err = m.conn.QueryRowCtx(ctx, &resp, query, args...)
	switch err {
	case nil:
		return &resp, nil
	default:
		return nil, err
	}
}

// FindOneById just like FindOneByQuery but use data.Id as query condition
func (m *defaultSysCpodNodeModel) FindOneById(ctx context.Context, data *SysCpodNode) (*SysCpodNode, error) {
	return m.FindOneByQuery(ctx, m.AllFieldsBuilder().Where("id = ?", data.Id))
}

// FindAll returns all valid rows in the table
func (m *defaultSysCpodNodeModel) FindAll(ctx context.Context, orderBy string) ([]*SysCpodNode, error) {
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

	var resp []*SysCpodNode
	err = m.conn.QueryRowsCtx(ctx, &resp, query, args...)
	switch err {
	case nil:
		return resp, nil
	default:
		return nil, err
	}
}

// Find returns all valid rows matched in the table
func (m *defaultSysCpodNodeModel) Find(ctx context.Context, selectBuilder squirrel.SelectBuilder) ([]*SysCpodNode, error) {
	query, args, err := selectBuilder.Where("deleted_at is null").ToSql()
	if err != nil {
		return nil, err
	}

	var resp []*SysCpodNode
	err = m.conn.QueryRowsCtx(ctx, &resp, query, args...)
	switch err {
	case nil:
		return resp, nil
	default:
		return nil, err
	}
}

// FindPageListByPage -
func (m *defaultSysCpodNodeModel) FindPageListByPage(ctx context.Context, selectBuilder squirrel.SelectBuilder, page, pageSize int64, orderBy string) ([]*SysCpodNode, error) {
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

	var resp []*SysCpodNode
	err = m.conn.QueryRowsCtx(ctx, &resp, query, args...)
	switch err {
	case nil:
		return resp, nil
	default:
		return nil, err
	}
}

// UpdateColsByCond -
func (m *defaultSysCpodNodeModel) UpdateColsByCond(ctx context.Context, updateBuilder squirrel.UpdateBuilder) (sql.Result, error) {
	query, args, err := updateBuilder.ToSql()
	if err != nil {
		return nil, err
	}

	return m.conn.ExecCtx(ctx, query, args...)
}
