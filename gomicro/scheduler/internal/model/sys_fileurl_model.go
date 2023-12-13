package model

import (
	"context"
	"github.com/Masterminds/squirrel"
	"github.com/zeromicro/go-zero/core/stores/sqlx"
)

var _ SysFileurlModel = (*customSysFileurlModel)(nil)

type (
	// SysFileurlModel is an interface to be customized, add more methods here,
	// and implement the added methods in customSysFileurlModel.
	SysFileurlModel interface {
		sysFileurlModel
		AllFieldsBuilder() squirrel.SelectBuilder

		FindOneByQuery(ctx context.Context, selectBuilder squirrel.SelectBuilder) (*SysFileurl, error)
		FindOneById(ctx context.Context, data *SysFileurl) (*SysFileurl, error)
		FindAll(ctx context.Context, orderBy string) ([]*SysFileurl, error)
		FindPageListByPage(ctx context.Context, selectBuilder squirrel.SelectBuilder, page, pageSize int64, orderBy string) ([]*SysFileurl, error)
	}

	customSysFileurlModel struct {
		*defaultSysFileurlModel
	}
)

// NewSysFileurlModel returns a model for the database table.
func NewSysFileurlModel(conn sqlx.SqlConn) SysFileurlModel {
	return &customSysFileurlModel{
		defaultSysFileurlModel: newSysFileurlModel(conn),
	}
}

// AllFieldsBuilder return a SelectBuilder with select("*")
func (m *defaultSysFileurlModel) AllFieldsBuilder() squirrel.SelectBuilder {
	return squirrel.Select("*").From(m.table)
}

// FindOneByQuery if table has deleted_at use FindOneByQuery instead of FindOne
func (m *defaultSysFileurlModel) FindOneByQuery(ctx context.Context, selectBuilder squirrel.SelectBuilder) (*SysFileurl, error) {
	selectBuilder = selectBuilder.Limit(1)
	query, args, err := selectBuilder.ToSql()
	if err != nil {
		return nil, err
	}

	var resp SysFileurl
	err = m.conn.QueryRowCtx(ctx, &resp, query, args...)
	switch err {
	case nil:
		return &resp, nil
	default:
		return nil, err
	}
}

// FindOneById just like FindOneByQuery but use data.Id as query condition
func (m *defaultSysFileurlModel) FindOneById(ctx context.Context, data *SysFileurl) (*SysFileurl, error) {
	return m.FindOneByQuery(ctx, m.AllFieldsBuilder().Where("file_id = ?", data.FileId))
}

// FindAll returns all valid rows in the table
func (m *defaultSysFileurlModel) FindAll(ctx context.Context, orderBy string) ([]*SysFileurl, error) {
	selectBuilder := m.AllFieldsBuilder()
	if orderBy == "" {
		selectBuilder = selectBuilder.OrderBy("file_id DESC")
	} else {
		selectBuilder = selectBuilder.OrderBy(orderBy)
	}

	query, args, err := selectBuilder.ToSql()
	if err != nil {
		return nil, err
	}

	var resp []*SysFileurl
	err = m.conn.QueryRowsCtx(ctx, &resp, query, args...)
	switch err {
	case nil:
		return resp, nil
	default:
		return nil, err
	}
}

// FindPageListByPage -
func (m *defaultSysFileurlModel) FindPageListByPage(ctx context.Context, selectBuilder squirrel.SelectBuilder, page, pageSize int64, orderBy string) ([]*SysFileurl, error) {
	if orderBy == "" {
		selectBuilder = selectBuilder.OrderBy("file_id DESC")
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

	var resp []*SysFileurl
	err = m.conn.QueryRowsCtx(ctx, &resp, query, args...)
	switch err {
	case nil:
		return resp, nil
	default:
		return nil, err
	}
}
