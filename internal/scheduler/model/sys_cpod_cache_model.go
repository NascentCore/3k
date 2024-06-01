package model

import (
	"context"
	"database/sql"
	"fmt"

	"github.com/Masterminds/squirrel"
	"github.com/zeromicro/go-zero/core/stores/sqlx"
)

const (
	CacheModel   = 1
	CacheDataset = 2
	CacheImage   = 3
	CacheAdapter = 4
)

const (
	CachePublic  = 1
	CachePrivate = 2
)

var _ SysCpodCacheModel = (*customSysCpodCacheModel)(nil)

type (
	// SysCpodCacheModel is an interface to be customized, add more methods here,
	// and implement the added methods in customSysCpodCacheModel.
	SysCpodCacheModel interface {
		sysCpodCacheModel
		AllFieldsBuilder() squirrel.SelectBuilder
		UpdateBuilder() squirrel.UpdateBuilder
		DeleteBuilder() squirrel.DeleteBuilder

		FindOneByQuery(ctx context.Context, selectBuilder squirrel.SelectBuilder) (*SysCpodCache, error)
		FindOneById(ctx context.Context, data *SysCpodCache) (*SysCpodCache, error)
		FindAll(ctx context.Context, orderBy string) ([]*SysCpodCache, error)
		Find(ctx context.Context, selectBuilder squirrel.SelectBuilder) ([]*SysCpodCache, error)
		FindPageListByPage(ctx context.Context, selectBuilder squirrel.SelectBuilder, page, pageSize int64, orderBy string) ([]*SysCpodCache, error)
		FindActive(ctx context.Context, dataType int, userID string, minutes int) ([]*SysCpodCache, error)
		UpdateColsByCond(ctx context.Context, updateBuilder squirrel.UpdateBuilder) (sql.Result, error)
		DeleteByCond(ctx context.Context, deleteBuilder squirrel.DeleteBuilder) error
	}

	customSysCpodCacheModel struct {
		*defaultSysCpodCacheModel
	}
)

// NewSysCpodCacheModel returns a model for the database table.
func NewSysCpodCacheModel(conn sqlx.SqlConn) SysCpodCacheModel {
	return &customSysCpodCacheModel{
		defaultSysCpodCacheModel: newSysCpodCacheModel(conn),
	}
}

// AllFieldsBuilder return a SelectBuilder with select("*")
func (m *defaultSysCpodCacheModel) AllFieldsBuilder() squirrel.SelectBuilder {
	return squirrel.Select("*").From(m.table)
}

// UpdateBuilder return a empty UpdateBuilder
func (m *defaultSysCpodCacheModel) UpdateBuilder() squirrel.UpdateBuilder {
	return squirrel.Update(m.table)
}

// DeleteBuilder return a empty DeleteBuilder
func (m *defaultSysCpodCacheModel) DeleteBuilder() squirrel.DeleteBuilder {
	return squirrel.Delete(m.table)
}

// FindOneByQuery if table has deleted_at use FindOneByQuery instead of FindOne
func (m *defaultSysCpodCacheModel) FindOneByQuery(ctx context.Context, selectBuilder squirrel.SelectBuilder) (*SysCpodCache, error) {
	selectBuilder = selectBuilder.Limit(1)
	query, args, err := selectBuilder.ToSql()
	if err != nil {
		return nil, err
	}

	var resp SysCpodCache
	err = m.conn.QueryRowCtx(ctx, &resp, query, args...)
	switch err {
	case nil:
		return &resp, nil
	default:
		return nil, err
	}
}

// FindOneById just like FindOneByQuery but use data.Id as query condition
func (m *defaultSysCpodCacheModel) FindOneById(ctx context.Context, data *SysCpodCache) (*SysCpodCache, error) {
	return m.FindOneByQuery(ctx, m.AllFieldsBuilder().Where("id = ?", data.Id))
}

// FindAll returns all valid rows in the table
func (m *defaultSysCpodCacheModel) FindAll(ctx context.Context, orderBy string) ([]*SysCpodCache, error) {
	selectBuilder := m.AllFieldsBuilder()
	if orderBy == "" {
		selectBuilder = selectBuilder.OrderBy("id DESC")
	} else {
		selectBuilder = selectBuilder.OrderBy(orderBy)
	}

	query, args, err := selectBuilder.ToSql()
	if err != nil {
		return nil, err
	}

	var resp []*SysCpodCache
	err = m.conn.QueryRowsCtx(ctx, &resp, query, args...)
	switch err {
	case nil:
		return resp, nil
	default:
		return nil, err
	}
}

func (m *defaultSysCpodCacheModel) Find(ctx context.Context, selectBuilder squirrel.SelectBuilder) ([]*SysCpodCache, error) {
	query, args, err := selectBuilder.ToSql()
	if err != nil {
		return nil, err
	}

	var resp []*SysCpodCache
	err = m.conn.QueryRowsCtx(ctx, &resp, query, args...)
	switch err {
	case nil:
		return resp, nil
	default:
		return nil, err
	}
}

// FindPageListByPage -
func (m *defaultSysCpodCacheModel) FindPageListByPage(ctx context.Context, selectBuilder squirrel.SelectBuilder, page, pageSize int64, orderBy string) ([]*SysCpodCache, error) {
	if orderBy == "" {
		selectBuilder = selectBuilder.OrderBy("id DESC")
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

	var resp []*SysCpodCache
	err = m.conn.QueryRowsCtx(ctx, &resp, query, args...)
	switch err {
	case nil:
		return resp, nil
	default:
		return nil, err
	}
}

func (m *defaultSysCpodCacheModel) FindActive(ctx context.Context, dataType int, userID string, minutes int) ([]*SysCpodCache, error) {
	caches := make([]*SysCpodCache, 0)
	err := m.conn.QueryRowsCtx(ctx, &caches, fmt.Sprintf(`select c.*
	from sys_cpod_cache c
	join (select cpod_id, max(update_time) as update_time from sys_cpod_main group by cpod_id) m on c.cpod_id = m.cpod_id
	where c.data_type = %d and c.new_user_id = '%s' and m.update_time > NOW() - INTERVAL %d MINUTE;`, dataType, userID, minutes),
	)
	if err != nil {
		return nil, err
	}

	return caches, nil
}

// UpdateColsByCond -
func (m *defaultSysCpodCacheModel) UpdateColsByCond(ctx context.Context, updateBuilder squirrel.UpdateBuilder) (sql.Result, error) {
	query, args, err := updateBuilder.ToSql()
	if err != nil {
		return nil, err
	}

	return m.conn.ExecCtx(ctx, query, args...)
}

// DeleteByCond -
func (m *defaultSysCpodCacheModel) DeleteByCond(ctx context.Context, deleteBuilder squirrel.DeleteBuilder) error {
	query, args, err := deleteBuilder.ToSql()
	if err != nil {
		return err
	}

	_, err = m.conn.ExecCtx(ctx, query, args...)
	return err
}
