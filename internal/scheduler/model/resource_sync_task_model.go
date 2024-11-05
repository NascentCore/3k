package model

import (
	"context"
	"database/sql"

	"github.com/Masterminds/squirrel"
	"github.com/zeromicro/go-zero/core/stores/sqlx"
)

const (
	ResourceSyncTaskStatusPending         = 0 // 待同步
	ResourceSyncTaskStatusGettingMeta     = 1 // 获取meta信息中
	ResourceSyncTaskStatusGettingMetaDone = 2 // 获取meta信息完成
	ResourceSyncTaskStatusTransfering     = 3 // 传输中(任务被获取)
	ResourceSyncTaskStatusUploaded        = 4 // 上传完成
	ResourceSyncTaskStatusRecord          = 5 // 资源表记录完成
	ResourceSyncTaskStatusFailed          = 9 // 同步失败
)

var _ ResourceSyncTaskModel = (*customResourceSyncTaskModel)(nil)

type (
	// ResourceSyncTaskModel is an interface to be customized, add more methods here,
	// and implement the added methods in customResourceSyncTaskModel.
	ResourceSyncTaskModel interface {
		resourceSyncTaskModel
		AllFieldsBuilder() squirrel.SelectBuilder
		UpdateBuilder() squirrel.UpdateBuilder

		DeleteSoft(ctx context.Context, data *ResourceSyncTask) error
		FindOneByQuery(ctx context.Context, selectBuilder squirrel.SelectBuilder) (*ResourceSyncTask, error)
		FindOneById(ctx context.Context, data *ResourceSyncTask) (*ResourceSyncTask, error)
		FindAll(ctx context.Context, orderBy string) ([]*ResourceSyncTask, error)
		Find(ctx context.Context, selectBuilder squirrel.SelectBuilder) ([]*ResourceSyncTask, error)
		FindPageListByPage(ctx context.Context, selectBuilder squirrel.SelectBuilder, page, pageSize int64, orderBy string) ([]*ResourceSyncTask, error)
		UpdateColsByCond(ctx context.Context, updateBuilder squirrel.UpdateBuilder) (sql.Result, error)
	}

	customResourceSyncTaskModel struct {
		*defaultResourceSyncTaskModel
	}
)

// NewResourceSyncTaskModel returns a model for the database table.
func NewResourceSyncTaskModel(conn sqlx.SqlConn) ResourceSyncTaskModel {
	return &customResourceSyncTaskModel{
		defaultResourceSyncTaskModel: newResourceSyncTaskModel(conn),
	}
}

// AllFieldsBuilder return a SelectBuilder with select("*")
func (m *defaultResourceSyncTaskModel) AllFieldsBuilder() squirrel.SelectBuilder {
	return squirrel.Select("*").From(m.table)
}

// UpdateBuilder return a empty UpdateBuilder
func (m *defaultResourceSyncTaskModel) UpdateBuilder() squirrel.UpdateBuilder {
	return squirrel.Update(m.table)
}

// DeleteSoft set deleted_at with CURRENT_TIMESTAMP
func (m *defaultResourceSyncTaskModel) DeleteSoft(ctx context.Context, data *ResourceSyncTask) error {
	builder := squirrel.Update(m.table)
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
func (m *defaultResourceSyncTaskModel) FindOneByQuery(ctx context.Context, selectBuilder squirrel.SelectBuilder) (*ResourceSyncTask, error) {
	selectBuilder = selectBuilder.Limit(1)
	query, args, err := selectBuilder.ToSql()
	if err != nil {
		return nil, err
	}

	var resp ResourceSyncTask
	err = m.conn.QueryRowCtx(ctx, &resp, query, args...)
	switch err {
	case nil:
		return &resp, nil
	default:
		return nil, err
	}
}

// FindOneById just like FindOneByQuery but use data.Id as query condition
func (m *defaultResourceSyncTaskModel) FindOneById(ctx context.Context, data *ResourceSyncTask) (*ResourceSyncTask, error) {
	return m.FindOneByQuery(ctx, m.AllFieldsBuilder().Where("id = ?", data.Id))
}

// FindAll returns all valid rows in the table
func (m *defaultResourceSyncTaskModel) FindAll(ctx context.Context, orderBy string) ([]*ResourceSyncTask, error) {
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

	var resp []*ResourceSyncTask
	err = m.conn.QueryRowsCtx(ctx, &resp, query, args...)
	switch err {
	case nil:
		return resp, nil
	default:
		return nil, err
	}
}

// Find returns all valid rows matched in the table
func (m *defaultResourceSyncTaskModel) Find(ctx context.Context, selectBuilder squirrel.SelectBuilder) ([]*ResourceSyncTask, error) {
	query, args, err := selectBuilder.ToSql()
	if err != nil {
		return nil, err
	}

	var resp []*ResourceSyncTask
	err = m.conn.QueryRowsCtx(ctx, &resp, query, args...)
	switch err {
	case nil:
		return resp, nil
	default:
		return nil, err
	}
}

// FindPageListByPage -
func (m *defaultResourceSyncTaskModel) FindPageListByPage(ctx context.Context, selectBuilder squirrel.SelectBuilder, page, pageSize int64, orderBy string) ([]*ResourceSyncTask, error) {
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

	var resp []*ResourceSyncTask
	err = m.conn.QueryRowsCtx(ctx, &resp, query, args...)
	switch err {
	case nil:
		return resp, nil
	default:
		return nil, err
	}
}

// UpdateColsByCond -
func (m *defaultResourceSyncTaskModel) UpdateColsByCond(ctx context.Context, updateBuilder squirrel.UpdateBuilder) (sql.Result, error) {
	query, args, err := updateBuilder.ToSql()
	if err != nil {
		return nil, err
	}

	return m.conn.ExecCtx(ctx, query, args...)
}
