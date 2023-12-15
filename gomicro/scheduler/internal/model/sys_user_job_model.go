package model

import (
	"context"
	"database/sql"
	"time"

	"github.com/Masterminds/squirrel"
	"github.com/zeromicro/go-zero/core/stores/sqlx"
)

var _ SysUserJobModel = (*customSysUserJobModel)(nil)

type (
	// SysUserJobModel is an interface to be customized, add more methods here,
	// and implement the added methods in customSysUserJobModel.
	SysUserJobModel interface {
		sysUserJobModel
		AllFieldsBuilder() squirrel.SelectBuilder
		UpdateBuilder() squirrel.UpdateBuilder

		DeleteSoft(ctx context.Context, data *SysUserJob) error
		FindOneByQuery(ctx context.Context, selectBuilder squirrel.SelectBuilder) (*SysUserJob, error)
		FindOneById(ctx context.Context, data *SysUserJob) (*SysUserJob, error)
		FindAll(ctx context.Context, orderBy string) ([]*SysUserJob, error)
		FindPageListByPage(ctx context.Context, selectBuilder squirrel.SelectBuilder, page, pageSize int64, orderBy string) ([]*SysUserJob, error)
		UpdateColsByCond(ctx context.Context, updateBuilder squirrel.UpdateBuilder) (sql.Result, error)
	}

	customSysUserJobModel struct {
		*defaultSysUserJobModel
	}
)

// NewSysUserJobModel returns a model for the database table.
func NewSysUserJobModel(conn sqlx.SqlConn) SysUserJobModel {
	return &customSysUserJobModel{
		defaultSysUserJobModel: newSysUserJobModel(conn),
	}
}

// AllFieldsBuilder return a SelectBuilder with select("*")
func (m *defaultSysUserJobModel) AllFieldsBuilder() squirrel.SelectBuilder {
	return squirrel.Select("*").From(m.table)
}

// UpdateBuilder return a empty UpdateBuilder
func (m *defaultSysUserJobModel) UpdateBuilder() squirrel.UpdateBuilder {
	return squirrel.Update(m.table)
}

// DeleteSoft set deleted_at with CURRENT_TIMESTAMP
func (m *defaultSysUserJobModel) DeleteSoft(ctx context.Context, data *SysUserJob) error {
	builder := squirrel.Update(m.table)
	builder = builder.Set("deleted", sql.NullTime{
		Time:  time.Now(),
		Valid: true,
	})
	builder = builder.Where("job_id = ?", data.JobId)
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
func (m *defaultSysUserJobModel) FindOneByQuery(ctx context.Context, selectBuilder squirrel.SelectBuilder) (*SysUserJob, error) {
	selectBuilder = selectBuilder.Where("deleted = 0").Limit(1)
	query, args, err := selectBuilder.ToSql()
	if err != nil {
		return nil, err
	}

	var resp SysUserJob
	err = m.conn.QueryRowCtx(ctx, &resp, query, args...)
	switch err {
	case nil:
		return &resp, nil
	default:
		return nil, err
	}
}

// FindOneById just like FindOneByQuery but use data.job_id as query condition
func (m *defaultSysUserJobModel) FindOneById(ctx context.Context, data *SysUserJob) (*SysUserJob, error) {
	return m.FindOneByQuery(ctx, m.AllFieldsBuilder().Where("job_id = ?", data.JobId))
}

// FindAll returns all valid rows in the table
func (m *defaultSysUserJobModel) FindAll(ctx context.Context, orderBy string) ([]*SysUserJob, error) {
	selectBuilder := m.AllFieldsBuilder()
	if orderBy == "" {
		selectBuilder = selectBuilder.OrderBy("job_id DESC")
	} else {
		selectBuilder = selectBuilder.OrderBy(orderBy)
	}

	query, args, err := selectBuilder.Where("deleted = 0").ToSql()
	if err != nil {
		return nil, err
	}

	var resp []*SysUserJob
	err = m.conn.QueryRowsCtx(ctx, &resp, query, args...)
	switch err {
	case nil:
		return resp, nil
	default:
		return nil, err
	}
}

// FindPageListByPage -
func (m *defaultSysUserJobModel) FindPageListByPage(ctx context.Context, selectBuilder squirrel.SelectBuilder, page, pageSize int64, orderBy string) ([]*SysUserJob, error) {
	if orderBy == "" {
		selectBuilder = selectBuilder.OrderBy("job_id DESC")
	} else {
		selectBuilder = selectBuilder.OrderBy(orderBy)
	}

	if page < 1 {
		page = 1
	}
	offset := (page - 1) * pageSize

	query, args, err := selectBuilder.Where("deleted = 0").Offset(uint64(offset)).Limit(uint64(pageSize)).ToSql()
	if err != nil {
		return nil, err
	}

	var resp []*SysUserJob
	err = m.conn.QueryRowsCtx(ctx, &resp, query, args...)
	switch err {
	case nil:
		return resp, nil
	default:
		return nil, err
	}
}

// UpdateColsByCond -
func (m *defaultSysUserJobModel) UpdateColsByCond(ctx context.Context, updateBuilder squirrel.UpdateBuilder) (sql.Result, error) {
	query, args, err := updateBuilder.ToSql()
	if err != nil {
		return nil, err
	}

	return m.conn.ExecCtx(ctx, query, args...)
}
