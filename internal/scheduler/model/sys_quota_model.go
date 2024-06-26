package model

import (
	"context"
	"database/sql"

	"github.com/Masterminds/squirrel"
	"github.com/zeromicro/go-zero/core/stores/sqlx"
)

var _ SysQuotaModel = (*customSysQuotaModel)(nil)

type (
	// SysQuotaModel is an interface to be customized, add more methods here,
	// and implement the added methods in customSysQuotaModel.
	SysQuotaModel interface {
		sysQuotaModel
		AllFieldsBuilder() squirrel.SelectBuilder
		UpdateBuilder() squirrel.UpdateBuilder

		// DeleteSoft(ctx context.Context, data *SysQuota) error
		FindOneByQuery(ctx context.Context, selectBuilder squirrel.SelectBuilder) (*SysQuota, error)
		FindOneById(ctx context.Context, data *SysQuota) (*SysQuota, error)
		FindAll(ctx context.Context, orderBy string) ([]*SysQuota, error)
		Find(ctx context.Context, selectBuilder squirrel.SelectBuilder) ([]*SysQuota, error)
		FindPageListByPage(ctx context.Context, selectBuilder squirrel.SelectBuilder, page, pageSize int64, orderBy string) ([]*SysQuota, error)
		UpdateColsByCond(ctx context.Context, updateBuilder squirrel.UpdateBuilder) (sql.Result, error)

		// Trans(ctx context.Context, fn func(context context.Context, session sqlx.Session) error) error
		// RowBuilder() squirrel.SelectBuilder
		// CountBuilder(field string) squirrel.SelectBuilder
		// SumBuilder(field string) squirrel.SelectBuilder
		// FindOneByQuery(ctx context.Context, rowBuilder squirrel.SelectBuilder) (*HomestayOrder, error)
		// FindSum(ctx context.Context, sumBuilder squirrel.SelectBuilder) (float64, error)
		// FindCount(ctx context.Context, countBuilder squirrel.SelectBuilder) (int64, error)
		// FindAll(ctx context.Context, rowBuilder squirrel.SelectBuilder, orderBy string) ([]*HomestayOrder, error)
		// FindPageListByPage(ctx context.Context, rowBuilder squirrel.SelectBuilder, page, pageSize int64, orderBy string) ([]*HomestayOrder, error)
		// FindPageListByIdDESC(ctx context.Context, rowBuilder squirrel.SelectBuilder, preMinId, pageSize int64) ([]*HomestayOrder, error)
		// FindPageListByIdASC(ctx context.Context, rowBuilder squirrel.SelectBuilder, preMaxId, pageSize int64) ([]*HomestayOrder, error)
	}

	customSysQuotaModel struct {
		*defaultSysQuotaModel
	}
)

// NewSysQuotaModel returns a model for the database table.
func NewSysQuotaModel(conn sqlx.SqlConn) SysQuotaModel {
	return &customSysQuotaModel{
		defaultSysQuotaModel: newSysQuotaModel(conn),
	}
}

// AllFieldsBuilder return a SelectBuilder with select("*")
func (m *defaultSysQuotaModel) AllFieldsBuilder() squirrel.SelectBuilder {
	return squirrel.Select("*").From(m.table)
}

// UpdateBuilder return a empty UpdateBuilder
func (m *defaultSysQuotaModel) UpdateBuilder() squirrel.UpdateBuilder {
	return squirrel.Update(m.table)
}

// // DeleteSoft set deleted_at with CURRENT_TIMESTAMP
// func (m *defaultSysQuotaModel) DeleteSoft(ctx context.Context, data *SysQuota) error {
// 	builder := squirrel.Update(m.table)
// 	builder = builder.Set("deleted_at", sql.NullTime{
// 		Time:  time.Now(),
// 		Valid: true,
// 	})
// 	builder = builder.Where("id = ?", data.Id)
// 	query, args, err := builder.ToSql()
// 	if err != nil {
// 		return err
// 	}
//
// 	if _, err := m.conn.ExecCtx(ctx, query, args...); err != nil {
// 		return err
// 	}
// 	return nil
// }

// FindOneByQuery if table has deleted_at use FindOneByQuery instead of FindOne
func (m *defaultSysQuotaModel) FindOneByQuery(ctx context.Context, selectBuilder squirrel.SelectBuilder) (*SysQuota, error) {
	selectBuilder = selectBuilder.Limit(1)
	query, args, err := selectBuilder.ToSql()
	if err != nil {
		return nil, err
	}

	var resp SysQuota
	err = m.conn.QueryRowCtx(ctx, &resp, query, args...)
	switch err {
	case nil:
		return &resp, nil
	default:
		return nil, err
	}
}

// FindOneById just like FindOneByQuery but use data.Id as query condition
func (m *defaultSysQuotaModel) FindOneById(ctx context.Context, data *SysQuota) (*SysQuota, error) {
	return m.FindOneByQuery(ctx, m.AllFieldsBuilder().Where("id = ?", data.Id))
}

// FindAll returns all valid rows in the table
func (m *defaultSysQuotaModel) FindAll(ctx context.Context, orderBy string) ([]*SysQuota, error) {
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

	var resp []*SysQuota
	err = m.conn.QueryRowsCtx(ctx, &resp, query, args...)
	switch err {
	case nil:
		return resp, nil
	default:
		return nil, err
	}
}

func (m *defaultSysQuotaModel) Find(ctx context.Context, selectBuilder squirrel.SelectBuilder) ([]*SysQuota, error) {
	query, args, err := selectBuilder.ToSql()
	if err != nil {
		return nil, err
	}

	var resp []*SysQuota
	err = m.conn.QueryRowsCtx(ctx, &resp, query, args...)
	switch err {
	case nil:
		return resp, nil
	default:
		return nil, err
	}
}

// FindPageListByPage -
func (m *defaultSysQuotaModel) FindPageListByPage(ctx context.Context, selectBuilder squirrel.SelectBuilder, page, pageSize int64, orderBy string) ([]*SysQuota, error) {
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

	var resp []*SysQuota
	err = m.conn.QueryRowsCtx(ctx, &resp, query, args...)
	switch err {
	case nil:
		return resp, nil
	default:
		return nil, err
	}
}

// UpdateColsByCond -
func (m *defaultSysQuotaModel) UpdateColsByCond(ctx context.Context, updateBuilder squirrel.UpdateBuilder) (sql.Result, error) {
	query, args, err := updateBuilder.ToSql()
	if err != nil {
		return nil, err
	}

	return m.conn.ExecCtx(ctx, query, args...)
}
