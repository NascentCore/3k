package model

import (
	"context"
	"database/sql"

	"github.com/Masterminds/squirrel"
	"github.com/zeromicro/go-zero/core/stores/sqlx"
)

var _ SysInferenceModel = (*customSysInferenceModel)(nil)

const (
	InferStatusDescWaitDeploy = "waitdeploy"
	InferStatusDescDeploying  = "deploying"
	InferStatusDescDeployed   = "deployed"
	InferStatusDescStopped    = "stopped"
)

const (
	InferStatusWaitDeploy = 0
	InferStatusDeploying  = 1
	InferStatusDeployed   = 2
	InferStatusStopped    = 3
)

var (
	InferStatusToDesc = map[int64]string{
		InferStatusWaitDeploy: InferStatusDescWaitDeploy,
		InferStatusDeploying:  InferStatusDescDeploying,
		InferStatusDeployed:   InferStatusDescDeployed,
		InferStatusStopped:    InferStatusDescStopped,
	}

	InferDescToStatus = map[string]int64{
		InferStatusDescWaitDeploy: InferStatusWaitDeploy,
		InferStatusDescDeploying:  InferStatusDeploying,
		InferStatusDescDeployed:   InferStatusDeployed,
		InferStatusDescStopped:    InferStatusStopped,
	}
)

type (
	// SysInferenceModel is an interface to be customized, add more methods here,
	// and implement the added methods in customSysInferenceModel.
	SysInferenceModel interface {
		sysInferenceModel
		AllFieldsBuilder() squirrel.SelectBuilder
		UpdateBuilder() squirrel.UpdateBuilder

		FindOneByQuery(ctx context.Context, selectBuilder squirrel.SelectBuilder) (*SysInference, error)
		FindOneById(ctx context.Context, data *SysInference) (*SysInference, error)
		FindAll(ctx context.Context, rowBuilder squirrel.SelectBuilder, orderBy string) ([]*SysInference, error)
		FindPageListByPage(ctx context.Context, selectBuilder squirrel.SelectBuilder, page, pageSize int64, orderBy string) ([]*SysInference, error)
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

	customSysInferenceModel struct {
		*defaultSysInferenceModel
	}
)

// NewSysInferenceModel returns a model for the database table.
func NewSysInferenceModel(conn sqlx.SqlConn) SysInferenceModel {
	return &customSysInferenceModel{
		defaultSysInferenceModel: newSysInferenceModel(conn),
	}
}

// AllFieldsBuilder return a SelectBuilder with select("*")
func (m *defaultSysInferenceModel) AllFieldsBuilder() squirrel.SelectBuilder {
	return squirrel.Select("*").From(m.table)
}

// UpdateBuilder return a empty UpdateBuilder
func (m *defaultSysInferenceModel) UpdateBuilder() squirrel.UpdateBuilder {
	return squirrel.Update(m.table)
}

// FindOneByQuery if table has deleted_at use FindOneByQuery instead of FindOne
func (m *defaultSysInferenceModel) FindOneByQuery(ctx context.Context, selectBuilder squirrel.SelectBuilder) (*SysInference, error) {
	selectBuilder = selectBuilder.Where("deleted_at is null").Limit(1)
	query, args, err := selectBuilder.ToSql()
	if err != nil {
		return nil, err
	}

	var resp SysInference
	err = m.conn.QueryRowCtx(ctx, &resp, query, args...)
	switch err {
	case nil:
		return &resp, nil
	default:
		return nil, err
	}
}

// FindOneById just like FindOneByQuery but use data.Id as query condition
func (m *defaultSysInferenceModel) FindOneById(ctx context.Context, data *SysInference) (*SysInference, error) {
	return m.FindOneByQuery(ctx, m.AllFieldsBuilder().Where("id = ?", data.Id))
}

// FindAll returns all valid rows in the table
func (m *defaultSysInferenceModel) FindAll(ctx context.Context, rowBuilder squirrel.SelectBuilder, orderBy string) ([]*SysInference, error) {
	if orderBy == "" {
		rowBuilder = rowBuilder.OrderBy("id DESC")
	} else {
		rowBuilder = rowBuilder.OrderBy(orderBy)
	}

	query, args, err := rowBuilder.ToSql()
	if err != nil {
		return nil, err
	}

	var resp []*SysInference
	err = m.conn.QueryRowsCtx(ctx, &resp, query, args...)
	switch err {
	case nil:
		return resp, nil
	default:
		return nil, err
	}
}

// FindPageListByPage -
func (m *defaultSysInferenceModel) FindPageListByPage(ctx context.Context, selectBuilder squirrel.SelectBuilder, page, pageSize int64, orderBy string) ([]*SysInference, error) {
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

	var resp []*SysInference
	err = m.conn.QueryRowsCtx(ctx, &resp, query, args...)
	switch err {
	case nil:
		return resp, nil
	default:
		return nil, err
	}
}

// UpdateColsByCond -
func (m *defaultSysInferenceModel) UpdateColsByCond(ctx context.Context, updateBuilder squirrel.UpdateBuilder) (sql.Result, error) {
	query, args, err := updateBuilder.ToSql()
	if err != nil {
		return nil, err
	}

	return m.conn.ExecCtx(ctx, query, args...)
}
