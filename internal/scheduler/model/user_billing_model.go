package model

import (
	"context"
	"database/sql"

	"github.com/Masterminds/squirrel"
	"github.com/zeromicro/go-zero/core/stores/sqlx"
)

const (
	BillingTypeCpodJob    = "cpodjob"
	BillingTypeFinetune   = "finetune"
	BillingTypeTraining   = "training"
	BillingTypeInference  = "inference"
	BillingTypeJupyterlab = "jupyterlab"
)

var _ UserBillingModel = (*customUserBillingModel)(nil)

type (
	// UserBillingModel is an interface to be customized, add more methods here,
	// and implement the added methods in customUserBillingModel.
	UserBillingModel interface {
		userBillingModel
		AllFieldsBuilder() squirrel.SelectBuilder
		UpdateBuilder() squirrel.UpdateBuilder
		InsertBuilder() squirrel.InsertBuilder

		FindOneByQuery(ctx context.Context, selectBuilder squirrel.SelectBuilder) (*UserBilling, error)
		FindOneById(ctx context.Context, data *UserBilling) (*UserBilling, error)
		FindAll(ctx context.Context, orderBy string) ([]*UserBilling, error)
		Find(ctx context.Context, selectBuilder squirrel.SelectBuilder) ([]*UserBilling, error)
		FindPageListByPage(ctx context.Context, selectBuilder squirrel.SelectBuilder, page, pageSize int64, orderBy string) ([]*UserBilling, error)
		UpdateColsByCond(ctx context.Context, updateBuilder squirrel.UpdateBuilder) (sql.Result, error)
	}

	customUserBillingModel struct {
		*defaultUserBillingModel
	}
)

// NewUserBillingModel returns a model for the database table.
func NewUserBillingModel(conn sqlx.SqlConn) UserBillingModel {
	return &customUserBillingModel{
		defaultUserBillingModel: newUserBillingModel(conn),
	}
}

// AllFieldsBuilder return a SelectBuilder with select("*")
func (m *defaultUserBillingModel) AllFieldsBuilder() squirrel.SelectBuilder {
	return squirrel.Select("*").From(m.table)
}

// UpdateBuilder return a empty UpdateBuilder
func (m *defaultUserBillingModel) UpdateBuilder() squirrel.UpdateBuilder {
	return squirrel.Update(m.table)
}

// InsertBuilder return a empty InsertBuilder
func (m *defaultUserBillingModel) InsertBuilder() squirrel.InsertBuilder {
	return squirrel.Insert(m.table)
}

// FindOneByQuery if table has deleted_at use FindOneByQuery instead of FindOne
func (m *defaultUserBillingModel) FindOneByQuery(ctx context.Context, selectBuilder squirrel.SelectBuilder) (*UserBilling, error) {
	selectBuilder = selectBuilder.Limit(1)
	query, args, err := selectBuilder.ToSql()
	if err != nil {
		return nil, err
	}

	var resp UserBilling
	err = m.conn.QueryRowCtx(ctx, &resp, query, args...)
	switch err {
	case nil:
		return &resp, nil
	default:
		return nil, err
	}
}

// FindOneById just like FindOneByQuery but use data.Id as query condition
func (m *defaultUserBillingModel) FindOneById(ctx context.Context, data *UserBilling) (*UserBilling, error) {
	return m.FindOneByQuery(ctx, m.AllFieldsBuilder().Where("id = ?", data.Id))
}

// FindAll returns all valid rows in the table
func (m *defaultUserBillingModel) FindAll(ctx context.Context, orderBy string) ([]*UserBilling, error) {
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

	var resp []*UserBilling
	err = m.conn.QueryRowsCtx(ctx, &resp, query, args...)
	switch err {
	case nil:
		return resp, nil
	default:
		return nil, err
	}
}

// Find returns all valid rows matched in the table
func (m *defaultUserBillingModel) Find(ctx context.Context, selectBuilder squirrel.SelectBuilder) ([]*UserBilling, error) {
	query, args, err := selectBuilder.ToSql()
	if err != nil {
		return nil, err
	}

	var resp []*UserBilling
	err = m.conn.QueryRowsCtx(ctx, &resp, query, args...)
	switch err {
	case nil:
		return resp, nil
	default:
		return nil, err
	}
}

// FindPageListByPage -
func (m *defaultUserBillingModel) FindPageListByPage(ctx context.Context, selectBuilder squirrel.SelectBuilder, page, pageSize int64, orderBy string) ([]*UserBilling, error) {
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

	var resp []*UserBilling
	err = m.conn.QueryRowsCtx(ctx, &resp, query, args...)
	switch err {
	case nil:
		return resp, nil
	default:
		return nil, err
	}
}

// UpdateColsByCond -
func (m *defaultUserBillingModel) UpdateColsByCond(ctx context.Context, updateBuilder squirrel.UpdateBuilder) (sql.Result, error) {
	query, args, err := updateBuilder.ToSql()
	if err != nil {
		return nil, err
	}

	return m.conn.ExecCtx(ctx, query, args...)
}
