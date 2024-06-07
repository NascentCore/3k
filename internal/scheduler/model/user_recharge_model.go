package model

import (
	"context"
	"database/sql"
	"time"

	"github.com/Masterminds/squirrel"
	"github.com/zeromicro/go-zero/core/stores/sqlx"
)

var _ UserRechargeModel = (*customUserRechargeModel)(nil)

type (
	// UserRechargeModel is an interface to be customized, add more methods here,
	// and implement the added methods in customUserRechargeModel.
	UserRechargeModel interface {
		userRechargeModel
		AllFieldsBuilder() squirrel.SelectBuilder
		UpdateBuilder() squirrel.UpdateBuilder

		DeleteSoft(ctx context.Context, data *UserRecharge) error
		FindOneByQuery(ctx context.Context, selectBuilder squirrel.SelectBuilder) (*UserRecharge, error)
		FindOneById(ctx context.Context, data *UserRecharge) (*UserRecharge, error)
		FindAll(ctx context.Context, orderBy string) ([]*UserRecharge, error)
		Find(ctx context.Context, selectBuilder squirrel.SelectBuilder) ([]*UserRecharge, error)
		FindPageListByPage(ctx context.Context, where interface{}, page, pageSize int64, orderBy string) ([]*UserRecharge, int64, error)
		TransInsert(ctx context.Context, session sqlx.Session, data *UserRecharge) (sql.Result, error)
		UpdateColsByCond(ctx context.Context, updateBuilder squirrel.UpdateBuilder) (sql.Result, error)
		Count(ctx context.Context, where interface{}) (int64, error)
	}

	customUserRechargeModel struct {
		*defaultUserRechargeModel
	}
)

// NewUserRechargeModel returns a model for the database table.
func NewUserRechargeModel(conn sqlx.SqlConn) UserRechargeModel {
	return &customUserRechargeModel{
		defaultUserRechargeModel: newUserRechargeModel(conn),
	}
}

// AllFieldsBuilder return a SelectBuilder with select("*")
func (m *defaultUserRechargeModel) AllFieldsBuilder() squirrel.SelectBuilder {
	return squirrel.Select("*").From(m.table)
}

// UpdateBuilder return an empty UpdateBuilder
func (m *defaultUserRechargeModel) UpdateBuilder() squirrel.UpdateBuilder {
	return squirrel.Update(m.table)
}

// DeleteSoft set deleted_at with CURRENT_TIMESTAMP
func (m *defaultUserRechargeModel) DeleteSoft(ctx context.Context, data *UserRecharge) error {
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
func (m *defaultUserRechargeModel) FindOneByQuery(ctx context.Context, selectBuilder squirrel.SelectBuilder) (*UserRecharge, error) {
	selectBuilder = selectBuilder.Where("deleted_at is null").Limit(1)
	query, args, err := selectBuilder.ToSql()
	if err != nil {
		return nil, err
	}

	var resp UserRecharge
	err = m.conn.QueryRowCtx(ctx, &resp, query, args...)
	switch err {
	case nil:
		return &resp, nil
	default:
		return nil, err
	}
}

// FindOneById just like FindOneByQuery but use data.Id as query condition
func (m *defaultUserRechargeModel) FindOneById(ctx context.Context, data *UserRecharge) (*UserRecharge, error) {
	return m.FindOneByQuery(ctx, m.AllFieldsBuilder().Where("id = ?", data.Id))
}

// FindAll returns all valid rows in the table
func (m *defaultUserRechargeModel) FindAll(ctx context.Context, orderBy string) ([]*UserRecharge, error) {
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

	var resp []*UserRecharge
	err = m.conn.QueryRowsCtx(ctx, &resp, query, args...)
	switch err {
	case nil:
		return resp, nil
	default:
		return nil, err
	}
}

// Find returns all valid rows matched in the table
func (m *defaultUserRechargeModel) Find(ctx context.Context, selectBuilder squirrel.SelectBuilder) ([]*UserRecharge, error) {
	query, args, err := selectBuilder.Where("deleted_at is null").ToSql()
	if err != nil {
		return nil, err
	}

	var resp []*UserRecharge
	err = m.conn.QueryRowsCtx(ctx, &resp, query, args...)
	switch err {
	case nil:
		return resp, nil
	default:
		return nil, err
	}
}

// FindPageListByPage -
func (m *defaultUserRechargeModel) FindPageListByPage(ctx context.Context, where interface{}, page, pageSize int64, orderBy string) ([]*UserRecharge, int64, error) {
	selectBuilder := m.AllFieldsBuilder()

	if orderBy == "" {
		selectBuilder = selectBuilder.OrderBy("id DESC")
	} else {
		selectBuilder = selectBuilder.OrderBy(orderBy)
	}

	selectBuilder = selectBuilder.Where(where)

	if page < 1 {
		page = 1
	}
	offset := (page - 1) * pageSize

	total, err := m.Count(ctx, where)
	if err != nil {
		return nil, 0, err
	}

	query, args, err := selectBuilder.Where("deleted_at is null").Offset(uint64(offset)).Limit(uint64(pageSize)).ToSql()
	if err != nil {
		return nil, 0, err
	}

	var resp []*UserRecharge
	err = m.conn.QueryRowsCtx(ctx, &resp, query, args...)
	switch err {
	case nil:
		return resp, total, nil
	default:
		return nil, 0, err
	}
}

func (m *defaultUserRechargeModel) TransInsert(ctx context.Context, session sqlx.Session, data *UserRecharge) (sql.Result, error) {
	// Build the insert query using squirrel
	insertBuilder := squirrel.Insert(m.table).
		Columns("recharge_id", "user_id", "amount", "before_balance", "after_balance", "description").
		Values(data.RechargeId, data.UserId, data.Amount, data.BeforeBalance, data.AfterBalance, data.Description)

	// Convert to SQL
	query, args, err := insertBuilder.ToSql()
	if err != nil {
		return nil, err
	}

	// Execute the query within the provided session
	result, err := session.ExecCtx(ctx, query, args...)
	if err != nil {
		return nil, err
	}

	return result, nil
}

// UpdateColsByCond -
func (m *defaultUserRechargeModel) UpdateColsByCond(ctx context.Context, updateBuilder squirrel.UpdateBuilder) (sql.Result, error) {
	query, args, err := updateBuilder.ToSql()
	if err != nil {
		return nil, err
	}

	return m.conn.ExecCtx(ctx, query, args...)
}

// func (m *defaultUserRechargeModel) Count(ctx context.Context, selectBuilder squirrel.SelectBuilder) (int64, error) {
// 	// Remove non-aggregated columns from the select list
// 	countBuilder := selectBuilder.Column(squirrel.Expr("COUNT(*)"))
//
// 	// Convert to SQL
// 	query, args, err := countBuilder.ToSql()
// 	if err != nil {
// 		return 0, err
// 	}
//
// 	// Execute the query
// 	var count int64
// 	err = m.conn.QueryRowCtx(ctx, &count, query, args...)
// 	if err != nil {
// 		return 0, err
// 	}
//
// 	return count, nil
// }

func (m *defaultUserRechargeModel) Count(ctx context.Context, where interface{}) (int64, error) {
	// Create a new builder for counting
	countBuilder := squirrel.Select("COUNT(*)").From(m.table)

	// Apply where conditions to countBuilder
	countBuilder = countBuilder.Where(where).Where("deleted_at is null")

	// Convert to SQL
	query, args, err := countBuilder.ToSql()
	if err != nil {
		return 0, err
	}

	// Execute the query
	var count int64
	err = m.conn.QueryRowCtx(ctx, &count, query, args...)
	if err != nil {
		return 0, err
	}

	return count, nil
}
