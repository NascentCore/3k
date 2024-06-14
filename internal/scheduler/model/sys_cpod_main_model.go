package model

import (
	"context"
	"database/sql"

	"github.com/zeromicro/go-zero/core/stores/sqlx"

	"github.com/Masterminds/squirrel"
)

// const (
// 	JobStatusObtainNeedSend    = 0
// 	JobStatusObtainNotNeedSend = 1
// 	JobStatusWorkerRunning     = 0
// 	JobStatusWorkerFail        = 1
// 	JobStatusWorkerSuccess     = 2
// 	JobStatusWorkerUrlSuccess  = 3
// 	JobStatusWorkerPreparing   = 4
// )

var _ SysCpodMainModel = (*customSysCpodMainModel)(nil)

type (
	// SysCpodMainModel is an interface to be customized, add more methods here,
	// and implement the added methods in customSysCpodMainModel.
	SysCpodMainModel interface {
		sysCpodMainModel
		AllFieldsBuilder() squirrel.SelectBuilder
		UpdateBuilder() squirrel.UpdateBuilder

		FindOneByQuery(ctx context.Context, selectBuilder squirrel.SelectBuilder) (*SysCpodMain, error)
		FindOneById(ctx context.Context, data *SysCpodMain) (*SysCpodMain, error)
		FindAll(ctx context.Context, orderBy string) ([]*SysCpodMain, error)
		Find(ctx context.Context, selectBuilder squirrel.SelectBuilder) ([]*SysCpodMain, error)
		FindPageListByPage(ctx context.Context, selectBuilder squirrel.SelectBuilder, page, pageSize int64, orderBy string) ([]*SysCpodMain, error)
		UpdateColsByCond(ctx context.Context, updateBuilder squirrel.UpdateBuilder) (sql.Result, error)
		GpuTypeAndPrice(ctx context.Context) ([]*GPUTypePrice, error)
	}

	customSysCpodMainModel struct {
		*defaultSysCpodMainModel
	}

	GPUTypePrice struct {
		GPUProd        string  `json:"gpu_prod" db:"gpu_prod"`
		GPUAllocatable int64   `json:"gpu_allocatable" db:"gpu_allocatable"`
		Amount         float64 `json:"amount" db:"amount"`
	}
)

// NewSysCpodMainModel returns a model for the database table.
func NewSysCpodMainModel(conn sqlx.SqlConn) SysCpodMainModel {
	return &customSysCpodMainModel{
		defaultSysCpodMainModel: newSysCpodMainModel(conn),
	}
}

// AllFieldsBuilder return a SelectBuilder with select("*")
func (m *defaultSysCpodMainModel) AllFieldsBuilder() squirrel.SelectBuilder {
	return squirrel.Select("*").From(m.table)
}

// UpdateBuilder return a empty UpdateBuilder
func (m *defaultSysCpodMainModel) UpdateBuilder() squirrel.UpdateBuilder {
	return squirrel.Update(m.table)
}

// FindOneByQuery if table has deleted_at use FindOneByQuery instead of FindOne
func (m *defaultSysCpodMainModel) FindOneByQuery(ctx context.Context, selectBuilder squirrel.SelectBuilder) (*SysCpodMain, error) {
	selectBuilder = selectBuilder.Limit(1)
	query, args, err := selectBuilder.ToSql()
	if err != nil {
		return nil, err
	}

	var resp SysCpodMain
	err = m.conn.QueryRowCtx(ctx, &resp, query, args...)
	switch err {
	case nil:
		return &resp, nil
	default:
		return nil, err
	}
}

// FindOneById just like FindOneByQuery but use data.Id as query condition
func (m *defaultSysCpodMainModel) FindOneById(ctx context.Context, data *SysCpodMain) (*SysCpodMain, error) {
	return m.FindOneByQuery(ctx, m.AllFieldsBuilder().Where("main_id = ?", data.MainId))
}

// FindAll returns all valid rows in the table
func (m *defaultSysCpodMainModel) FindAll(ctx context.Context, orderBy string) ([]*SysCpodMain, error) {
	selectBuilder := m.AllFieldsBuilder()
	if orderBy == "" {
		selectBuilder = selectBuilder.OrderBy("main_id DESC")
	} else {
		selectBuilder = selectBuilder.OrderBy(orderBy)
	}

	query, args, err := selectBuilder.ToSql()
	if err != nil {
		return nil, err
	}

	var resp []*SysCpodMain
	err = m.conn.QueryRowsCtx(ctx, &resp, query, args...)
	switch err {
	case nil:
		return resp, nil
	default:
		return nil, err
	}
}

func (m *defaultSysCpodMainModel) Find(ctx context.Context, selectBuilder squirrel.SelectBuilder) ([]*SysCpodMain, error) {
	query, args, err := selectBuilder.ToSql()
	if err != nil {
		return nil, err
	}

	var resp []*SysCpodMain
	err = m.conn.QueryRowsCtx(ctx, &resp, query, args...)
	switch err {
	case nil:
		return resp, nil
	default:
		return nil, err
	}
}

// FindPageListByPage -
func (m *defaultSysCpodMainModel) FindPageListByPage(ctx context.Context, selectBuilder squirrel.SelectBuilder, page, pageSize int64, orderBy string) ([]*SysCpodMain, error) {
	if orderBy == "" {
		selectBuilder = selectBuilder.OrderBy("main_id DESC")
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

	var resp []*SysCpodMain
	err = m.conn.QueryRowsCtx(ctx, &resp, query, args...)
	switch err {
	case nil:
		return resp, nil
	default:
		return nil, err
	}
}

// UpdateColsByCond -
func (m *defaultSysCpodMainModel) UpdateColsByCond(ctx context.Context, updateBuilder squirrel.UpdateBuilder) (sql.Result, error) {
	query, args, err := updateBuilder.ToSql()
	if err != nil {
		return nil, err
	}

	return m.conn.ExecCtx(ctx, query, args...)
}

func (m *defaultSysCpodMainModel) GpuTypeAndPrice(ctx context.Context) ([]*GPUTypePrice, error) {
	gpuPrices := make([]*GPUTypePrice, 0)
	err := m.conn.QueryRowsCtx(ctx, &gpuPrices, `select distinct c.gpu_prod, c.gpu_allocatable, p.amount
		from sys_cpod_main c,
		sys_price p
		where c.gpu_prod != ''
		AND c.gpu_prod = p.gpu_prod
		AND c.update_time > NOW() - INTERVAL 30 MINUTE;
	`)
	if err != nil {
		return nil, err
	}

	return gpuPrices, nil
}
