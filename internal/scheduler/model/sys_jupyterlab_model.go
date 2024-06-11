package model

import (
	"context"
	"database/sql"

	"github.com/Masterminds/squirrel"
	"github.com/zeromicro/go-zero/core/stores/sqlx"
)

var _ SysJupyterlabModel = (*customSysJupyterlabModel)(nil)

const (
	JupyterStatusDescWaitDeploy = "waitdeploy"
	JupyterStatusDescDeploying  = "deploying"
	JupyterStatusDescDeployed   = "deployed"
	JupyterStatusDescStopped    = "stopped"
	JupyterStatusDescReady      = "ready"
	JupyterStatusDescNotReady   = "notready"
	JupyterStatusDescFailed     = "failed"
)

const (
	JupyterStatusWaitDeploy = 0
	JupyterStatusDeploying  = 1
	JupyterStatusDeployed   = 2
	JupyterStatusStopped    = 3
	JupyterStatusFailed     = 4
)

const (
	JupyterReplicasRunning = 1
	JupyterReplicasStop    = 0
)

var (
	JupyterStatusToDesc = map[int64]string{
		JupyterStatusWaitDeploy: JupyterStatusDescWaitDeploy,
		JupyterStatusDeploying:  JupyterStatusDescDeploying,
		JupyterStatusDeployed:   JupyterStatusDescDeployed,
		JupyterStatusStopped:    JupyterStatusDescStopped,
		JupyterStatusFailed:     JupyterStatusDescFailed,
	}
)

type (
	// SysJupyterlabModel is an interface to be customized, add more methods here,
	// and implement the added methods in customSysJupyterlabModel.
	SysJupyterlabModel interface {
		sysJupyterlabModel
		AllFieldsBuilder() squirrel.SelectBuilder
		UpdateBuilder() squirrel.UpdateBuilder

		FindOneByQuery(ctx context.Context, selectBuilder squirrel.SelectBuilder) (*SysJupyterlab, error)
		FindOneById(ctx context.Context, data *SysJupyterlab) (*SysJupyterlab, error)
		FindAll(ctx context.Context, orderBy string) ([]*SysJupyterlab, error)
		Find(ctx context.Context, selectBuilder squirrel.SelectBuilder) ([]*SysJupyterlab, error)
		FindPageListByPage(ctx context.Context, selectBuilder squirrel.SelectBuilder, page, pageSize int64, orderBy string) ([]*SysJupyterlab, error)
		UpdateColsByCond(ctx context.Context, updateBuilder squirrel.UpdateBuilder) (sql.Result, error)
	}

	customSysJupyterlabModel struct {
		*defaultSysJupyterlabModel
	}
)

// NewSysJupyterlabModel returns a model for the database table.
func NewSysJupyterlabModel(conn sqlx.SqlConn) SysJupyterlabModel {
	return &customSysJupyterlabModel{
		defaultSysJupyterlabModel: newSysJupyterlabModel(conn),
	}
}

// AllFieldsBuilder return a SelectBuilder with select("*")
func (m *defaultSysJupyterlabModel) AllFieldsBuilder() squirrel.SelectBuilder {
	return squirrel.Select("*").From(m.table)
}

// UpdateBuilder return a empty UpdateBuilder
func (m *defaultSysJupyterlabModel) UpdateBuilder() squirrel.UpdateBuilder {
	return squirrel.Update(m.table)
}

// FindOneByQuery if table has deleted_at use FindOneByQuery instead of FindOne
func (m *defaultSysJupyterlabModel) FindOneByQuery(ctx context.Context, selectBuilder squirrel.SelectBuilder) (*SysJupyterlab, error) {
	selectBuilder = selectBuilder.Limit(1)
	query, args, err := selectBuilder.ToSql()
	if err != nil {
		return nil, err
	}

	var resp SysJupyterlab
	err = m.conn.QueryRowCtx(ctx, &resp, query, args...)
	switch err {
	case nil:
		return &resp, nil
	default:
		return nil, err
	}
}

// FindOneById just like FindOneByQuery but use data.Id as query condition
func (m *defaultSysJupyterlabModel) FindOneById(ctx context.Context, data *SysJupyterlab) (*SysJupyterlab, error) {
	return m.FindOneByQuery(ctx, m.AllFieldsBuilder().Where("id = ?", data.Id))
}

// FindAll returns all valid rows in the table
func (m *defaultSysJupyterlabModel) FindAll(ctx context.Context, orderBy string) ([]*SysJupyterlab, error) {
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

	var resp []*SysJupyterlab
	err = m.conn.QueryRowsCtx(ctx, &resp, query, args...)
	switch err {
	case nil:
		return resp, nil
	default:
		return nil, err
	}
}

// Find returns all valid rows matched in the table
func (m *defaultSysJupyterlabModel) Find(ctx context.Context, selectBuilder squirrel.SelectBuilder) ([]*SysJupyterlab, error) {
	query, args, err := selectBuilder.ToSql()
	if err != nil {
		return nil, err
	}

	var resp []*SysJupyterlab
	err = m.conn.QueryRowsCtx(ctx, &resp, query, args...)
	switch err {
	case nil:
		return resp, nil
	default:
		return nil, err
	}
}

// FindPageListByPage -
func (m *defaultSysJupyterlabModel) FindPageListByPage(ctx context.Context, selectBuilder squirrel.SelectBuilder, page, pageSize int64, orderBy string) ([]*SysJupyterlab, error) {
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

	var resp []*SysJupyterlab
	err = m.conn.QueryRowsCtx(ctx, &resp, query, args...)
	switch err {
	case nil:
		return resp, nil
	default:
		return nil, err
	}
}

// UpdateColsByCond -
func (m *defaultSysJupyterlabModel) UpdateColsByCond(ctx context.Context, updateBuilder squirrel.UpdateBuilder) (sql.Result, error) {
	query, args, err := updateBuilder.ToSql()
	if err != nil {
		return nil, err
	}

	return m.conn.ExecCtx(ctx, query, args...)
}
