package model

import (
	"context"
	"database/sql"

	"github.com/Masterminds/squirrel"
	"github.com/zeromicro/go-zero/core/stores/sqlx"
)

type OssResourceModelMeta struct {
	Template          string `json:"template"`
	Category          string `json:"category"`
	CanFinetune       bool   `json:"can_finetune"`
	CanInference      bool   `json:"can_inference"`
	FinetuneGPUCount  int    `json:"finetune_gpu_count"`
	InferenceGPUCount int    `json:"inference_gpu_count"`
}

type OssResourceAdapterMeta struct {
	BaseModel string `json:"base_model"`
}

var _ SysOssResourceModel = (*customSysOssResourceModel)(nil)

type (
	// SysOssResourceModel is an interface to be customized, add more methods here,
	// and implement the added methods in customSysOssResourceModel.
	SysOssResourceModel interface {
		sysOssResourceModel
		AllFieldsBuilder() squirrel.SelectBuilder
		UpdateBuilder() squirrel.UpdateBuilder
		InsertBuilder() squirrel.InsertBuilder

		FindOneByQuery(ctx context.Context, selectBuilder squirrel.SelectBuilder) (*SysOssResource, error)
		FindOneById(ctx context.Context, data *SysOssResource) (*SysOssResource, error)
		FindAll(ctx context.Context, orderBy string) ([]*SysOssResource, error)
		Find(ctx context.Context, selectBuilder squirrel.SelectBuilder) ([]*SysOssResource, error)
		FindPageListByPage(ctx context.Context, selectBuilder squirrel.SelectBuilder, page, pageSize int64, orderBy string) ([]*SysOssResource, error)
		UpdateColsByCond(ctx context.Context, updateBuilder squirrel.UpdateBuilder) (sql.Result, error)
		DeleteByResourceID(ctx context.Context, resourceID string) error
	}

	customSysOssResourceModel struct {
		*defaultSysOssResourceModel
	}
)

// NewSysOssResourceModel returns a model for the database table.
func NewSysOssResourceModel(conn sqlx.SqlConn) SysOssResourceModel {
	return &customSysOssResourceModel{
		defaultSysOssResourceModel: newSysOssResourceModel(conn),
	}
}

// AllFieldsBuilder return a SelectBuilder with select("*")
func (m *defaultSysOssResourceModel) AllFieldsBuilder() squirrel.SelectBuilder {
	return squirrel.Select("*").From(m.table)
}

// UpdateBuilder return a empty UpdateBuilder
func (m *defaultSysOssResourceModel) UpdateBuilder() squirrel.UpdateBuilder {
	return squirrel.Update(m.table)
}

// InsertBuilder return a empty InsertBuilder
func (m *defaultSysOssResourceModel) InsertBuilder() squirrel.InsertBuilder {
	return squirrel.Insert(m.table)
}

// FindOneByQuery if table has deleted_at use FindOneByQuery instead of FindOne
func (m *defaultSysOssResourceModel) FindOneByQuery(ctx context.Context, selectBuilder squirrel.SelectBuilder) (*SysOssResource, error) {
	selectBuilder = selectBuilder.Limit(1)
	query, args, err := selectBuilder.ToSql()
	if err != nil {
		return nil, err
	}

	var resp SysOssResource
	err = m.conn.QueryRowCtx(ctx, &resp, query, args...)
	switch err {
	case nil:
		return &resp, nil
	default:
		return nil, err
	}
}

// FindOneById just like FindOneByQuery but use data.Id as query condition
func (m *defaultSysOssResourceModel) FindOneById(ctx context.Context, data *SysOssResource) (*SysOssResource, error) {
	return m.FindOneByQuery(ctx, m.AllFieldsBuilder().Where("id = ?", data.Id))
}

// FindAll returns all valid rows in the table
func (m *defaultSysOssResourceModel) FindAll(ctx context.Context, orderBy string) ([]*SysOssResource, error) {
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

	var resp []*SysOssResource
	err = m.conn.QueryRowsCtx(ctx, &resp, query, args...)
	switch err {
	case nil:
		return resp, nil
	default:
		return nil, err
	}
}

// Find returns all valid rows matched in the table
func (m *defaultSysOssResourceModel) Find(ctx context.Context, selectBuilder squirrel.SelectBuilder) ([]*SysOssResource, error) {
	query, args, err := selectBuilder.ToSql()
	if err != nil {
		return nil, err
	}

	var resp []*SysOssResource
	err = m.conn.QueryRowsCtx(ctx, &resp, query, args...)
	switch err {
	case nil:
		return resp, nil
	default:
		return nil, err
	}
}

// FindPageListByPage -
func (m *defaultSysOssResourceModel) FindPageListByPage(ctx context.Context, selectBuilder squirrel.SelectBuilder, page, pageSize int64, orderBy string) ([]*SysOssResource, error) {
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

	var resp []*SysOssResource
	err = m.conn.QueryRowsCtx(ctx, &resp, query, args...)
	switch err {
	case nil:
		return resp, nil
	default:
		return nil, err
	}
}

// UpdateColsByCond -
func (m *defaultSysOssResourceModel) UpdateColsByCond(ctx context.Context, updateBuilder squirrel.UpdateBuilder) (sql.Result, error) {
	query, args, err := updateBuilder.ToSql()
	if err != nil {
		return nil, err
	}

	return m.conn.ExecCtx(ctx, query, args...)
}

func (m *defaultSysOssResourceModel) DeleteByResourceID(ctx context.Context, resourceID string) error {
	deleteBuilder := squirrel.Delete(m.table)
	query, args, err := deleteBuilder.Where(squirrel.Eq{"resource_id": resourceID}).ToSql()
	if err != nil {
		return err
	}

	_, err = m.conn.ExecCtx(ctx, query, args)
	return err
}
