// Code generated by goctl. DO NOT EDIT.

package model

import (
	"context"
	"database/sql"
	"fmt"
	"strings"
	"time"

	"github.com/zeromicro/go-zero/core/stores/builder"
	"github.com/zeromicro/go-zero/core/stores/sqlc"
	"github.com/zeromicro/go-zero/core/stores/sqlx"
	"github.com/zeromicro/go-zero/core/stringx"
)

var (
	sysCpodCacheFieldNames          = builder.RawFieldNames(&SysCpodCache{})
	sysCpodCacheRows                = strings.Join(sysCpodCacheFieldNames, ",")
	sysCpodCacheRowsExpectAutoSet   = strings.Join(stringx.Remove(sysCpodCacheFieldNames, "`id`", "`create_at`", "`create_time`", "`created_at`", "`update_at`", "`update_time`", "`updated_at`"), ",")
	sysCpodCacheRowsWithPlaceHolder = strings.Join(stringx.Remove(sysCpodCacheFieldNames, "`id`", "`create_at`", "`create_time`", "`created_at`", "`update_at`", "`update_time`", "`updated_at`"), "=?,") + "=?"
)

type (
	sysCpodCacheModel interface {
		Insert(ctx context.Context, data *SysCpodCache) (sql.Result, error)
		FindOne(ctx context.Context, id int64) (*SysCpodCache, error)
		Update(ctx context.Context, data *SysCpodCache) error
		Delete(ctx context.Context, id int64) error
	}

	defaultSysCpodCacheModel struct {
		conn  sqlx.SqlConn
		table string
	}

	SysCpodCache struct {
		Id                int64     `db:"id"`                  // 自增ID
		CpodId            string    `db:"cpod_id"`             // cpod id
		CpodVersion       string    `db:"cpod_version"`        // pod 版本
		DataType          int64     `db:"data_type"`           // 缓存的数据类型
		DataName          string    `db:"data_name"`           // 缓存的数据名字
		DataId            string    `db:"data_id"`             // 缓存的数据id
		DataSize          int64     `db:"data_size"`           // 资源体积(字节)
		DataSource        string    `db:"data_source"`         // 缓存的数据来源
		Template          string    `db:"template"`            // 模型推理模版
		FinetuneGpuCount  int64     `db:"finetune_gpu_count"`  // 微调需要最少GPU
		InferenceGpuCount int64     `db:"inference_gpu_count"` // 推理需要最少GPU
		CreatedAt         time.Time `db:"created_at"`          // 创建时间
		UpdatedAt         time.Time `db:"updated_at"`          // 更新时间
	}
)

func newSysCpodCacheModel(conn sqlx.SqlConn) *defaultSysCpodCacheModel {
	return &defaultSysCpodCacheModel{
		conn:  conn,
		table: "`sys_cpod_cache`",
	}
}

func (m *defaultSysCpodCacheModel) Delete(ctx context.Context, id int64) error {
	query := fmt.Sprintf("delete from %s where `id` = ?", m.table)
	_, err := m.conn.ExecCtx(ctx, query, id)
	return err
}

func (m *defaultSysCpodCacheModel) FindOne(ctx context.Context, id int64) (*SysCpodCache, error) {
	query := fmt.Sprintf("select %s from %s where `id` = ? limit 1", sysCpodCacheRows, m.table)
	var resp SysCpodCache
	err := m.conn.QueryRowCtx(ctx, &resp, query, id)
	switch err {
	case nil:
		return &resp, nil
	case sqlc.ErrNotFound:
		return nil, ErrNotFound
	default:
		return nil, err
	}
}

func (m *defaultSysCpodCacheModel) Insert(ctx context.Context, data *SysCpodCache) (sql.Result, error) {
	query := fmt.Sprintf("insert into %s (%s) values (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)", m.table, sysCpodCacheRowsExpectAutoSet)
	ret, err := m.conn.ExecCtx(ctx, query, data.CpodId, data.CpodVersion, data.DataType, data.DataName, data.DataId, data.DataSize, data.DataSource, data.Template, data.FinetuneGpuCount, data.InferenceGpuCount)
	return ret, err
}

func (m *defaultSysCpodCacheModel) Update(ctx context.Context, data *SysCpodCache) error {
	query := fmt.Sprintf("update %s set %s where `id` = ?", m.table, sysCpodCacheRowsWithPlaceHolder)
	_, err := m.conn.ExecCtx(ctx, query, data.CpodId, data.CpodVersion, data.DataType, data.DataName, data.DataId, data.DataSize, data.DataSource, data.Template, data.FinetuneGpuCount, data.InferenceGpuCount, data.Id)
	return err
}

func (m *defaultSysCpodCacheModel) tableName() string {
	return m.table
}
