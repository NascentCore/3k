// Code generated by goctl. DO NOT EDIT.

package model

import (
	"context"
	"database/sql"
	"fmt"
	"strings"

	"github.com/zeromicro/go-zero/core/stores/builder"
	"github.com/zeromicro/go-zero/core/stores/sqlc"
	"github.com/zeromicro/go-zero/core/stores/sqlx"
	"github.com/zeromicro/go-zero/core/stringx"
)

var (
	sysUserJobFieldNames          = builder.RawFieldNames(&SysUserJob{})
	sysUserJobRows                = strings.Join(sysUserJobFieldNames, ",")
	sysUserJobRowsExpectAutoSet   = strings.Join(stringx.Remove(sysUserJobFieldNames, "`job_id`", "`create_at`"), ",")
	sysUserJobRowsWithPlaceHolder = strings.Join(stringx.Remove(sysUserJobFieldNames, "`job_id`", "`create_at`"), "=?,") + "=?"
)

type (
	sysUserJobModel interface {
		Insert(ctx context.Context, data *SysUserJob) (sql.Result, error)
		FindOne(ctx context.Context, jobId int64) (*SysUserJob, error)
		Update(ctx context.Context, data *SysUserJob) error
		Delete(ctx context.Context, jobId int64) error
	}

	defaultSysUserJobModel struct {
		conn  sqlx.SqlConn
		table string
	}

	SysUserJob struct {
		JobId               int64          `db:"job_id"`                // ID
		UserId              int64          `db:"user_id"`               // 用户ID
		CpodId              sql.NullString `db:"cpod_id"`               // cpod id
		WorkStatus          int64          `db:"work_status"`           // 状态：1失败、0运行、2完成
		ObtainStatus        int64          `db:"obtain_status"`         // 状态：1不需要下发、0需要下发
		BillingStatus       int64          `db:"billing_status"`        // 账单状态（0 未结清、1 已结清）
		JobName             sql.NullString `db:"job_name"`              // 任务名称
		GpuNumber           sql.NullInt64  `db:"gpu_number"`            // GPU数量
		GpuType             sql.NullString `db:"gpu_type"`              // GPU型号
		CkptPath            sql.NullString `db:"ckpt_path"`             // cktp路径
		CkptVol             sql.NullString `db:"ckpt_vol"`              // cktp容量
		ModelPath           sql.NullString `db:"model_path"`            // save model路径
		ModelVol            sql.NullString `db:"model_vol"`             // save model容量
		ImagePath           sql.NullString `db:"image_path"`            // 镜像路径
		HfUrl               sql.NullString `db:"hf_url"`                // HF公开训练数据URL
		DatasetPath         sql.NullString `db:"dataset_path"`          // 挂载路径
		JobType             sql.NullString `db:"job_type"`              // 任务类型 mpi
		StopType            sql.NullInt64  `db:"stop_type"`             // 0 自然终止 1设定时长
		StopTime            sql.NullInt64  `db:"stop_time"`             // 设定时常以分钟为单位
		CreateTime          sql.NullTime   `db:"create_time"`           // 创建日期
		UpdateTime          sql.NullTime   `db:"update_time"`           // 更新时间
		PretrainedModelName sql.NullString `db:"pretrained_model_name"` // 模型基座名称
		RunCommand          sql.NullString `db:"run_command"`           // 模型启动命令
		CallbackUrl         sql.NullString `db:"callback_url"`          // 第三方回调接口url
		PretrainedModelPath sql.NullString `db:"pretrained_model_path"` // 模型基座路径
		DatasetName         sql.NullString `db:"dataset_name"`          // 挂载路径名称
		JsonAll             sql.NullString `db:"json_all"`              // json数据全包
		Deleted             int64          `db:"deleted"`               // 逻辑删除 0 未删除 1逻辑删除
	}
)

func newSysUserJobModel(conn sqlx.SqlConn) *defaultSysUserJobModel {
	return &defaultSysUserJobModel{
		conn:  conn,
		table: "`sys_user_job`",
	}
}

func (m *defaultSysUserJobModel) Delete(ctx context.Context, jobId int64) error {
	query := fmt.Sprintf("delete from %s where `job_id` = ?", m.table)
	_, err := m.conn.ExecCtx(ctx, query, jobId)
	return err
}

func (m *defaultSysUserJobModel) FindOne(ctx context.Context, jobId int64) (*SysUserJob, error) {
	query := fmt.Sprintf("select %s from %s where `job_id` = ? limit 1", sysUserJobRows, m.table)
	var resp SysUserJob
	err := m.conn.QueryRowCtx(ctx, &resp, query, jobId)
	switch err {
	case nil:
		return &resp, nil
	case sqlc.ErrNotFound:
		return nil, ErrNotFound
	default:
		return nil, err
	}
}

func (m *defaultSysUserJobModel) Insert(ctx context.Context, data *SysUserJob) (sql.Result, error) {
	query := fmt.Sprintf("insert into %s (%s) values (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)", m.table, sysUserJobRowsExpectAutoSet)
	ret, err := m.conn.ExecCtx(ctx, query, data.UserId, data.CpodId, data.WorkStatus, data.ObtainStatus, data.BillingStatus, data.JobName, data.GpuNumber, data.GpuType, data.CkptPath, data.CkptVol, data.ModelPath, data.ModelVol, data.ImagePath, data.HfUrl, data.DatasetPath, data.JobType, data.StopType, data.StopTime, data.CreateTime, data.UpdateTime, data.PretrainedModelName, data.RunCommand, data.CallbackUrl, data.PretrainedModelPath, data.DatasetName, data.JsonAll, data.Deleted)
	return ret, err
}

func (m *defaultSysUserJobModel) Update(ctx context.Context, data *SysUserJob) error {
	query := fmt.Sprintf("update %s set %s where `job_id` = ?", m.table, sysUserJobRowsWithPlaceHolder)
	_, err := m.conn.ExecCtx(ctx, query, data.UserId, data.CpodId, data.WorkStatus, data.ObtainStatus, data.BillingStatus, data.JobName, data.GpuNumber, data.GpuType, data.CkptPath, data.CkptVol, data.ModelPath, data.ModelVol, data.ImagePath, data.HfUrl, data.DatasetPath, data.JobType, data.StopType, data.StopTime, data.CreateTime, data.UpdateTime, data.PretrainedModelName, data.RunCommand, data.CallbackUrl, data.PretrainedModelPath, data.DatasetName, data.JsonAll, data.Deleted, data.JobId)
	return err
}

func (m *defaultSysUserJobModel) tableName() string {
	return m.table
}
