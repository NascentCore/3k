package svc

import (
	"sxwl/3k/internal/scheduler/config"
	"sxwl/3k/internal/scheduler/model"

	"github.com/zeromicro/go-zero/core/stores/sqlx"
)

type ServiceContext struct {
	Config         config.Config
	CpodMainModel  model.SysCpodMainModel
	UserJobModel   model.SysUserJobModel
	FileURLModel   model.SysFileurlModel
	CpodCacheModel model.SysCpodCacheModel
	InferenceModel model.SysInferenceModel
	QuotaModel     model.SysQuotaModel
	UserModel      model.SysUserModel
}

func NewServiceContext(c config.Config) *ServiceContext {
	conn := sqlx.NewMysql(c.DB.DataSource)
	return &ServiceContext{
		Config:         c,
		CpodMainModel:  model.NewSysCpodMainModel(conn),
		UserJobModel:   model.NewSysUserJobModel(conn),
		FileURLModel:   model.NewSysFileurlModel(conn),
		CpodCacheModel: model.NewSysCpodCacheModel(conn),
		InferenceModel: model.NewSysInferenceModel(conn),
		QuotaModel:     model.NewSysQuotaModel(conn),
		UserModel:      model.NewSysUserModel(conn),
	}
}
