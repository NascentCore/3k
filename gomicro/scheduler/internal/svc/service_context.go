package svc

import (
	"sxwl/3k/gomicro/scheduler/internal/config"
	"sxwl/3k/gomicro/scheduler/internal/model"

	"github.com/zeromicro/go-zero/core/stores/sqlx"
)

type ServiceContext struct {
	Config         config.Config
	CpodMainModel  model.SysCpodMainModel
	UserJobModel   model.SysUserJobModel
	FileURLModel   model.SysFileurlModel
	CpodCacheModel model.SysCpodCacheModel
}

func NewServiceContext(c config.Config) *ServiceContext {
	conn := sqlx.NewMysql(c.DB.DataSource)
	return &ServiceContext{
		Config:         c,
		CpodMainModel:  model.NewSysCpodMainModel(conn),
		UserJobModel:   model.NewSysUserJobModel(conn),
		FileURLModel:   model.NewSysFileurlModel(conn),
		CpodCacheModel: model.NewSysCpodCacheModel(conn),
	}
}
