package svc

import (
	"log"
	"os"
	"sxwl/3k/internal/scheduler/config"
	"sxwl/3k/internal/scheduler/model"
	"sxwl/3k/pkg/email"
	"sxwl/3k/pkg/fs"

	"github.com/zeromicro/go-zero/core/stores/sqlx"
)

type ServiceContext struct {
	Config          config.Config
	CpodMainModel   model.SysCpodMainModel
	UserJobModel    model.SysUserJobModel
	FileURLModel    model.SysFileurlModel
	CpodCacheModel  model.SysCpodCacheModel
	InferenceModel  model.SysInferenceModel
	QuotaModel      model.SysQuotaModel
	UserModel       model.SysUserModel
	VerifyCodeModel model.VerifyCodeModel
	JupyterlabModel model.SysJupyterlabModel
	EmailSender     email.Emailer
}

func NewServiceContext(c config.Config) *ServiceContext {
	conn := sqlx.NewMysql(c.DB.DataSource)

	emailSender := email.NewSMTPClient(&c.Email)
	ftlList, err := fs.ListFilesInDir("ftl/", ".ftl")
	if err != nil {
		log.Fatalf("fs.ListFilesInDir err=%s", err)
	}

	for _, ftlPath := range ftlList {
		ftlName := fs.FileNameWithoutExtension(ftlPath)
		ftlFile, err := os.Open(ftlPath)
		if err != nil {
			log.Fatalf("Error opening template file: %v", err)
		}

		err = emailSender.AddTemplate(ftlName, ftlFile)
		if err != nil {
			log.Fatalf("AddTemplate ftlName err=%s", err)
		}
		_ = ftlFile.Close()
	}

	return &ServiceContext{
		Config:          c,
		CpodMainModel:   model.NewSysCpodMainModel(conn),
		UserJobModel:    model.NewSysUserJobModel(conn),
		FileURLModel:    model.NewSysFileurlModel(conn),
		CpodCacheModel:  model.NewSysCpodCacheModel(conn),
		InferenceModel:  model.NewSysInferenceModel(conn),
		QuotaModel:      model.NewSysQuotaModel(conn),
		UserModel:       model.NewSysUserModel(conn),
		VerifyCodeModel: model.NewVerifyCodeModel(conn),
		JupyterlabModel: model.NewSysJupyterlabModel(conn),
		EmailSender:     emailSender,
	}
}
