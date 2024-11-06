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
	Config                config.Config
	UserJobModel          model.SysUserJobModel
	FileURLModel          model.SysFileurlModel
	CpodCacheModel        model.SysCpodCacheModel
	InferenceModel        model.SysInferenceModel
	QuotaModel            model.SysQuotaModel
	UserModel             model.SysUserModel
	VerifyCodeModel       model.VerifyCodeModel
	PriceModel            model.SysPriceModel
	JupyterlabModel       model.SysJupyterlabModel
	UserBalanceModel      model.UserBalanceModel
	UserBillingModel      model.UserBillingModel
	RechargeModel         model.UserRechargeModel
	CpodNodeModel         model.SysCpodNodeModel
	OssResourceModel      model.SysOssResourceModel
	AppModel              model.SysAppModel
	AppJobModel           model.SysAppJobModel
	ResourceSyncTaskModel model.ResourceSyncTaskModel
	EmailSender           email.Emailer
	DB                    sqlx.SqlConn
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
		Config:                c,
		UserJobModel:          model.NewSysUserJobModel(conn),
		FileURLModel:          model.NewSysFileurlModel(conn),
		CpodCacheModel:        model.NewSysCpodCacheModel(conn),
		InferenceModel:        model.NewSysInferenceModel(conn),
		QuotaModel:            model.NewSysQuotaModel(conn),
		UserModel:             model.NewSysUserModel(conn),
		VerifyCodeModel:       model.NewVerifyCodeModel(conn),
		PriceModel:            model.NewSysPriceModel(conn),
		JupyterlabModel:       model.NewSysJupyterlabModel(conn),
		UserBalanceModel:      model.NewUserBalanceModel(conn),
		UserBillingModel:      model.NewUserBillingModel(conn),
		RechargeModel:         model.NewUserRechargeModel(conn),
		CpodNodeModel:         model.NewSysCpodNodeModel(conn),
		OssResourceModel:      model.NewSysOssResourceModel(conn),
		AppModel:              model.NewSysAppModel(conn),
		AppJobModel:           model.NewSysAppJobModel(conn),
		ResourceSyncTaskModel: model.NewResourceSyncTaskModel(conn),
		EmailSender:           emailSender,
		DB:                    conn,
	}
}
