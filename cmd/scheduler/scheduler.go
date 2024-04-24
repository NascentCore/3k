package main

import (
	"fmt"
	"log"
	"os"
	"strconv"
	"sxwl/3k/internal/scheduler/config"
	"sxwl/3k/internal/scheduler/handler"
	"sxwl/3k/internal/scheduler/svc"
	"sxwl/3k/pkg/storage"

	"github.com/zeromicro/go-zero/core/conf"
	"github.com/zeromicro/go-zero/rest"
)

func main() {
	var configFile string
	env := os.Getenv("SCHEDULER_ENV")
	switch env {
	case "prod":
		configFile = "etc/scheduler-api_prod.yaml"
	case "test":
		configFile = "etc/scheduler-api_test.yaml"
	case "k8s":
		configFile = "etc/scheduler-api_k8s.yaml"
	case "dev":
		configFile = "etc/scheduler-api.yaml"
	default:
		log.Fatalf("env SCHEDULER_ENV not defined")
	}

	var c config.Config
	conf.MustLoad(configFile, &c)

	// insert dsn
	dsn := os.Getenv("SCHEDULER_DSN")
	c.DB.DataSource = dsn

	// insert access
	uploadAccessID := os.Getenv("UPLOAD_ACCESS_ID")
	uploadAccessKey := os.Getenv("UPLOAD_ACCESS_KEY")
	if uploadAccessID == "" || uploadAccessKey == "" {
		log.Fatalf("env UPLOAD_ACCESS_ID or UPLOAD_ACCESS_KEY not defined")
	}
	c.OSSAccess.UploadAccessID = uploadAccessID
	c.OSSAccess.UploadAccessKey = uploadAccessKey
	adminAccessID := os.Getenv("ADMIN_ACCESS_ID")
	adminAccessKey := os.Getenv("ADMIN_ACCESS_KEY")
	if adminAccessID == "" || adminAccessKey == "" {
		log.Fatalf("env ADMIN_ACCESS_ID or ADMIN_ACCESS_KEY not defined")
	}
	c.OSSAccess.AdminAccessID = adminAccessID
	c.OSSAccess.AdminAccessKey = adminAccessKey

	// insert inference url format
	inferenceUrlFormat := os.Getenv("INFERENCE_URL_FORMAT")
	if inferenceUrlFormat != "" {
		c.Inference.UrlFormat = inferenceUrlFormat
	}

	// insert k8s config
	c.K8S.BaseApi = os.Getenv("K8S_BASE_API")
	c.K8S.BaseUrl = os.Getenv("K8S_BASE_URL")

	// insert auth secret
	c.Auth.Secret = os.Getenv("AUTH_SECRET")

	// init oss
	storage.InitClient(adminAccessID, adminAccessKey)

	// insert email config
	c.Email.Host = os.Getenv("EMAIL_HOST")
	c.Email.Port, _ = strconv.Atoi(os.Getenv("EMAIL_PORT"))
	c.Email.Username = os.Getenv("EMAIL_USERNAME")
	c.Email.SenderName = os.Getenv("EMAIL_SENDER_NAME")
	c.Email.Password = os.Getenv("EMAIL_PASSWORD")

	// insert ras key
	c.Rsa.PrivateKey = os.Getenv("RSA_PRIVATE_KEY")

	// error handler
	handler.InitErrorHandler()

	server := rest.MustNewServer(c.RestConf)
	defer server.Stop()

	ctx := svc.NewServiceContext(c)
	handler.RegisterHandlers(server, ctx)
	handler.RegisterCustomHandlers(server, ctx)

	fmt.Printf("Starting server at %s:%d...\n", c.Host, c.Port)
	server.Start()
}
