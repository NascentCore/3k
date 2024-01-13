package main

import (
	"fmt"
	"log"
	"os"

	"sxwl/3k/internal/scheduler/config"
	"sxwl/3k/internal/scheduler/handler"
	"sxwl/3k/internal/scheduler/svc"

	"github.com/zeromicro/go-zero/core/conf"
	"github.com/zeromicro/go-zero/rest"
)

func main() {
	var configFile string
	env := os.Getenv("SCHEDULER_ENV")
	dsn := os.Getenv("SCHEDULER_DSN")
	switch env {
	case "prod":
		configFile = "etc/scheduler-api_prod.yaml"
	case "test":
		configFile = "etc/scheduler-api_test.yaml"
	case "dev":
		configFile = "etc/scheduler-api.yaml"
	default:
		log.Fatalf("env SCHEDULER_ENV not defined")
	}

	var c config.Config
	conf.MustLoad(configFile, &c)

	// insert dsn
	c.DB.DataSource = dsn

	server := rest.MustNewServer(c.RestConf)
	defer server.Stop()

	ctx := svc.NewServiceContext(c)
	handler.RegisterHandlers(server, ctx)
	handler.RegisterCustomHandlers(server, ctx)

	fmt.Printf("Starting server at %s:%d...\n", c.Host, c.Port)
	server.Start()
}