package main

import (
	"fmt"
	"log"
	"os"

	"sxwl/3k/gomicro/scheduler/internal/config"
	"sxwl/3k/gomicro/scheduler/internal/handler"
	"sxwl/3k/gomicro/scheduler/internal/svc"

	"github.com/zeromicro/go-zero/core/conf"
	"github.com/zeromicro/go-zero/rest"
	_ "sxwl/3k/gomicro/pkg/sxwlzero"
)

func main() {
	var configFile string
	env := os.Getenv("SCHEDULER_ENV")
	switch env {
	case "prod":
		configFile = "etc/scheduler-api_prod.yaml"
	case "test":
		configFile = "etc/scheduler-api_test.yaml"
	default:
		log.Fatalf("env SCHEDULER_ENV not defined")
	}

	var c config.Config
	conf.MustLoad(configFile, &c)

	server := rest.MustNewServer(c.RestConf)
	defer server.Stop()

	ctx := svc.NewServiceContext(c)
	handler.RegisterHandlers(server, ctx)

	fmt.Printf("Starting server at %s:%d...\n", c.Host, c.Port)
	server.Start()
}
