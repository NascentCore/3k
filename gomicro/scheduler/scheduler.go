package main

import (
	"flag"
	"fmt"

	"sxwl/3k/gomicro/scheduler/internal/config"
	"sxwl/3k/gomicro/scheduler/internal/handler"
	"sxwl/3k/gomicro/scheduler/internal/svc"

	"github.com/zeromicro/go-zero/core/conf"
	"github.com/zeromicro/go-zero/rest"
	_ "sxwl/3k/gomicro/pkg/sxwlzero"
)

var configFile = flag.String("f", "etc/scheduler-api.yaml", "the config file")

func main() {
	flag.Parse()

	var c config.Config
	conf.MustLoad(*configFile, &c)

	server := rest.MustNewServer(c.RestConf)
	defer server.Stop()

	ctx := svc.NewServiceContext(c)
	handler.RegisterHandlers(server, ctx)

	fmt.Printf("Starting server at %s:%d...\n", c.Host, c.Port)
	server.Start()
}
