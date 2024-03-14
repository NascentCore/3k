package main

import (
	"fmt"
	"log"
	"os"
	"sxwl/3k/internal/gateway/config"
	"sxwl/3k/internal/gateway/gateway"

	"github.com/zeromicro/go-zero/core/conf"
	"github.com/zeromicro/go-zero/rest"
)

func main() {
	var configFile string
	env := os.Getenv("GATEWAY_ENV")
	authSecret := os.Getenv("AUTH_SECRET")
	dsn := os.Getenv("GATEWAY_DSN")

	switch env {
	case "prod":
		configFile = "etc/gateway-api_prod.yaml"
	case "test":
		configFile = "etc/gateway-api_test.yaml"
	case "k8s":
		configFile = "etc/gateway-api_k8s.yaml"
	case "dev":
		configFile = "etc/gateway-api.yaml"
	default:
		log.Fatalf("env GATEWAY_ENV not defined")
	}

	var c config.Config
	conf.MustLoad(configFile, &c)

	// init matcher
	gateway.GlobalMatcher = gateway.NewMatcher(c.Gateway)

	server := rest.MustNewServer(c.RestConf, rest.WithNotFoundHandler(gateway.NewPassHandler(authSecret, dsn)))
	defer server.Stop()

	fmt.Printf("Starting server at %s:%d...\n", c.Host, c.Port)
	server.Start()
}
