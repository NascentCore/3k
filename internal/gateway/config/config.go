package config

import (
	"sxwl/3k/internal/gateway/gateway"

	"github.com/zeromicro/go-zero/rest"
)

type Config struct {
	rest.RestConf
	Gateway gateway.Config
}
