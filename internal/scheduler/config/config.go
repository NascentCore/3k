package config

import (
	"github.com/zeromicro/go-zero/rest"
)

type Config struct {
	rest.RestConf
	DB struct {
		DataSource string
	} `json:"DB,optional" `
	FinetuneModel map[string]struct {
		Image    string `json:"image"`
		GPUMem   int64  `json:"gpu_mem"`
		Command  string `json:"command"`
		ModelVol int64  `json:"model_vol"`
		GPUNum   int64  `json:"gpu_num"`
	} `json:"FinetuneModel"`
}
