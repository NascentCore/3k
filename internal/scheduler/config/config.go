package config

import (
	"github.com/zeromicro/go-zero/rest"
)

type Config struct {
	rest.RestConf
	DB struct {
		DataSource string
	} `json:"-"`
	OSS struct {
		Bucket           string
		Endpoint         string
		PublicModelDir   string
		PublicDatasetDir string
		UserModelDir     string
		UserDatasetDir   string
	} `json:"OSS"`
	OSSAccess struct {
		UploadAccessID  string
		UploadAccessKey string
		AdminAccessID   string
		AdminAccessKey  string
	} `json:"-"`
	FinetuneModel map[string]struct {
		Image    string `json:"image"`
		GPUMem   int64  `json:"gpu_mem"`
		Command  string `json:"command"`
		ModelVol int64  `json:"model_vol"`
		GPUNum   int64  `json:"gpu_num"`
	} `json:"FinetuneModel"`
	Inference map[string]struct {
		UrlFormat string `json:"url_format"`
	} `json:"inference"`
}
